import os
import sys
import time
import traceback

import matplotlib.pyplot as plt
import nflows as nf
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.autograd as autograd
from jetnet.evaluation import cov_mmd, fpnd, w1efp, w1m, w1p
from nflows.flows import base
from nflows.nn import nets
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import \
    PiecewiseRationalQuadraticCouplingTransform
from nflows.utils.torchutils import create_random_binary_mask
from torch import Tensor, nn
from torch.autograd import Variable
from torch.nn import TransformerEncoderLayer
from torch.nn import functional as FF
from torch.nn.functional import leaky_relu, sigmoid

from helpers import CosineWarmupScheduler, Scheduler, EqualLR,equal_lr

from pointflow import PF
from torch.optim.swa_utils import AveragedModel
sys.path.insert(1, "/home/kaechben/plots")
from functools import partial
from plotswb import *
from models import *
from metrics import *
# class

class ProGamer(pl.LightningModule):
    

    def __init__(self,num_batches=100, **kwargs):
        """This initializes the model and its hyperparameters"""
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.kwargs=kwargs
        self.n_part=kwargs["n_part"]
        self.flow_prior=kwargs["flow_prior"]
       
        self.freq_d =kwargs["freq"]
        self.n_dim = kwargs["n_dim"]

        self.latent = kwargs["latent"]
        if self.flow_prior:
            self.latent=False
            self.flow=PF.load_from_checkpoint(kwargs["load_ckpt"]).flow.eval()
        self.lr_g = kwargs["lr_g"]
        self.lr_d = kwargs["lr_d"]
        print("training with ",self.n_part," particles")
        self.n_current = kwargs["n_start"]
        self.smart_batching=kwargs["smart_batching"]
        self.num_batches = int(num_batches)
        self.start_gen=False
        self.add_corr = kwargs["add_corr"]
        self.d_losses=torch.ones(20)

        self.gen_net = Gen(**kwargs)
        self.dis_net = Disc(**kwargs)
        
        self.sig = nn.Sigmoid()
        self.counter=0
        self.l_dim = kwargs["l_dim"]
        self.l_dim_gen = kwargs["l_dim_gen"]
        self.sched = kwargs["sched"]
        self.opt = kwargs["opt"]
        self.max_epochs = kwargs["max_epochs"]
        self.warmup = kwargs["warmup"]
        self.aux = kwargs["aux"]
        self.parton = kwargs["parton"]
        self.k=0
        for p in self.dis_net.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal(p)
        for p in self.gen_net.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal(p)


    # #if self.kwargs["swa"]:
    #     self.dis_net= AveragedModel(self.dis_net)
    # #if self.kwargs["swagen"]:
    #     self.gen_net= AveragedModel(self.gen_net)

    def on_validation_epoch_start(self, *args, **kwargs):
        self.dis_net.train()
        self.gen_net.train()
        if self.flow_prior:
            self.flow.eval()
            self.flow = self.flow.to("cpu")
        self.dis_net = self.dis_net.cpu()
        self.gen_net = self.gen_net.cpu()

    def on_validation_epoch_end(self, *args, **kwargs):
        if self.flow_prior:
            self.flow = self.flow.to("cuda")
        self.gen_net = self.gen_net.to("cuda")
        self.dis_net = self.dis_net.to("cuda")

    def load_datamodule(self, data_module):
        """needed for lightning training to work, it just sets the dataloader for training and validation"""
        self.data_module = data_module

    def early_stopping(self,cov,fpndv,w1m_):
        if (cov>0.45 and w1m_<0.01 and self.n_current<30) or (self.n_current>30 and cov>0.4 and w1m_<0.006 and self.n_current<self.n_part) :
            self.n_current+=10
            self.n_current=min(self.n_part,self.n_current)
            self.start_gen = False
            self.counter=0
            print("number particles set to to ",self.n_current)
   
    def calc_log_metrics(self, fake_scaled,true_scaled):
        cov, mmd = cov_mmd( true_scaled,fake_scaled, use_tqdm=False)
        if self.n_current==30:
            fpndv = fpnd(fake_scaled[:50000,:].numpy(), use_tqdm=False, jet_type=self.parton)
        else:
            fpndv = 1000
        w1m_ = w1m(fake_scaled, true_scaled)[0]
        w1p_ = w1p(fake_scaled, true_scaled)[0]
        w1efp_ = w1efp(fake_scaled, true_scaled)[0]
        # try:
        #     fgd_=fgd(true_scaled,fake_scaled,p=self.parton,n=self.n_current)[0]
        # except:
        #     fgd_=1000

        temp = {"val_fpnd": fpndv,"val_mmd": mmd,"val_cov": cov,"val_w1m": w1m_,
                "val_w1efp": w1efp_,"val_w1p": w1p_,"step": self.global_step,}
        print("epoch {}: ".format(self.current_epoch), temp)
        self.log("hp_metric", w1m_,)
        self.log("w1m", w1m_,)
        self.log("w1p", w1p_,)
        self.log("w1efp", w1efp_,)
        self.log("cov", cov,  )
        self.log("fpnd", fpndv,  )
        self.log("mmd", mmd,  )
        self.log("n_current",self.n_current)
        if not self.smart_batching:
            self.early_stopping(cov,fpndv,w1m_)
        

    def sample_n(self, mask):
        #Samples a mask where the zero padded particles are True, rest False
        mask_test = torch.ones_like(mask)
        n, counts = np.unique(self.data_module.n, return_counts=True)
        counts_prob = torch.tensor(counts / len(self.data_module.n) )
        n_test=n[torch.multinomial(counts_prob,replacement=True,num_samples=(len(mask)))] 
        indices = torch.arange(self.n_current, device=mask.device)
        mask_test = (indices.view(1, -1) < torch.tensor(n_test).view(-1, 1))      
        mask_test=~mask_test.bool()
        return (mask_test)
    
    def sampleandscale(self, batch, mask=None, scale=False):
        """This is a helper function that samples from the flow (i.e. generates a new sample)
        and reverses the standard scaling that is done in the preprocessing. This allows to calculate the mass
        on the generative sample and to compare to the simulated one, we need to inverse the scaling before calculating the mass
        because calculating the mass is a non linear transformation and does not commute with the mass calculation"""
        assert mask.dtype==torch.bool
        with torch.no_grad():
            if self.flow_prior:
                z = self.flow.sample(len(batch)*batch.shape[1]).reshape(len(batch), 
                batch.shape[1], self.n_dim)
                
            # else:
                
            if self.config["latent"]:
                z=torch.normal(torch.zeros(len(batch),self.config["latent"],device=batch.device),torch.ones(len(batch),self.config["latent"],device=batch.device)).reshape(len(batch),self.config["latent"])
            else:
            
                if self.latent:
                    z=torch.normal(torch.zeros(len(batch),self.latent,device=batch.device),torch.ones(len(batch),self.latent,device=batch.device)).reshape(len(batch),self.latent)
                else:
                    z=torch.normal(torch.zeros(len(batch)*batch.shape[1],self.n_dim,device=batch.device),torch.ones(len(batch)*batch.shape[1],self.n_dim,device=batch.device)).reshape(len(batch),batch.shape[1],self.n_dim)
        fake=self.gen_net(z,mask=mask)
        if self.add_corr and not self.latent:
            fake=z+fake
  
        if scale:
            fake_scaled = fake.clone()
            true = batch.clone()
            self.data_module.scaler = self.data_module.scaler.to(batch.device)
            fake_scaled=self.data_module.scaler.inverse_transform(fake_scaled)
            true=self.data_module.scaler.inverse_transform(true)
            fake_scaled[mask]=0
            return fake, fake_scaled, true# z_scaled
        else:
            return fake

    def scheduler(self,opt_g,opt_d):
        if self.sched == "cosine":
            max_iter = (self.max_epochs) * self.num_batches -self.global_step
            lr_scheduler_d = CosineWarmupScheduler(opt_d, warmup=self.warmup * self.num_batches, max_iters=max_iter)
            lr_scheduler_g = CosineWarmupScheduler(opt_g, warmup=self.warmup * self.num_batches, max_iters=max_iter)
        elif self.sched == "cosine2":         
            max_iter = (self.max_epochs) * self.num_batches//3-self.global_step
            lr_scheduler_d = CosineWarmupScheduler(opt_d, warmup=self.warmup * self.num_batches, max_iters=max_iter )#15 // 3
            lr_scheduler_g = CosineWarmupScheduler(opt_g, warmup=self.warmup * self.num_batches , max_iters=max_iter)#  // 3
        elif self.sched == "linear":
            max_iter = (self.max_epochs // 2) * self.num_batches-self.global_step
            lr_scheduler_d = Scheduler(opt_d,dim_embed=self.l_dim, warmup_steps=self.warmup * self.num_batches )#15 // 3
            lr_scheduler_g = Scheduler(opt_g, dim_embed=self.l_dim, warmup_steps=self.warmup * self.num_batches )#  // 3
        else:
            lr_scheduler_d = None
            lr_scheduler_g = None
        return lr_scheduler_d,lr_scheduler_g

    def configure_optimizers(self):
        self.losses = []
        # mlosses are initialized with None during the time it is not turned on, makes it easier to plot
        if self.opt == "Adam":
            opt_g = torch.optim.Adam(self.gen_net.parameters(), lr=self.lr_g, betas=(0, 0.99))
            opt_d = torch.optim.Adam(self.dis_net.parameters(), lr=self.lr_d,betas=(0, 0.99))# 
        else:
            opt_g = torch.optim.RMSprop(self.gen_net.parameters(), lr=self.lr_g)
            opt_d = torch.optim.RMSprop(self.dis_net.parameters(), lr=self.lr_d)
        if str(self.sched) != "None":
            lr_scheduler_d,lr_scheduler_g=self.scheduler(opt_d,opt_g)
            return [opt_d, opt_g], [lr_scheduler_d, lr_scheduler_g]
        else:
            return [ opt_d, opt_g]

    def train_disc(self,batch,mask,opt_d):
        with torch.no_grad():
            fake = self.sampleandscale(batch, mask, scale=False)
            target_real = torch.ones_like(batch[:,0,0].unsqueeze(-1))
            target_fake = torch.zeros_like(fake[:,0,0].unsqueeze(-1))
        
        # pred_real = self.dis_net(batch, mask=mask,aux=self.aux)
        # pred_fake = self.dis_net(fake.detach(), mask=mask,aux=self.aux)
        # if self.aux:
        #     m=torch.cat((mass(batch),mass(fake.detach())),dim=0)
        #     p=torch.cat((batch[:,:,-1].sum(1),fake[:,:,-1].detach().sum(1)),dim=0)
        #     m_loss = nn.MSELoss()(m.reshape(-1),torch.cat((pred_real[1],pred_fake[1]),dim=0).reshape(-1))
        #     self.log("Training/m_loss",m_loss,logger=True,prog_bar=False,on_step=True)
        #     pred_real,pred_fake=pred_real[0],pred_fake[0]
        # #p_loss = nn.MSELoss()(p.reshape(-1),torch.cat((p_t,p_f),dim=0).reshape(-1))
        pred=self.dis_net(torch.cat((batch,fake),dim=0),torch.cat((mask,mask),dim=0))
        #pred = torch.vstack((pred_real, pred_fake))
        target = torch.vstack((target_real, target_fake))
        d_loss = nn.MSELoss()(pred, target).mean()
        # if self.aux and  m_loss==m_loss:#+0.01*p_loss
            
        #         d_loss+=0.01*m_loss 
        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()
        # if self.aux and m_loss==m_loss:
        #     d_loss-=0.01*m_loss#+0.01*p_loss
        self.k+=1
        self.d_losses[self.k%20]=d_loss.detach()
        if self.d_losses.mean()<0.001 and self.freq_d!=1:
            print("freq set to 1")
            self.freq_d=1
        elif self.d_losses.mean()>0.2 and self.freq_d!=5:
            print("freq set to 5")

            self.freq_d=5
        self.log("Training/d_loss", d_loss, logger=True, prog_bar=False,on_step=True)
        #self.log("Training/p_loss",p_loss,logger=True,prog_bar=False,on_step=True)
        try:
            if self.current_epoch % self.kwargs["val_check"] == 0 and self.global_step%self.num_batches==1:
                    self.plot.plot_scores(pred[:len(batch)].detach().cpu().numpy(),pred[len(batch):].detach().cpu().numpy(),train=True,step=self.current_epoch)
        except:
            pass
        assert d_loss==d_loss
        if self.global_step < 2:
            print("passed test disc")

    def train_gen(self,batch,mask,opt_g):
        opt_g.zero_grad()
        fake = self.sampleandscale(batch, mask, scale=False)
        pred = self.dis_net(fake, mask=mask)
        target = torch.ones_like(pred)
        g_loss = nn.MSELoss()(pred, target).mean()
        self.manual_backward(g_loss)  
        if self.global_step > 10:
            opt_g.step()
        else:
            opt_g.zero_grad()
        self.log("Training/g_loss", g_loss, logger=True, prog_bar=False,on_step=True)
        if self.global_step < 3:
            print("passed test gen")


    def training_step(self, batch, batch_idx):
        
        """training loop of the model, here all the data is passed forward to a gaussian
        This is the important part what is happening here. This is all the training we do"""
        # assert batch.shape[1]==self.n_current
        if self.smart_batching:
            self.n_current=batch.shape[1]
        mask = batch[:, :self.n_current,self.n_dim].bool()
        batch = batch[:, :self.n_current,:self.n_dim]
        batch[mask]=0
        opt_d, opt_g = self.optimizers()
        if str(self.sched) != "None":
            sched_d, sched_g = self.lr_schedulers()
            sched_d.step() 
            sched_g.step() 
            self.log("Training/lr_g", sched_g.get_last_lr()[-1])
            self.log("Training/lr_d", sched_d.get_last_lr()[-1])
        ### GAN PART
        self.train_disc(batch,mask,opt_d)    
        if not self.start_gen and self.d_losses.mean()<0.1:
            self.start_gen=True
            self.log("n_current",self.n_current)
            print("start training gen with {} particles".format(self.n_current))
        if (self.start_gen and self.global_step%self.freq_d<2) or self.global_step <= 3:
            self.train_gen(batch,mask,opt_g)
            

    def validation_step(self, batch, batch_idx):
        """This calculates some important metrics on the hold out set (checking for overtraining)"""
        if self.smart_batching:
            self.n_current=batch.shape[1]
        mask = batch[:, :self.n_current,self.n_dim].bool().cpu()
        batch = batch[:, :self.n_current,:self.n_dim].cpu()
        mask_test=self.sample_n(mask).bool()
        batch = batch.to("cpu") 
        with torch.no_grad():
            gen, fake_scaled, true_scaled = self.sampleandscale(batch,mask, scale=True)
            scores_real = self.dis_net(batch, mask=mask)
            scores_fake = self.dis_net(gen, mask=mask)            
        true_scaled[mask]=0
        # Reverse Standard Scaling (this has nothing to do with flows, it is a standard preprocessing step)
        for i in range(self.n_current):
            fake_scaled[fake_scaled[:, i,2] < 0, i,2] = 0
        self.plot = plotting_point_cloud(model=self,gen=fake_scaled,true=true_scaled,config=self.kwargs,step=self.global_step,logger=self.logger, n=self.n_current,p=self.parton)#,nf=z_scaled.reshape(-1,self.n_current,self.n_dim)
    #self.plot.plot_mom(self.global_step)
        try:
            self.plot.plot_mass(save=None, bins=50)
            self.plot.plot_scores(scores_real.reshape(-1).detach().cpu().numpy(), scores_fake.reshape(-1).detach().cpu().numpy(), train=False, step=self.current_epoch)
        except Exception as e:
            traceback.print_exc()
        self.calc_log_metrics(fake_scaled,true_scaled)
        # if self.global_step==0:
        #     self.data_module.setup("train",self.n_current)

        
       
