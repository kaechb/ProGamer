import math
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
from torch.optim.swa_utils import AveragedModel

from helpers import CosineWarmupScheduler, EqualLR, Scheduler, equal_lr
from pointflow import PF

sys.path.insert(1, "/home/kaechben/plots")
from functools import partial

from plotswb import *

from metrics import *
from models import *

# class

class ProGamer(pl.LightningModule):
    

    def __init__(self,num_batches=100,num_prog=1, **kwargs):
        
        """This initializes the model and its hyperparameters"""
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.kwargs=kwargs
        self.n_part=kwargs["n_part"]
        kwargs["activation"]="gelu"
        self.freq_d =kwargs["freq"]
        self.n_dim = kwargs["n_dim"]
        self.num_prog = num_prog
        self.lr_g = kwargs["lr_g"]
        self.lr_d = kwargs["lr_d"]     
        self.n_current = kwargs["n_start"]
        self.smart_batching=kwargs["smart_batching"]
        self.num_batches = int(num_batches)
        self.start_gen=False    
        self.d_losses=torch.ones(20)
        self.eval_=kwargs["eval"]
        self.gen_net = Gen(**kwargs)
        self.dis_net = Disc(**kwargs)
        self.beta1= kwargs["beta1"]
        self.beta2= kwargs["beta2"]
        self.sig = nn.Sigmoid()
        self.counter=0
        self.l_dim = kwargs["l_dim"]
        self.l_dim_gen = kwargs["l_dim_gen"]
        self.sched = kwargs["sched"]
        self.opt = kwargs["opt"]
        self.max_epochs = kwargs["max_epochs"]
        self.warmup = kwargs["warmup"]
        self.norm = kwargs["norm"]
        self.parton = kwargs["parton"]
        self.k=0
        self.gan=kwargs["gan"]
        print("training with ",self.n_current," particles")
        if self.gan=="ls":
            self.loss=nn.MSELoss()
        else:
            self.loss=nn.BCEWithLogitsLoss()
        for p in self.dis_net.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal(p)
        for p in self.gen_net.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal(p)
   
        
        for name, W in self.dis_net.named_parameters():
            if 'weight' in name:
                W.data = W.data*kwargs["num_layers"]**(-0.25)*0.67
        for name, W in self.gen_net.named_parameters():
            if 'weight' in name:
                W.data = W.data*kwargs["num_layers"]**(-0.25)*0.67
        torch.nn.init.normal_(self.dis_net.embbed.weight,mean=0,std=self.l_dim**(-0.5))
        torch.nn.init.normal_(self.gen_net.embbed.weight,mean=0,std=self.l_dim**(-0.5))
        if kwargs["swa"]:
            self.dis_net= AveragedModel(self.dis_net)
        if kwargs["swagen"]:
            self.gen_net= AveragedModel(self.gen_net)


    # #if self.kwargs["swa"]:
    #     self.dis_net= AveragedModel(self.dis_net)
    # #if self.kwargs["swagen"]:
    #     self.gen_net= AveragedModel(self.gen_net)

    def on_validation_epoch_start(self, *args, **kwargs):
        self.gen_net.train()
        self.dis_net.train()
        self.dis_net.cpu()
        self.gen_net.cpu()
        
    def on_validation_epoch_end(self, *args, **kwargs):

        self.gen_net = self.gen_net.to("cuda")
        self.dis_net = self.dis_net.to("cuda")
        self.gen_net.train()
        self.dis_net.train()

    def load_datamodule(self, data_module):
        """needed for lightning training to work, it just sets the dataloader for training and validation"""
        self.data_module = data_module

    def early_stopping(self,cov,fpndv,w1m_):
        if (cov>0.55 and w1m_<0.0015 and self.n_current<self.n_part) :
            self.n_current+=self.num_prog
            self.n_current=min(self.n_part,self.n_current)
            #self.start_gen = False
            if self.n_part!=self.n_current:
                self.counter=0 
            print("number particles set to to ",self.n_current)
   
    def calc_log_metrics(self, fake_scaled,true_scaled):
        #fake_scaled=torch.clamp(fake_scaled,min=-1,max=3)
        cov, mmd = cov_mmd( true_scaled,fake_scaled, use_tqdm=False)
        if self.n_current==30:
            try:
                fpndv = fpnd(fake_scaled[:50000,:].numpy(), use_tqdm=False, jet_type=self.parton)
            except:
                fpndv = 1001
        else:
            fpndv = 1000
        w1m_ = w1m(fake_scaled, true_scaled)[0]
        w1p_ = w1p(fake_scaled, true_scaled)[0]
        w1efp_ = w1efp(fake_scaled, true_scaled)[0]
        try:
            fgd_=fgd(true_scaled,fake_scaled,p=self.parton,n=self.n_current)[0]
        except:
            traceback.print_exc()
            fgd_=1000

        temp = {"val_fpnd": fpndv,"val_mmd": mmd,"val_cov": cov,"val_w1m": w1m_,
                "val_w1efp": w1efp_,"val_w1p": w1p_,"step": self.global_step,"fgd":fgd_}
        print("epoch {}: ".format(self.current_epoch), temp)
        self.log("hp_metric", w1m_,)
        self.log("w1m", w1m_,)
        self.log("w1p", w1p_,)
        self.log("w1efp", w1efp_,)
        self.log("cov", cov,  )
        self.log("fpnd", fpndv,  )
        self.log("fgd", fgd_,  )
        self.log("mmd", mmd,  )
        self.log("n_current",self.n_current)
        #if not self.smart_batching:
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
    def fade(self,mask):
        
        if self.n_current>30:
            mask[:,-self.num_prog:]=mask[:,-self.num_prog:]-100*torch.exp(-self.counter*torch.ones_like(mask[:,-self.num_prog:])/10)
        return mask.float()



    
    def sampleandscale(self, batch, mask=None, scale=False):
        """This is a helper function that samples from the flow (i.e. generates a new sample)
        and reverses the standard scaling that is done in the preprocessing. This allows to calculate the mass
        on the generative sample and to compare to the simulated one, we need to inverse the scaling before calculating the mass
        because calculating the mass is a non linear transformation and does not commute with the mass calculation"""
        assert (mask<=0).all()
        with torch.no_grad():
                z=torch.normal(torch.zeros(len(batch)*batch.shape[1],self.n_dim,device=batch.device),torch.ones(len(batch)*batch.shape[1],self.n_dim,device=batch.device)).reshape(len(batch),batch.shape[1],self.n_dim)
        fake=self.gen_net(z,mask=mask,)
        
        fake=z+fake
        assert (fake==fake).all()

        if scale:
            fake_scaled = self.data_module.scale(fake.clone(),mask)
            true = self.data_module.scale(batch.clone(),mask)
            fake_scaled[fake_scaled[:,:,2]<0]=0
            true[true[:,:,2]<0]=0
            assert (fake_scaled==fake_scaled).all()
            #fake_scaled=torch.where(fake_scaled[:,:,2]<0,fake_scaled[:,:,2],torch.tensor(0.).float())
           
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
            max_iter = self.max_epochs  * self.num_batches
            lr_scheduler_d = Scheduler(opt_d,dim_embed=self.l_dim, warmup_steps=self.warmup * self.num_batches )#15 // 3
            lr_scheduler_g = Scheduler(opt_g, dim_embed=self.l_dim, warmup_steps=self.warmup * self.num_batches )#  // 3
        else:
            raise
        return lr_scheduler_d,lr_scheduler_g

    def configure_optimizers(self):
        self.losses = []
        # mlosses are initialized with None during the time it is not turned on, makes it easier to plot
        if self.opt == "Adam":
            opt_g = torch.optim.Adam(self.gen_net.parameters(), lr=self.lr_g, betas=(0,0.9))
            opt_d = torch.optim.Adam(self.dis_net.parameters(), lr=self.lr_d,betas=(0, 0.9))# 
        elif self.opt == "RMSprop":
            opt_g = torch.optim.RMSprop(self.gen_net.parameters(), lr=self.lr_g)
            opt_d = torch.optim.RMSprop(self.dis_net.parameters(), lr=self.lr_d)
        else:
            raise
            opt_g = torch.optim.AdamW(self.gen_net.parameters(), lr=self.lr_g, betas=(self.beta1, self.beta2),weight_decay=1e-4)
            opt_d = torch.optim.AdamW(self.dis_net.parameters(), lr=self.lr_d, betas=(self.beta1, self.beta2),weight_decay=1e-4)
        if str(self.sched) != "None":
            lr_scheduler_d,lr_scheduler_g=self.scheduler(opt_d,opt_g)
            return [opt_d, opt_g], [lr_scheduler_d, lr_scheduler_g]
        else:
            return [ opt_d, opt_g]
    
    def train_disc(self,batch,mask,opt_d):

        with torch.no_grad():
            # self.dis_net.train()
            # if self.eval_:
            #     self.gen_net.eval()
            fake = self.sampleandscale(batch, mask, scale=False)
        target_real = torch.ones_like(batch[:,0,0].unsqueeze(-1))
        target_fake = torch.zeros_like(fake[:,0,0].unsqueeze(-1))
 
        if self.gan.find("wgan")==-1:
            target_real = torch.ones_like(batch[:,0,0].unsqueeze(-1))
            target_fake = torch.zeros_like(fake[:,0,0].unsqueeze(-1))
            pred=torch.cat((self.dis_net(batch,mask),self.dis_net(fake,mask,)),dim=0)  
            target = torch.vstack((target_real, target_fake))
            d_loss = self.loss(pred, target).mean()
        else:
            d_loss=-self.dis_net(batch,mask).mean()+self.dis_net(fake,mask).mean()
            if self.gan=="wgangp":
                d_loss+=self.gp(batch,fake)
        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()
        self.k+=1
        self.d_losses[self.k%20]=d_loss.detach()
        self.log("Training/d_loss", d_loss, logger=True, prog_bar=False,on_step=True)
        if self.gan=="ns":
                pred=self.sig(pred)          
                
        try:
            
            if self.current_epoch % self.kwargs["val_check"] == 0 and self.global_step%self.num_batches==1:
                    self.plot.plot_scores(pred[:len(batch)].detach().cpu().numpy(),pred[len(batch):].detach().cpu().numpy(),train=True,step=self.current_epoch)
        except:
            pass
        if self.global_step < 2:
            print("passed test disc")

    def train_gen(self,batch,mask,opt_g):
        # if self.eval_:
        #     self.gen_net.eval()
        # else:
        #     self.gen_net.train()
        # self.dis_net.train()
        opt_g.zero_grad()
        fake = self.sampleandscale(batch, mask, scale=False)
        pred = self.dis_net(fake, mask=mask)
        target = torch.ones_like(pred)
        if self.gan.find("wgan")==-1:
            g_loss=self.loss(pred,target).mean()
        else:
            g_loss=-pred.mean()

        self.manual_backward(g_loss) 
        opt_g.step()
        self.log("Training/g_loss", g_loss, logger=True, prog_bar=False,on_step=True)
        if self.global_step < 3:
            print("passed test gen")
        


    def training_step(self, batch, batch_idx):
        if self.kwargs["smart_batching"]:
            self.n_current=batch.shape[1]
        orig_mask = batch[:, :self.n_current,self.n_dim].bool()
        mask=orig_mask.clone().float().reshape(-1)
        """training loop of the model, here all the data is passed forward to a gaussian
        This is the important part what is happening here. This is all the training we do"""
        # assert batch.shape[1]==self.n_current
        # if self.smart_batching:
        #     self.n_current=batch.shape[1]

        with torch.no_grad():
            mask[mask>0]=-torch.inf
            mask=mask.reshape(-1,orig_mask.shape[1])
            #mask=self.fade(mask.float())
        
        batch = batch[:, :self.n_current,:self.n_dim]
        batch[orig_mask]=0
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
            self.counter=self.counter+1
            self.train_gen(batch,mask,opt_g)
            

    def validation_step(self, batch, batch_idx):
        """This calculates some important metrics on the hold out set (checking for overtraining)"""
        # if self.smart_batching:
        #     self.n_current=batch.shape[1]
        # if self.eval_:
        #     self.gen_net.eval()
        # else:
        #     self.gen_net.train()

        if self.kwargs["smart_batching"]:
            self.n_current=batch.shape[1]
        orig_mask = batch[:, :self.n_current,self.n_dim].cpu().bool()
        mask=orig_mask.clone().reshape(-1).float()
        with torch.no_grad():
            mask[mask>0]=-torch.inf
            mask=mask.reshape(-1,orig_mask.shape[1])
            #mask=self.fade(mask.float())
        assert (mask<=0).all()
        mask=mask.reshape(-1,orig_mask.shape[1])
        batch = batch[:, :self.n_current,:self.n_dim].cpu()
        mask_test=self.sample_n(mask)
        batch = batch.to("cpu") 
        with torch.no_grad():
            gen, fake_scaled, true_scaled = self.sampleandscale(batch,mask, scale=True)
            scores_real = self.dis_net(batch, mask=mask)
            scores_fake = self.dis_net(gen, mask=mask)  
            if self.gan=="ns":
                scores_real=self.sig(scores_real)          
                scores_fake=self.sig(scores_fake)          
        true_scaled[orig_mask]=0
        fake_scaled[orig_mask]=0
        assert (fake_scaled==fake_scaled).all()
        assert (true_scaled==true_scaled).all()
        # Reverse Standard Scaling (this has nothing to do with flows, it is a standard preprocessing step)
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

        
       
