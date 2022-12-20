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
from torch.nn import functional as FF
from torch.nn.functional import leaky_relu, sigmoid

from helpers import CosineWarmupScheduler, Scheduler

from torch.optim.swa_utils import AveragedModel
sys.path.insert(1, "/home/kaechben/plots")
from functools import partial
from plotswb import *
from models import *
from metrics import *
# class

class JetFlow(pl.LightningModule):
    def create_resnet(self,in_features, out_features):
        '''This is the network that outputs the parameters of the invertible transformation
        The only arguments can be the in dimension and the out dimenson, the structure
        of the network is defined over the config which is a class attribute
        Context Features: Amount of features used to condition the flow - in our case 
        this is usually the mass
        num_blocks: How many Resnet blocks should be used, one res net block is are 1 input+ 2 layers
        and an additive skip connection from the first to the third'''
        c=self.config["context_features"]
        return nets.ResidualNet(
                in_features,
                out_features,
                hidden_features=self.config["network_nodes"],
                context_features=c,
                num_blocks=self.config["network_layers"],
                activation=self.config["activation"]  if "activation" in self.config.keys() else FF.relu,
                dropout_probability=self.config["dropout"] if "dropout" in self.config.keys() else 0,
                use_batch_norm=self.config["batchnorm"] if "batchnorm" in self.config.keys() else 0,
                    )

    def __init__(self, config, num_batches,path="/"):
        """This initializes the model and its hyperparameters"""
        super().__init__()
        self.config = config
        self.automatic_optimization = False
        self.train_gan=config["train_gan"]
        # Loss function of the Normalizing flows
        self.flows = []
        self.save_hyperparameters()
        self.n_dim = self.config["n_dim"]
        self.n_part = config["n_part"]
        self.num_batches = int(num_batches)
        self.n_current=config["n_start"]
        K=self.config["coupling_layers"]
        for i in range(K):
            '''This creates the masks for the coupling layers, particle masks are masks
            created such that each feature particle (eta,phi,pt) is masked together or not'''
            mask=create_random_binary_mask(self.n_dim*self.n_part)  
            #Here are the coupling layers of the flow. There seem to be 3 choices but actually its more or less only 2
            #The autoregressive one is incredibly slow while sampling which does not work together with the constraint
            self.flows += [PiecewiseRationalQuadraticCouplingTransform(
                mask=mask,
                transform_net_create_fn=self.create_resnet, 
                tails='linear',
                tail_bound=self.config["tail_bound"],
                num_bins=self.config["bins"] )]
        #This sets the distribution in the latent space on which we want to morph onto        
        self.q0 = nf.distributions.normal.StandardNormal([self.n_dim*self.n_part])
        self.q_test =nf.distributions.normal.StandardNormal([self.n_dim*self.n_part])
        #Creates working flow model from the list of layer modules
        self.flows=CompositeTransform(self.flows)
        # Construct flow model
        self.flow = base.Flow(distribution=self.q0, transform=self.flows)
        # if self.config["swa"]:
        #     self.flow= AveragedModel(self.flow)
        if self.train_gan=="dis":
            self.dis_net=Disc()
        elif self.train_gan=="ref":
            self.dis_net=Disc()
            self.gen_net=Gen()
        
        

    def on_validation_epoch_start(self, *args, **kwargs):
        self.flow.eval()
        self.flow = self.flow.to("cpu")


    def on_validation_epoch_end(self, *args, **kwargs):
        self.flow = self.flow.to("cuda")


    def load_datamodule(self, data_module):
        """needed for lightning training to work, it just sets the dataloader for training and validation"""
        self.data_module = data_module


    def calc_log_metrics(self, fake_scaled,z_scaled,true_scaled):
        cov, mmd = cov_mmd( true_scaled,fake_scaled, use_tqdm=False)
        cov_nf, mmd_nf = cov_mmd(true_scaled,z_scaled,  use_tqdm=False)
        if self.n_current==30:
            fpndv = fpnd(fake_scaled[:50000,:].numpy(), use_tqdm=False, jet_type=self.config["parton"])
        else:
            fpndv = 1000
        w1m_ = w1m(fake_scaled, true_scaled)[0]
        w1p_ = w1p(fake_scaled, true_scaled)[0]
        w1efp_ = w1efp(fake_scaled, true_scaled)[0]
        temp = {"fpnd": fpndv,"mmd": mmd,"cov": cov,"w1m": w1m_, "w1efp": w1efp_,"w1p": w1p_,"step": self.global_step,}
        print("epoch {}: ".format(self.current_epoch), temp)
        self.log("hp_metric", w1m_,)
        self.log("w1m", w1m_,)
        self.log("w1p", w1p_,)
        self.log("w1efp", w1efp_,)
        self.log("cov", cov,  )
        self.log("cov_nf", cov_nf,  )
        self.log("fpnd", fpndv,  )
        self.log("mmd", mmd,  )
        self.log("n_current",self.n_current)
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

    def sample(self):
        if self.config["context_features"]>0:            
            fake=self.flow.sample(1,c).reshape(len(batch),self.n_part,self.n_dim)
        else:
            fake=self.flow.sample(len(batch)).reshape(len(batch),self.n_part,self.n_dim)
        return fake
    def sampleandscale(self, batch,c=None,  scale=False):
        """This is a helper function that samples from the flow (i.e. generates a new sample)
        and reverses the standard scaling that is done in the preprocessing. This allows to calculate the mass
        on the generative sample and to compare to the simulated one, we need to inverse the scaling before calculating the mass
        because calculating the mass is a non linear transformation and does not commute with the mass calculation"""
        self.flow.eval()

        with torch.no_grad():
            fake=self.sample()
            
            
        if scale:
            fake_scaled = fake.clone()
            true = batch.clone()
            self.data_module.scaler = self.data_module.scaler.to(batch.device)
            fake_scaled=self.data_module.scaler.inverse_transform(fake_scaled)
            true=self.data_module.scaler.inverse_transform(true)
            return fake, fake_scaled, true, z_scaled
        else:
            return fake

    def scheduler(self,opt):
        max_iter = (self.config["max_epochs"]) * self.num_batches-self.global_step
        lr_scheduler = CosineWarmupScheduler(opt, warmup=self.config["warmup"] * self.num_batches, max_iters=max_iter)
        
        if self.config["sched"] == "linear":
            lr_scheduler = Scheduler(opt,dim_embed=self.config["l_dim"], warmup_steps=self.config["warmup"] * self.num_batches )#15 // 3
        else:
            lr_scheduler = None
        return lr_scheduler

    def configure_optimizers(self):
        if not self.train_gan:
            if self.config["opt"] == "Adam":
                opt = torch.optim.Adam(self.flow.parameters(), lr=self.config["lr_f"])  
                
            else:
                opt = torch.optim.RMSprop(self.flow.parameters(), lr=self.config["lr_f"])
            if self.config["sched"] != None:
                lr_scheduler=self.scheduler(opt)
                return [opt,[],[]], [lr_scheduler,[],[]]
            else:
                return [ opt,[],[]]
        elif self.train_gan=="dis":
            if self.config["opt"] == "Adam":
                opt = torch.optim.Adam(self.flow.parameters(), lr=self.config["lr_f"])  
                opt_d = torch.optim.Adam(self.dis_net.parameters(), lr=self.config["lr_d"],betas=(0, 0.9))# 
            else:
                opt = torch.optim.RMSprop(self.flow.parameters(), lr=self.config["lr_f"])
                opt_d = torch.optim.RMSprop(self.dis_net.parameters(), lr=self.config["lr_d"])
            
            if self.config["sched"] != None:
                lr_scheduler=self.scheduler(opt)
                lr_scheduler_d=self.scheduler(opt_d)
                return [opt,opt_d,[]], [lr_scheduler,lr_scheduler_d,[]]
            else:
                return [ opt,opt_d,[]]
        elif self.train_gan=="ref":
            if self.config["opt"] == "Adam":
                opt = torch.optim.Adam(self.flow.parameters(), lr=self.config["lr_f"]) 
                opt_g = torch.optim.Adam(self.gen_net.parameters(), lr=self.config["lr_g"], betas=(0, 0.9)) 
                opt_d = torch.optim.Adam(self.dis_net.parameters(), lr=self.config["lr_d"],betas=(0, 0.9))# 
            else:
                opt = torch.optim.RMSprop(self.flow.parameters(), lr=self.config["lr_f"])
                opt_g = torch.optim.RMSprop(self.gen_net.parameters(), lr=self.config["lr_g"])
                opt_d = torch.optim.RMSprop(self.dis_net.parameters(), lr=self.config["lr_d"])
            
            if self.config["sched"] != None:
                lr_scheduler=self.scheduler(opt)
                lr_scheduler_g=self.scheduler(opt_g)
                lr_scheduler_d=self.scheduler(opt_d)
                return [opt,opt_d,opt_g], [lr_scheduler,lr_scheduler_d,lr_scheduler_g]
            else:
                return [opt,opt_d,opt_g]

    def train_gan(self,batch,c=None, mask=None):
        opt_d,opt_g=self.optimizers
        with torch.no_grad():
            fake = self.sampleandscale(batch,c, mask, scale=False)
        pred_real = self.dis_net(batch, mask=mask,aux=self.config["aux"])
        pred_fake = self.dis_net(fake.detach(), mask=mask,aux=self.config["aux"])
    
        target_real = torch.ones_like(pred_real)
        target_fake = torch.zeros_like(pred_fake)
        pred = torch.vstack((pred_real, pred_fake))
        target = torch.vstack((target_real, target_fake))
        d_loss = nn.MSELoss()(pred, target).mean()
        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()
        self.d_losses[self.global_step%5]=d_loss.detach()
        self.log("Training/d_loss", d_loss, logger=True, prog_bar=False,on_step=True)
        try:
            if self.current_epoch % 25 == 0 and self.global_step%self.num_batches<3:
                    self.plot.plot_scores(pred_real.detach().cpu().numpy(),pred_fake.detach().cpu().numpy(),train=True,step=self.current_epoch)
        except:
            pass
        if self.global_step < 2:
            print("passed test disc")
        if self.global_step%self.freq_d<2:
            if self.train_gan=="ref":
                opt_g.zero_grad()
                fake = self.sampleandscale(batch, c,mask, scale=False)
                pred = self.dis_net(fake, mask=mask)
                target = torch.ones_like(pred)
                g_loss = nn.MSELoss()(pred, target).mean()
                self.manual_backward(g_loss)
                opt_g.step()
                self.log("Training/g_loss", g_loss, logger=True, prog_bar=False,on_step=True)

            else:
                opt.zero_grad()
                fake = self.sampleandscale(batch,c, mask, scale=False)
                pred = self.dis_net(fake, mask=mask)
                target = torch.ones_like(pred)
                g_loss = nn.MSELoss()(pred, target).mean()
                self.manual_backward(g_loss)
                opt.step()



    def training_step(self, batch, batch_idx):
        opt=self.optimizers()[0]
        mask=batch[:,:,-1]
        x= batch[:,:,:-1]
        if self.config["context_features"]>0:
            c=c[:,0].reshape(-1,self.config["context_features"])
        elif self.config["context_features"]==0:
            c=None
        self.opt_g.zero_grad()

        g_loss = -self.flow.to(self.device).log_prob(x.reshape(-1,(self.n_dim*self.n_part)),c if self.config["context_features"] else None).mean()/(self.n_dim*self.n_part)
        self.log("logprob", g_loss, on_step=True, on_epoch=False, logger=True) 
        self.losses.append(g_loss.detach().cpu().numpy())
        if self.gan_corr:
            self.train_gan(batch,mask)
            

    def validation_step(self, batch, batch_idx):
        """This calculates some important metrics on the hold out set (checking for overtraining)"""
        #print("start val")
        # assert batch.shape[1]==self.n_current
        mask = batch[:, :self.n_current,self.n_dim].bool().cpu()
        batch = batch[:, :self.n_current,:self.n_dim].cpu()
        mask_test=self.sample_n(mask).bool()
        batch = batch.to("cpu") 
        with torch.no_grad():
            gen, fake_scaled, true_scaled, z_scaled = self.sampleandscale(batch,mask, scale=True)
            batch[mask]=0
            scores_real = self.dis_net(batch, mask=mask)
            scores_fake = self.dis_net(gen, mask=mask)            
        true_scaled[mask]=0
        # Reverse Standard Scaling (this has nothing to do with flows, it is a standard preprocessing step)
        for i in range(self.n_current):
            fake_scaled[fake_scaled[:, i,2] < 0, i,2] = 0
            z_scaled[z_scaled[:, i,2] < 0, i,2] = 0
        self.plot = plotting_point_cloud(model=self,gen=fake_scaled.reshape(-1,self.n_current,self.n_dim),true=true_scaled.reshape(-1,self.     n_current,self.n_dim),config=self.config,step=self.global_step,logger=self.logger, n=self.n_current,p=self.config["parton"])#,nf=z_scaled.reshape(-1,self.n_current,self.n_dim)

        try:
            self.plot.plot_mass(save=None, bins=50)
            self.plot.plot_scores(scores_real.reshape(-1).detach().cpu().numpy(), scores_fake.reshape(-1).detach().cpu().numpy(), train=False, step=self.current_epoch)
        except Exception as e:
            traceback.print_exc()
        
        self.calc_log_metrics(fake_scaled,z_scaled,true_scaled)

        
       
