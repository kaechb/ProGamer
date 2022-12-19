


import os
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torch.nn import functional as FF
from torch.nn.functional import leaky_relu, sigmoid

from jetnet.evaluation import cov_mmd, fpnd, w1efp, w1m, w1p

import nflows as nf
from nflows.flows import base
from nflows.nn import nets
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import \
    PiecewiseRationalQuadraticCouplingTransform
from nflows.utils.torchutils import create_random_binary_mask

import sys 
sys.path.insert(1,"/home/kaechben/plots")
from plotswb import *

class PF(pl.LightningModule):
    def create_resnet(self, in_features, out_features):
        """This is the network that outputs the parameters of the invertible transformation
        The only arguments can be the in dimension and the out dimenson, the structure
        of the network is defined over the config which is a class attribute
        Context Features: Amount of features used to condition the flow - in our case
        this is usually the mass
        num_blocks: How many Resnet blocks should be used, one res net block is are 1 input+ 2 layers
        and an additive skip connection from the first to the third"""
       
        return nets.ResidualNet(
            in_features,
            out_features,
            hidden_features=self.config["network_nodes_nf"],
            context_features=None,
            num_blocks=self.config["network_layers_nf"],
            activation=self.config["activation"] if "activation" in self.config.keys() else FF.relu,
            dropout_probability=self.config["dropout"] if "dropout" in self.config.keys() else 0,
            use_batch_norm=0,
        )

    def __init__(self, config, num_batches):
        """This initializes the model and its hyperparameters"""
        super().__init__()
        self.config = config
        # Loss function of the Normalizing flows
        self.save_hyperparameters()
        self.n_dim=config["n_dim"]
        self.n_part=config["n_part"]
        self.alpha = 1
        self.num_batches = int(num_batches)
        self.start = time.time()
        self.config = config
        self.automatic_optimization = False
        # Loss function of the Normalizing flows
        self.logprobs = []
        self.n_part = config["n_part"]
        self.save_hyperparameters()
        self.flows = []
        self.fpnds = []
        self.w1ms = []
        self.n_dim = self.config["n_dim"]
        self.n_part = config["n_part"]
        self.num_batches = int(num_batches)
        self.build_flow()
        self.df = pd.DataFrame()
        self.error=0
        self.nf=True
        
        
    def load_datamodule(self, data_module):
        """needed for lightning training to work, it just sets the dataloader for training and validation"""
        self.data_module = data_module



    
    
    def sample_n(self, mask):
        #Samples a mask where the zero padded particles are True, rest False
        mask_test = torch.ones_like(mask)
        n, counts = np.unique(self.data_module.n, return_counts=True)
        counts_prob = torch.tensor(counts / len(self.data_module.n) )
        n_test=n[torch.multinomial(counts_prob,replacement=True,num_samples=(len(mask)))] 
        indices = torch.arange(30, device=mask.device)
        mask_test = (indices.view(1, -1) < torch.tensor(n_test).view(-1, 1))      
        mask_test=~mask_test.bool()
        return (mask_test)
    
    def build_flow(self):
        K = self.config["coupling_layers"]
        for i in range(K):
            """This creates the masks for the coupling layers, particle masks are masks
            created such that each feature particle (eta,phi,pt) is masked together or not"""
            mask = create_random_binary_mask(self.n_dim)
            self.flows += [PiecewiseRationalQuadraticCouplingTransform(mask=mask,transform_net_create_fn=self.create_resnet, tails="linear",tail_bound=self.config["tail_bound"],num_bins=self.config["bins"],)]
        self.q0 = nf.distributions.normal.StandardNormal([self.n_dim])
        # Creates working flow model from the list of layer modules
        self.flows = CompositeTransform(self.flows)
        # Construct flow model
        self.flow = base.Flow(distribution=self.q0, transform=self.flows)

    

    def configure_optimizers(self):
        self.losses = []
        # mlosses are initialized with None during the time it is not turned on, makes it easier to plot
        opt_nf = torch.optim.Adam(self.flow.parameters(), lr=self.config["lr_nf"])
        return  [opt_nf]

    def nf_loss(self,batch):
        nf_loss = -self.flow.to(self.device).log_prob(batch).mean()
        nf_loss /= self.n_dim
        return nf_loss


    def training_step(self, batch, batch_idx):
        """training loop of the model, here all the data is passed forward to a gaussian
        This is the important part what is happening here. This is all the training we do"""
        opt_nf= self.optimizers()
        mask = batch[:, :,self.n_dim].bool()
        batch = batch[:,:,:self.n_dim:]
        flat_batch =batch.reshape(-1,self.n_dim)
        mask=mask.reshape(-1)
        flat_batch=flat_batch[~mask,:]
        nf_loss=self.nf_loss(flat_batch)      
        opt_nf.zero_grad()
        self.manual_backward(nf_loss)
        opt_nf.step()
        self.log("logprob", nf_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        
              

    def validation_step(self, batch, batch_idx):
        """This calculates some important metrics on the hold out set (checking for overtraining)"""
        self.flow = self.flow.to("cpu")
        self.flow.eval()
        batch = batch.to("cpu")
        mask = batch[:, :,self.n_dim].cpu().bool().reshape(-1)
        batch = batch[:,:,:self.n_dim].cpu()
        flat_batch=batch.reshape(len(batch)*self.n_part,self.n_dim)
        empty = torch.zeros_like(flat_batch) 
        with torch.no_grad():
            empty[~mask]=self.flow.sample(len(flat_batch[~mask]))
            logprob = -self.flow.log_prob(flat_batch[~mask]).mean() / self.n_dim
            self.data_module.scaler = self.data_module.scaler.to(batch.device)
            empty[~mask]=self.data_module.scaler.inverse_transform(empty[~mask])
            flat_batch[~mask]=self.data_module.scaler.inverse_transform(flat_batch[~mask] )
            flat_batch[mask]=0
            true_scaled=flat_batch.reshape(len(batch),self.n_part, self.n_dim)
            
            z_scaled=empty.reshape(len(batch),self.n_part,self.n_dim)

        m_t = mass(true_scaled[:,:self.n_part,:self.n_dim])
        m_c = mass(z_scaled[:,:self.n_part,:self.n_dim])
        
        for i in range(self.n_part):
            z_scaled[z_scaled[:, i,2] < 0, i,2] = 0
        # Some metrics we track
        cov, mmd = cov_mmd( true_scaled,z_scaled, use_tqdm=False)
        if self.n_part==30:
            
            fpndv = fpnd(z_scaled[:50000,:].numpy(), use_tqdm=False, jet_type=self.config["parton"])
        else:
            fpndv = 1000

        w1m_ = w1m(z_scaled, true_scaled)[0]
        w1p_ = w1p(z_scaled, true_scaled)[0]
        w1efp_ = w1efp(z_scaled, true_scaled)[0]

        self.w1ms.append(w1m_)
        self.fpnds.append(fpndv)
        temp = {"val_logprob": float(logprob.numpy()),"val_fpnd": fpndv,"val_mmd": mmd,"val_cov": cov,"val_w1m": w1m_,
                "val_w1efp": w1efp_,"val_w1p": w1p_,"step": self.global_step,}
        print("epoch {}: ".format(self.current_epoch), temp)
      
        self.log("hp_metric", w1m_, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("w1m", w1m_, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("w1p", w1p_, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("w1efp", w1efp_, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("logprob", logprob, prog_bar=True, logger=True)
        self.log("cov", cov, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("fpnd", fpndv, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("mmd", mmd, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.plot = plotting_point_cloud(model=self,gen=z_scaled,true=true_scaled,config=self.config,step=self.global_step,n=self.n_part,
            logger=self.logger, p=self.config["parton"])
        #self.plot.plot_mom(self.global_step)
        try:
            
            self.plot.plot_mass( save=None, bins=50)
            # self.plot.plot_2d(save=True)
        #     self.plot.var_part(true=true[:,:self.n_dim],gen=gen_corr[:,:self.n_dim],true_n=n_true,gen_n=n_gen_corr,
        #                          m_true=m_t,m_gen=m_test ,save=True)
        except Exception as e:
            traceback.print_exc()
        self.flow = self.flow.to("cuda")