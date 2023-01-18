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
import torch.nn.functional as F
from helpers import CosineWarmupScheduler, Scheduler

from torch.optim.swa_utils import AveragedModel
sys.path.insert(1, "/home/kaechben/plots")
from functools import partial
from plotswb import *
from models import *
from metrics import *
# class
class EquiBlock(nn.Module):
    """A general-purpose residual block. Works only with 1-dim inputs."""

    def __init__(
        self,
        features,
        context_features,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        zero_initialization=True,
    ):
        super().__init__()
        self.activation = activation

        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList(
                [nn.LayerNorm(features, eps=1e-3) for _ in range(2)]
            )
        if context_features is not None:
            self.context_layer = nn.Linear(context_features, features)
        self.linear_layers = nn.ModuleList(
            [nn.Linear(features, 10*features),nn.Linear(10*features, features)]
        )
        self.attn=nn.MultiheadAttention(features,num_heads=1,batch_first=True)
        self.dropout = nn.Dropout(p=dropout_probability)
        if zero_initialization:
            nn.init.uniform_(self.linear_layers[-1].weight, -1e-3, 1e-3)
            nn.init.uniform_(self.linear_layers[-1].bias, -1e-3, 1e-3)

    def forward(self, inputs, context=None):
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps=self.attn(temps,temps,temps)[0]
        temps = self.linear_layers[0](temps)
        temps = self.activation(temps)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.linear_layers[1](temps)
        if context is not None:
            temps = F.glu(torch.cat((temps, self.context_layer(context)), dim=1), dim=1)
        return inputs + temps


class EquiNet(nn.Module):
    """A general-purpose residual network. Works only with 1-dim inputs."""

    def __init__(
        self,
        in_features,
        out_features,
        hidden_features,
        context_features=None,
        num_blocks=2,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
    ):
        super().__init__()
        self.hidden_features = hidden_features
        self.context_features = context_features
        if context_features is not None:
            self.initial_layer = nn.Linear(
                in_features + context_features, hidden_features
            )
        else:
            self.initial_layer = nn.Linear(3, hidden_features)
        self.blocks = nn.ModuleList(
            [
                EquiBlock(
                    features=hidden_features,
                    context_features=context_features,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=False#use_batch_norm,
                )
                for _ in range(num_blocks)
            ]
        )
        self.reduce = nn.Linear(2*hidden_features,hidden_features)
        self.final_layer = nn.Linear(hidden_features, out_features)
        self.token = nn.Parameter()
    def forward(self, inputs, context=None):
        inputs=inputs.reshape(inputs.shape[0],15,-1)
        if context is None:
            temps = self.initial_layer(inputs)
        else:
            temps = self.initial_layer(torch.cat((inputs, context), dim=1))
        for block in self.blocks:
            temps = block(temps, context=context)
        temps = self.reduce(torch.cat((F.max_pool1d(temps.permute(0,2,1),15),F.avg_pool1d(temps.permute(0,2,1),15)),axis=1).squeeze(-1))
        outputs = self.final_layer(temps)
        return outputs

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
    def create_equinet(self,in_features, out_features):
            c=self.config["context_features"]
            return EquiNet(
                in_features,
                out_features,
                hidden_features=self.config["network_nodes"],
                context_features=None,#c,
                num_blocks=self.config["network_layers"],
                activation= F.relu,
                dropout_probability=self.config["dropout"] if "dropout" in self.config.keys() else 0,
                use_batch_norm=False,
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
        for i in range(self.config["coupling_layers"]):
            '''This creates the masks for the coupling layers, particle masks are masks
            created such that each feature particle (eta,phi,pt) is masked together or not'''
            mask=create_random_binary_mask(self.n_part)  
            mask=torch.repeat_interleave(mask,self.n_dim,0)
            #Here are the coupling layers of the flow. There seem to be 3 choices but actually its more or less only 2
            #The autoregressive one is incredibly slow while sampling which does not work together with the constraint
            self.flows += [PiecewiseRationalQuadraticCouplingTransform(
                mask=mask,
                transform_net_create_fn=self.create_equinet, 
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

        

    def on_validation_epoch_start(self, *args, **kwargs):
        self.flow.eval()
        self.flow = self.flow.to("cpu")


    def on_validation_epoch_end(self, *args, **kwargs):
        self.flow = self.flow.to("cuda")


    def load_datamodule(self, data_module):
        """needed for lightning training to work, it just sets the dataloader for training and validation"""
        self.data_module = data_module


    def calc_log_metrics(self, fake_scaled,true_scaled):
        cov, mmd = cov_mmd( true_scaled,fake_scaled, use_tqdm=False)

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

        self.log("fpnd", fpndv,  )
        self.log("mmd", mmd,  )
        self.log("n_current",self.n_current)
       # self.early_stopping(cov,fpndv,w1m_)
        

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

    def sample(self,batch):
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
            fake=self.sample(batch)
            
            
        if scale:
            fake_scaled = fake.clone()
            true = batch.clone()
            self.data_module.scaler = self.data_module.scaler.to(batch.device)
            fake_scaled=self.data_module.scale(fake_scaled,torch.zeros_like(fake_scaled[:,:,-1]))
            true=self.data_module.scale(true,torch.zeros_like(fake_scaled[:,:,-1]))
            return fake, fake_scaled, true
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

            if self.config["opt"] == "Adam":
                opt = torch.optim.Adam(self.flow.parameters(), lr=0.001)  
                
            else:
                opt = torch.optim.RMSprop(self.flow.parameters(), lr=self.config["lr_f"])
            if self.config["sched"] != None:
                lr_scheduler=self.scheduler(opt)
                return ([opt],[lr_scheduler])
            else:
                return opt
      

    def training_step(self, batch, batch_idx):
        opt=self.optimizers()
        mask=batch[:,:,-1]
        x= batch[:,:,:-1]
        if self.config["context_features"]>0:
            c=c[:,0].reshape(-1,self.config["context_features"])
        elif self.config["context_features"]==0:
            c=None
        opt.zero_grad()

        g_loss = -self.flow.to(self.device).log_prob(x.reshape(-1,(self.n_dim*self.n_part)),c if self.
        config["context_features"] else None).mean()/(self.n_dim*self.n_part)
        g_loss.backward()
        opt.step()
        self.log("logprob", g_loss, on_step=True, on_epoch=False, logger=True) 
       
       
            

    def validation_step(self, batch, batch_idx):
        """This calculates some important metrics on the hold out set (checking for overtraining)"""
        #print("start val")
        # assert batch.shape[1]==self.n_current
        mask = batch[:, :self.n_current,self.n_dim].bool().cpu()
        batch = batch[:, :self.n_current,:self.n_dim].cpu()
        mask_test=self.sample_n(mask).bool()
        batch = batch.to("cpu") 
        with torch.no_grad():
            gen, fake_scaled, true_scaled = self.sampleandscale(batch,mask, scale=True)
            batch[mask]=0
            # scores_real = self.dis_net(batch, mask=mask)
            # scores_fake = self.dis_net(gen, mask=mask)            
        true_scaled[mask]=0
        # Reverse Standard Scaling (this has nothing to do with flows, it is a standard preprocessing step)
        for i in range(self.n_current):
            fake_scaled[fake_scaled[:, i,2] < 0, i,2] = 0
           
        self.plot = plotting_point_cloud(model=self,gen=fake_scaled.reshape(-1,self.n_current,self.n_dim),true=true_scaled.reshape(-1,self.     n_current,self.n_dim),config=self.config,step=self.global_step,logger=self.logger, n=self.n_current,p=self.config["parton"])#,nf=z_scaled.reshape(-1,self.n_current,self.n_dim)

        try:
            self.plot.plot_mass(save=None, bins=50)
            # self.plot.plot_scores(scores_real.reshape(-1).detach().cpu().numpy(), scores_fake.reshape(-1).detach().cpu().numpy(), train=False, step=self.current_epoch)
        except Exception as e:
            traceback.print_exc()
        
        self.calc_log_metrics(fake_scaled,true_scaled)

        
       
