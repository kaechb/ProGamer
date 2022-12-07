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

from helpers import CosineWarmupScheduler, Scheduler

from pointflow import PF
from torch.optim.swa_utils import AveragedModel
sys.path.insert(1, "/home/kaechben/plots")
from functools import partial
from plotswb import *
from models import *
from metrics import *
# class

class ProGamer(pl.LightningModule):
    def create_resnet(self, in_features, out_features):
        """This is the network that outputs the parameters of the invertible transformation
        The only arguments can be the in dimension and the out dimenson, the structure
        of the network is defined over the config which is a class attribute
        Context Features: Amount of features used to condition the flow - in our case
        this is usually the mass
        num_blocks: How many Resnet blocks should be used, one res net block is are 1 input+ 2 layers
        and an additive skip connection from the first to the third"""
        c = self.config["context_features"]
        return nets.ResidualNet(
            in_features,
            out_features,
            hidden_features=self.config["network_nodes_nf"],
            context_features=c,
            num_blocks=self.config["network_layers_nf"],
            activation=self.config["activation"] if "activation" in self.config.keys() else FF.relu,
            dropout_probability=self.config["dropout"] if "dropout" in self.config.keys() else 0,
            use_batch_norm=self.config["batchnorm"] if "batchnorm" in self.config.keys() else 0,
        )

    def __init__(self, config, num_batches,path="/"):
        """This initializes the model and its hyperparameters"""
        super().__init__()
        self.hyperopt = True
        self.start = time.time()
        self.config = config
        self.automatic_optimization = False
        self.freq_d = config["freq"]

        # Loss function of the Normalizing flows
        self.logprobs = []

        self.save_hyperparameters()
        self.flows = []
        self.fpnds = []
        self.w1ms = []
        self.stop_train=False
        self.n_dim = self.config["n_dim"]
        self.n_part = config["n_part"]
        self.n_current = 2
        self.alpha = 1
        self.num_batches = int(num_batches)
        #self.build_flow()
        
        nf=PF.load_from_checkpoint(config["load_ckpt"])
        #self.config["context_features"]=nf.config["context_features"]
        self.flow=nf.flow
        self.flow.eval()
        self.train_gen=False
        self.d_losses=torch.ones(5)
        self.mse=nn.MSELoss()
        self.gen_net = Gen(n_dim=self.n_dim,hidden=config["hidden"],num_layers=config["num_layers"],dropout=config["dropout"],no_hidden=config["no_hidden_gen"],l_dim=config["l_dim"],num_heads=config["heads"],  norm_first=config["normfirst"]).cuda()
        self.dis_net = Disc(n_dim=self.n_dim,hidden=config["hidden"],l_dim=config["l_dim"],num_layers=config["num_layers"], norm_first=config["normfirst"],num_heads=config["heads"],dropout=config["dropout"],cls=config["cls"]).cuda()
        self.sig = nn.Sigmoid()
        self.df = pd.DataFrame()
        for p in self.dis_net.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal(p)

        for p in self.gen_net.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal(p)
        self.train_nf = int(config["max_epochs"] * config["frac_pretrain"])
        self.refinement=False
        self.counter=0
        if self.config["swa"]:
            self.dis_net= AveragedModel(self.dis_net)
        if self.config["swagen"]:
            self.gen_net= AveragedModel(self.gen_net)

    def load_datamodule(self, data_module):
        """needed for lightning training to work, it just sets the dataloader for training and validation"""
        self.data_module = data_module

    def _summary(self, temp):
        self.summary_path = "/beegfs/desy/user/{}/{}/summary.csv".format(os.environ["USER"], self.config["name"])
        if os.path.isfile(self.summary_path):
            summary = pd.read_csv(self.summary_path).set_index(["path_index"])
        else:
            print("summary not found")
            summary = pd.DataFrame()
        summary.loc[self.logger.log_dir, self.config.keys()] = self.config.values()
        summary.loc[self.logger.log_dir, temp.keys()] = temp.values()
        summary.loc[self.logger.log_dir, "time"] = time.time() - self.start
        summary.to_csv(self.summary_path, index_label=["path_index"])
        return summary

    def _results(self, temp):
        self.df = pd.concat([self.df,pd.DataFrame([temp],index=[self.current_epoch])])
        self.df.to_csv(self.logger.log_dir + "result.csv", index_label=["index"])

    # def on_after_backward(self) -> None:
    #     """This is a genious little hook, sometimes my model dies, i have no clue why. This saves the training from crashing and continues"""
    #     valid_gradients = False
    #     for name, param in self.named_parameters():
    #         if param.grad is not None:
    #             valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
    #             if not valid_gradients:
    #                 break
    #     if not valid_gradients:
    #         print("not valid grads", self.counter)
    #         self.zero_grad()
    #         self.counter += 1
    #         if self.counter > 5:
    #             raise ValueError("5 nangrads in a row")
    #         self.stop_train=True
    #     else:
    #         self.counter = 0
    
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
        self.flow.eval()
        with torch.no_grad():
            if self.config["flow_prior"]:
                
                #print("start flow sample")
                z = self.flow.sample(len(batch)*batch.shape[1]).reshape(len(batch), 
                batch.shape[1], self.n_dim)
            else:
                z = torch.normal(torch.zeros(batch.shape,device=batch.device),torch.ones(batch.shape,device=batch.device))

            #print("end")
        #print("start trans sample")
        if self.config["add_corr"]:
            fake = z + self.gen_net(z, mask=mask)
        else:
            fake=self.gen_net(z,mask)
        #print("end")

        # fake = fake*((~mask).reshape(len(batch),30,1).float()) #somehow this gives nangrad
        fake[mask]=0
        if scale:
            fake_scaled = fake.clone()
            true = batch.clone()
            z_scaled = z.clone()
            self.data_module.scaler = self.data_module.scaler.to(batch.device)
            fake_scaled=self.data_module.scaler.inverse_transform(fake_scaled)
            z_scaled=self.data_module.scaler.inverse_transform(z_scaled)
            true=self.data_module.scaler.inverse_transform(true)
            
            fake_scaled[mask]=0
            z_scaled[mask]=0
            return fake, fake_scaled, true, z_scaled
        else:
            return fake

    def scheduler(self,opt_g,opt_d):
        if self.config["sched"] == "cosine":

           
            max_iter = (self.config["max_epochs"] - self.train_nf) * self.num_batches -self.global_step
            lr_scheduler_d = CosineWarmupScheduler(opt_d, warmup=self.config["warmup"] * self.num_batches, max_iters=max_iter)
            lr_scheduler_g = CosineWarmupScheduler(opt_g, warmup=self.config["warmup"] * self.num_batches, max_iters=max_iter)
            # if self.config["bullshitbingo2"]:
            #     self.freq_d-=1
        elif self.config["sched"] == "cosine2":
            
            max_iter = (self.config["max_epochs"] - self.train_nf) * self.num_batches//3-self.global_step
            lr_scheduler_d = CosineWarmupScheduler(opt_d, warmup=self.config["warmup"] * self.num_batches, max_iters=max_iter )#15,150 // 3
            lr_scheduler_g = CosineWarmupScheduler(opt_g, warmup=self.config["warmup"] * self.num_batches , max_iters=max_iter)#  // 3
            # if self.config["bullshitbingo2"]:
            #     self.freq_d-=1
        elif self.config["sched"] == "linear":
            max_iter = (self.config["max_epochs"] - self.train_nf // 2) * self.num_batches-self.global_step
          
            lr_scheduler_d = Scheduler(opt_d,dim_embed=self.config["l_dim"], warmup_steps=self.config["warmup"] * self.num_batches )#15,150 // 3
            lr_scheduler_g = Scheduler(opt_g, dim_embed=self.config["l_dim"], warmup_steps=self.config["warmup"] * self.num_batches )#  // 3

        else:
            lr_scheduler_d = None
            lr_scheduler_g = None
        return lr_scheduler_d,lr_scheduler_g

    def configure_optimizers(self):
        self.losses = []
        # mlosses are initialized with None during the time it is not turned on, makes it easier to plot
        if self.config["opt"] == "Adam":
            opt_g = torch.optim.Adam(self.gen_net.parameters(), lr=self.config["lr_g"], betas=(0, 0.9))
            opt_d = torch.optim.Adam(self.dis_net.parameters(), lr=self.config["lr_d"], betas=(0, 0.9))
        elif self.config["opt"] == "mixed" and False:
            opt_g = torch.optim.Adam(
                self.gen_net.parameters(), lr=self.config["lr_g"], betas=(0, 0.9)
            )
            opt_d = torch.optim.SGD(self.dis_net.parameters(), lr=self.config["lr_d"])#TRAINING WITH SGD DOES NOT WORK AT ALL
        else:
            opt_g = torch.optim.RMSprop(self.gen_net.parameters(), lr=self.config["lr_g"])
            opt_d = torch.optim.RMSprop(self.dis_net.parameters(), lr=self.config["lr_d"])
        lr_scheduler_d,lr_scheduler_g=self.scheduler(opt_d,opt_g)
        if self.config["sched"] != None:
            return [opt_d, opt_g], [lr_scheduler_d, lr_scheduler_g]
        else:
            return [ opt_d, opt_g]

   
    def training_step(self, batch, batch_idx):
        
        """training loop of the model, here all the data is passed forward to a gaussian
        This is the important part what is happening here. This is all the training we do"""
        
        mask = batch[:, self.n_current*self.n_dim:].bool()
        batch = batch[:, :self.n_current*self.n_dim]
        batch = batch.reshape(len(batch), self.n_current, self.n_dim)
        batch[mask]=0
        opt_d, opt_g = self.optimizers()
        if self.config["sched"]:
           sched_d, sched_g = self.lr_schedulers()
           sched_d.step() 
           sched_g.step() 
        # ### NF PART
        if self.config["sched"] != None:
            self.log("Training/lr_g", sched_g.get_last_lr()[-1])
            self.log("Training/lr_d", sched_d.get_last_lr()[-1])

    
        ### GAN PART

        fake = self.sampleandscale(batch, mask, scale=False)
    
        pred_real = self.dis_net(batch, mask=mask)
        pred_fake = self.dis_net(fake.detach(), mask=mask)
        target_real = torch.ones_like(pred_real)
        target_fake = torch.zeros_like(pred_fake)
        pred = torch.vstack((pred_real, pred_fake))
        target = torch.vstack((target_real, target_fake))
        d_loss = self.mse(pred, target).mean()
        opt_d.zero_grad()
        self.manual_backward(d_loss)

        opt_d.step()
        self.d_losses[self.global_step%5]=d_loss.detach()
        if self.current_epoch % 5 == 0 and self.global_step%self.num_batches<3 and self.current_epoch > self.train_nf / 2:
                self.plot.plot_scores(pred_real.detach().cpu().numpy(),pred_fake.detach().cpu().numpy(),train=True,step=self.current_epoch)
                
        if not self.train_gen and self.d_losses.mean()<0.1:
            self.train_gen=True
            self.log("n_current",self.n_current)
            print("start training gen with {} particles".format(self.n_current))
        
        self.log("Training/d_loss", d_loss, logger=True, prog_bar=False,on_step=True)
        if self.global_step < 2:
            print("passed test disc")

        if self.train_gen or self.global_step <= 3:
            opt_g.zero_grad()
            fake = self.sampleandscale(batch, mask, scale=False)#~(mask.bool())
            
        
            pred_fake = self.dis_net(fake, mask=mask)
            target_real = torch.ones_like(pred_fake)
            # if self.wgan:
            #     g_loss = -torch.mean(pred_fake.view(-1))
            # else:
            g_loss = self.mse((pred_fake.view(-1)), target_real.view(-1))
            self.manual_backward(g_loss)
            if self.global_step > 10:
                opt_g.step()
            else:
                opt_g.zero_grad()
            self.log("Training/g_loss", g_loss, logger=True, prog_bar=False,on_step=True)
            
            if self.global_step < 3:
                print("passed test gen")

            # Control plot train
            

    def validation_step(self, batch, batch_idx):
        """This calculates some important metrics on the hold out set (checking for overtraining)"""
        #print("start val")
        mask = batch[:, self.n_current*self.n_dim:].bool().cpu()
        batch = batch[:, :self.n_current*self.n_dim].cpu()
                
        self.dis_net.train()
        self.gen_net.train()
        self.flow.eval()
        mask_test=self.sample_n(mask).bool()
        batch = batch.to("cpu")
        self.flow = self.flow.to("cpu")
        self.dis_net = self.dis_net.cpu()
        self.gen_net = self.gen_net.cpu()
        
        #print("start sampling")
        with torch.no_grad():
            # logprob = -self.flow.log_prob(batch).mean() / 90
            batch = batch.reshape(len(batch),self.n_current,self.n_dim)
            #TODO CHange this back to its right mask_test
            gen, fake_scaled, true_scaled, z_scaled = self.sampleandscale(batch,mask, 
            scale=True)
            #print("end sampling")
            #mask_test
            batch[mask]=0

            #print("start scores")
            scores_real = self.dis_net(batch, mask=mask)
            scores_fake = self.dis_net(gen, mask=mask)
            #print("end scores")
        #TODO CHange this back to its right mask_test
        
        true_scaled[mask]=0
        fake_scaled[mask] = 0
        z_scaled[mask]  = 0
        # Reverse Standard Scaling (this has nothing to do with flows, it is a standard preprocessing step)
        m_t = mass(true_scaled)
        m_gen = mass(z_scaled)
        m_c = mass(fake_scaled)
        
        for i in range(self.n_current):
            fake_scaled[fake_scaled[:, i,2] < 0, i,2] = 0
            z_scaled[z_scaled[:, i,2] < 0, i,2] = 0
        # Some metrics we track
        #print("start metrics")
        cov, mmd = cov_mmd( true_scaled,fake_scaled, use_tqdm=False)
        
        cov_nf, mmd_nf = cov_mmd(true_scaled,z_scaled,  use_tqdm=False)
        try:
            
            fpndv = fpnd(fake_scaled[:50000,:].numpy(), use_tqdm=False, jet_type=self.config["parton"])
        except:
            fpndv = 1000

        w1m_ = w1m(fake_scaled, true_scaled)[0]
        w1p_ = w1p(fake_scaled, true_scaled)[0]
        w1efp_ = w1efp(fake_scaled, true_scaled)[0]
        try:
            fgd_=fgd(true_scaled,fake_scaled,p=self.config["parton"],n=self.n_current)[0]
        except:
            print("error calc fgd")
            print((true_scaled==true_scaled).all())
            print((fake_scaled==fake_scaled).all())
            fgd_=1000
        #print("end metrics")
        # if fpndv<0.08 and w1m_<0.001 and not self.refinement:
        #     opt_nf,opt_d,opt_g=self.optimizers()
        #     for g in opt_nf.param_groups:
        #         g['lr'] *= 0.001
        #     for g in opt_d.param_groups:
        #         g['lr'] *= 0.1
        #     for g in opt_g.param_groups:
        #         g['lr'] *= 0.001
        #     self.refinement=True
        self.w1ms.append(w1m_)
        self.fpnds.append(fpndv)
        temp = {"val_fpnd": fpndv,"val_fgd":fgd_,"val_mmd": mmd,"val_cov": cov,"val_w1m": w1m_,
                "val_w1efp": w1efp_,"val_w1p": w1p_,"step": self.global_step,}
        print("epoch {}: ".format(self.current_epoch), temp)
        
        
        
        #print("start logging")
        self.log("hp_metric", w1m_,)
        self.log("fgd",fgd_)
        self.log("w1m", w1m_,)
        self.log("w1p", w1p_,)
        self.log("w1efp", w1efp_,)
        self.log("cov", cov,  )
        self.log("cov_nf", cov_nf,  )
        self.log("fpnd", fpndv,  )
        self.log("mmd", mmd,  )
        self.log("n_current",self.n_current)
        #print("end logging")
        self.plot = plotting_point_cloud(model=self,gen=fake_scaled.reshape(-1,self.n_current,self.n_dim),true=true_scaled.reshape(-1,self.n_current,self.n_dim),config=self.config,step=self.global_step,
        logger=self.logger, n=self.n_current,p=self.config["parton"])#,nf=z_scaled.reshape(-1,self.n_current,self.n_dim)
        # self.plot.plot_scores(scores_real,scores_fake,train=True,step=self.global_step)
        self.plot.plot_mom(self.global_step)
        try:
            self.plot.plot_mass(save=None, bins=50)
            # self.plot.plot_2d(save=True)
        #     self.plot.var_part(true=true[:,:self.n_dim],gen=gen_corr[:,:self.n_dim],true_n=n_true,gen_n=n_gen_corr,
        #                          m_true=m_t,m_gen=m_test ,save=True)
            self.plot.plot_scores(scores_real.reshape(-1).detach().cpu().numpy(), scores_fake.reshape(-1).detach().cpu().numpy(), train=False, step=self.current_epoch)
        except Exception as e:
            traceback.print_exc()
        # if (np.array([self.fpnds]) > 30).all() and self.global_step > 300000:
        #     print("no convergence, stop training")
             
        #     summary = self._summary(temp)
        #     raise

        self.flow = self.flow.to("cuda")
        self.gen_net = self.gen_net.to("cuda")
        self.dis_net = self.dis_net.to("cuda")
        
        if (cov>0.45 and w1m_<0.01 and self.n_current<30) or (fpndv<0.5 and self.n_current<self.n_part) or (self.n_current>30 and cov>0.5 and w1m_<0.002 and self.n_current<self.n_part):
            self.n_current+=1
            self.train_gen = False
            self.counter=0
            self.trainer.accelerator.setup(self)
            self.data_module.setup("train",self.n_current)
            print("number particles set to to ",self.n_current)
        if self.current_epoch>50 and self.n_current==2:
            print("not converging, n:",self.n_current)
            raise
        if self.current_epoch>100 and self.n_current<5:
            print("not converging, n:",self.n_current)
            raise
        if self.current_epoch>300 and self.n_current<15:
            print("not converging, n:",self.n_current)
            raise
        if self.current_epoch>500 and self.n_current<30:
            print("not converging, n:",self.n_current)
            raise
        # if self.hyperopt and self.global_step > 3:
            # try:
            #     self._results(temp)
            # except:
            #     print("error in results")
            # if (fpndv<self.fpnds).all():             
            #     summary = self._summary(temp)
