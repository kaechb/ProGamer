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
from helpers import CosineWarmupScheduler, Scheduler
from preprocess import center_jets_tensor

sys.path.insert(1, "/home/kaechben/plots")
from functools import partial

from plotswb import *

from metrics import *
from models import *

# class
MASS=False

class ProGamer(pl.LightningModule):


    def __init__(self, config, path="/",**kwargs):
        """This initializes the model and its hyperparameters"""
        super().__init__()
        self.opt = config["opt"]
        self.parton = config["parton"]
        self.automatic_optimization = False
        config["mass"]=MASS
        self.n_dim = config["n_dim"]
        self.n_part = config["n_part"]
        self.n_current = config["n_start"]
        # self.momentum=config["momentum_gen"]
        self.lr_g=config["lr_g"]
        self.lr_d=config["lr_d"]
        #self.beta=config["beta1"]
        self.n_start=config["n_start"]
        self.cond_dim=config["cond_dim"]
        self.gen_net = Gen(**config).cuda()
        self.dis_net = Disc(**config).cuda()
        self.gan = kwargs["gan"]
        self.w1m_best= 0.2
        if self.gan=="ls":
            self.loss=nn.MSELoss()
        self.counter=0
        self.save_hyperparameters()
        self.d_loss_mean=0
        self.g_loss_mean=0
        self.timesteps=0
        self.sum_parameter_gen=[]
        self.sum_parameter_dis=[]

        self.fadein=np.inf
        self.relu=torch.nn.ReLU()
        self.dis_net_dict={"average":False,"model":self.dis_net,"step":0}
        self.gen_net_dict={"average":False,"model":self.gen_net,"step":0}
        self.mean_field_loss=True
        self.stop_mean=config["stop_mean"]
        # if config["swa"]:
        # self.dis_net= AveragedModel(self.dis_net)
        # # if config["swagen"]:
        # self.gen_net= AveragedModel(self.gen_net)


    def on_validation_epoch_start(self, *args, **kwargs):
        self.dis_net = self.dis_net.cpu()
        self.gen_net = self.gen_net.cpu()
        self.gen_net.eval()
        self.dis_net.train()

    def on_validation_epoch_end(self, *args, **kwargs):
        self.gen_net = self.gen_net.to("cuda")
        self.dis_net = self.dis_net.to("cuda")
        self.gen_net.train()
        self.dis_net.train()

    def load_datamodule(self, data_module):
        """needed for lightning training to work, it just sets the dataloader for training and validation"""
        self.data_module = data_module

    def sampleandscale(self, batch, mask=None, scale=False):

       # with torch.no_grad():
        z=torch.normal(torch.zeros_like(batch),torch.ones_like(batch))
        #cond=torch.normal(torch.zeros(len(batch),self.cond_dim,device=batch.device),torch.ones(len(batch),self.cond_dim, device=batch.device)).unsqueeze(1)
        cond=None
        fake=self.gen_net(z,cond,mask=mask)
        if scale:
            fake_scaled = fake.clone()
            true = batch.clone()
            self.data_module.scaler = self.data_module.scaler.to(batch.device)
            fake_scaled=self.data_module.scaler.inverse_transform(fake_scaled)
            true=self.data_module.scaler.inverse_transform(true)
            return fake, fake_scaled, true
        else:
            return fake

    def get_model_weights(self,model):
        average = {}
        params = dict(model.named_parameters())
        for p in params:
            average[p] = params[p].detach()
        return average

    def historical_loss(self,model_dict):

        if not model_dict["average"]:
            print("Starting historical weight averaging for discriminator")
            model_dict["average"] = self.get_model_weights(model_dict["model"])
            model_dict["step"] += 1
            return torch.tensor([0]).float().cuda()
        else:
            params = dict(model_dict["model"].named_parameters())
            err=torch.tensor([0]).float().cuda()
            for p in params:
                err += self.loss(params[p], model_dict["average"][p])
                model_dict["average"][p] = (model_dict["average"][p] * (model_dict["step"]-1) + params[p].detach())/model_dict["step"]
            model_dict["step"] += 1
            return err.mean()

    def configure_optimizers(self):
        if self.opt == "Adam":
            opt_g = torch.optim.Adam(self.gen_net.parameters(), lr=self.lr_g, betas=(0, 0.999), eps=1e-14)
            opt_d = torch.optim.Adam(self.dis_net.parameters(), lr=self.lr_d,betas=(0, 0.999), eps=1e-14)#
        else:
            raise
        return [opt_d, opt_g]

    def train_disc(self,batch,mask,opt_d):

        with torch.no_grad():
            fake = self.sampleandscale(batch, mask, scale=False)
        if self.fadein<10000:
            mask[mask!=0]= -torch.tensor(float('inf'))
            mask[:,self.n_current-10:]-=torch.exp(torch.tensor(-self.fadein).cuda())*10000
            self.fadein+=1
        opt_d.zero_grad()
        self.dis_net.zero_grad()
        if self.mean_field_loss:
            pred_real,mean_field = self.dis_net(batch, mask=mask,mean_field=True)#mass=m_true
        else:
            mean_field=None
            pred_real = self.dis_net(batch, mask=mask,)#mass=m_true
        pred_fake = self.dis_net(fake.detach(), mask=mask,)
        d_loss = self.relu(1.0 - pred_real).mean() +self.relu(1.0 + pred_fake).mean()
        if False:
            historical_dis = self.historical_loss(self.dis_net_dict)
            if self.dis_net_dict["step"]>1:
                self.manual_backward(historical_dis)
                self.log("Training/dis_historical", historical_dis, logger=True,prog_bar=False,on_step=True)
        self.manual_backward(d_loss)
        self.log("Training/d_loss", d_loss, logger=True,prog_bar=False,on_step=True)
        opt_d.step()
        # self.log("Training/d_loss", d_loss, logger=True,prog_bar=False,on_step=True)

        return mean_field

    def train_gen(self,batch,mask,opt_g,mean_field=None):
        # opt_g.zero_grad()
        opt_g.zero_grad()
        self.gen_net.zero_grad()
        fake = self.sampleandscale(batch, mask, scale=False)
        if mean_field is not None:
            pred,mean_field_gen = self.dis_net(fake, mask=mask,mean_field=True )
            mean_field = torch.clamp(self.loss(mean_field_gen,mean_field.detach()).mean(),min=0,max=40)
            mean_field.backward(retain_graph=True)
            g_loss=-pred.mean()
            self.manual_backward(mean_field,retain_graph=True)
            if False:
                historical_gen = self.historical_loss(self.gen_net_dict)
                if self.gen_net_dict["step"]>1:
                    self.manual_backward(historical_gen)
                self.log("Training/gen_historical", historical_gen, logger=True,prog_bar=False,on_step=True)
            self.log("Training/mean_field", mean_field, logger=True,prog_bar=False,on_step=True)

        else:
            pred = self.dis_net(fake, mask=mask, )
            g_loss=-pred.mean()
        self.manual_backward(g_loss)
        opt_g.step()
        self.log("Training/g_loss", g_loss, logger=True, prog_bar=False,on_step=True)

    def training_step(self, batch):
        if self.global_step>300000 and self.stop_mean:
            self.mean_field_loss=False
        if len(batch)==1:
            return None
        mask= batch[:, :self.n_current,self.n_dim].float()
        batch = batch[:, :self.n_current,:self.n_dim]
        opt_d, opt_g = self.optimizers()
        ### GAN PART
        mean_field=self.train_disc(batch,mask,opt_d)
        self.train_gen(batch,mask,opt_g,mean_field)


    def validation_step(self, batch, batch_idx):
        """This calculates some important metrics on the hold out set (checking for overtraining)"""
        #print("start val")
        # if self.config["smart_batching"]:
        # self.n_current=batch.shape[1]
        if self.global_step==0:
            self.n_current=self.n_start
            self.data_module.n_part=self.n_current
            self.log("n_current",self.n_current)
            # self.data_module.setup("validate")
            # batch=self.data_module.val_dataloader().dataset
        batch = batch.to("cpu")
        mask = batch[:,:self.n_current,-1].bool()
        assert not mask[:,0].all()
        batch = batch[:, :self.n_current,:self.n_dim].cpu()
        with torch.no_grad():
            gen, fake_scaled, true_scaled = self.sampleandscale(batch,mask, scale=True)
            gen[mask]=batch[mask]
            scores_real = self.dis_net(batch, mask=mask, )
            scores_fake = self.dis_net(gen, mask=mask, )
            if self.gan=="ns":
                scores_real=self.sig(scores_real)
                scores_fake=self.sig(scores_fake)
        fake_scaled[mask]=0
        true_scaled[mask]=0
        for i in range(self.n_current):
            fake_scaled[fake_scaled[:, i,2] < 0, i,2] = true_scaled[:,:,2].min()
            #z_scaled[z_scaled[:, i,2] < 0, i,2] = 0
        fake_scaled[:,:,:3] = center_jets_tensor(fake_scaled[:,:,:3][...,[2,0,1]])[...,[1,2,0]]
        fake_scaled[mask]=0
        true_scaled[mask]=0
        self.calc_log_metrics(fake_scaled,true_scaled,scores_real,scores_fake)

    def calc_log_metrics(self, fake_scaled,true_scaled,scores_real,scores_fake):
        w1m_ = w1m(fake_scaled, true_scaled,num_eval_samples=10000)[0]
        # if w1m_<0.001:
        #     self.mean_field_loss=False
        if  w1m_<0.001 and self.n_current<self.n_part :#
            self.n_current+=10
            self.fadein=0
            self.data_module.n_part=self.n_current
            self.w1m_best=100
            self.mean_field_loss=True
            print("number particles increased to {}".format(self.n_current))
            self.log("n_current",self.n_current)
        temp = {"w1m": w1m_,}
        try:
            cov, mmd = cov_mmd( true_scaled,fake_scaled, use_tqdm=False)
            temp = {"mmd": mmd,"cov": cov,"w1m": w1m_,}
        except:
            print("cov mmd failed")
        print("epoch {}: ".format(self.current_epoch), temp)
        self.logger.log_metrics(temp,step=self.global_step)
        self.log("w1m",w1m_)
        if w1m_<self.w1m_best or self.global_step==0:
            self.w1m_best=w1m_

            self.log("best_w1m",w1m_)
            self.plot = plotting_point_cloud(model=self,gen=fake_scaled,true=true_scaled,n_dim=self.n_dim, n_part=self.n_current, step=self.global_step,logger=self.logger, n=self.n_current,p=self.parton)
            try:
                self.plot.plot_scores(scores_real.reshape(-1).numpy(),scores_fake.reshape(-1).numpy(),False,self.global_step)

                self.plot.plot_mass(save=None, bins=50)
                # self.plot.plot_scores(scores_real.reshape(-1).detach().cpu().numpy(), scores_fake.reshape(-1).detach().cpu().numpy(), train=False, step=self.current_epoch)
            except Exception as e:
                traceback.print_exc()
        if self.n_part==30 and w1m_<0.001 or self.w1m_best==w1m_ and self.n_part==30 :
                fpnd_best=fpnd(fake_scaled,jet_type="t",use_tqdm=False)
                self.log("fpnd",fpnd_best)
        self.log("n_current",self.n_current)