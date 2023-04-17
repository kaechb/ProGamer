import os
import sys
import time
import traceback
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.autograd as autograd
from jetnet.evaluation import cov_mmd, fpnd, w1efp, w1m, w1p

from torch import Tensor, nn
from torch.autograd import Variable
from torch.nn import TransformerEncoderLayer
from torch.nn import functional as FF
from torch.nn.functional import leaky_relu, sigmoid
from torch.optim.swa_utils import AveragedModel
from helpers import CosineWarmupScheduler, Scheduler
from preprocess import center_jets_tensor
from scipy.stats import wasserstein_distance
sys.path.insert(1, "/home/kaechben/plots")
from functools import partial

from plotswb import *

# from metrics import *
from models import *
import models2 as best
FREQ=1
class ProGamer(pl.LightningModule):


    def __init__(self, config, path="/",**kwargs):
        """This initializes the model and its hyperparameters"""
        super().__init__()
        self.opt = config["opt"]
        self.parton = config["parton"]
        self.n_dim = config["n_dim"]
        self.n_part = config["n_part"]
        self.n_current = config["n_start"]
        # self.momentum=config["momentum_gen"]
        self.lr_g=config["lr_g"]
        self.lr_d=config["lr_d"]
        #self.beta=config["beta1"]
        self.n_start=config["n_start"]
        self.part_increase=config["part_increase"]
        # self.cond_dim=config["cond_dim"]
        if config["best"]:
            self.gen_net = Gen(**config).cuda()
            self.dis_net =  Disc(**config).cuda()
        else:
            self.gen_net = best.Gen(**config).cuda()
            self.dis_net =  best.Disc(**config).cuda()

        self.gan = kwargs["gan"]
        self.w1m_best= 0.2
        if self.gan=="ls":
            self.loss=nn.MSELoss()
        else:
            self.loss=lambda x,y:self.relu(1.0 - x).mean() +self.relu(1.0 + y).mean()
        self.gp_weight=10
        self.save_hyperparameters()
        self.fadein=np.inf
        self.relu=torch.nn.ReLU()
        self.dis_net_dict={"average":False,"model":self.dis_net,"step":0}
        self.gen_net_dict={"average":False,"model":self.gen_net,"step":0}
        if "mean_field_loss" in config.keys():
            self.mean_field_loss=config["mean_field_loss"]
        else:
            self.mean_field_loss=False
        if "fine_tune" in config.keys():
            self.fine_tune=config["fine_tune"]
        else:
            self.fine_tune=True
        self.mass=config["mass"]
        self.mse=nn.MSELoss()
        self.stop_mean=config["stop_mean"]
        self.automatic_optimization = False
        self.target_real = torch.ones(config["batch_size"],1).cuda()
        self.target_fake = torch.zeros(config["batch_size"],1).cuda()
        self.i=0
        self.g_loss_mean=None


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

        if self.global_step==0:
            self.n_current=self.n_start
            self.data_module.n_part=self.n_current

    def sampleandscale(self, batch, mask=None, scale=False):

        if not scale:
            with torch.no_grad():
                z=torch.normal(torch.zeros_like(batch),torch.ones_like(batch))
        else:
            with torch.no_grad():
                z=torch.normal(torch.zeros(batch.shape[0],batch.shape[1],batch.shape[2]),torch.ones(batch.shape[0],batch.shape[1],batch.shape[2]))
        z[mask]*=0
        fake=self.gen_net(z,mask=mask, weight=False)

        fake[:,:,2]=F.relu(fake[:,:,2]-self.min_pt)+self.min_pt
        fake=fake*(~mask.bool()).unsqueeze(-1).float()
        if scale:
            fake_scaled = fake.clone()
            true = batch.clone()
            self.data_module.scaler = self.data_module.scaler.to(batch.device)
            fake_scaled=self.data_module.scaler.inverse_transform(fake_scaled)
            true=self.data_module.scaler.inverse_transform(true)
            return fake, fake_scaled, true
        else:
            return fake

    def _gradient_penalty(self, real_data, generated_data,mask):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1)
        alpha = alpha.expand_as(real_data)

        alpha = alpha.cuda()
        interpolated = alpha * real_data + (1 - alpha) * generated_data
        interpolated = Variable(interpolated, requires_grad=True).cuda()

        # Calculate probability of interpolated examples

        if self.mean_field_loss:
            prob_interpolated,_,_ = self.dis_net(interpolated,mask=torch.zeros_like(mask).bool(), weight=False)
        else:
            prob_interpolated,_,_ = self.dis_net(interpolated,mask=torch.zeros_like(mask).bool(), weight=False)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones_like(prob_interpolated),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)
        #self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data[0])

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def configure_optimizers(self):

        self.min_pt=self.data_module.min_pt
        if self.opt == "Adam":
            opt_g = torch.optim.Adam(self.gen_net.parameters(), lr=self.lr_g, betas=(0., 0.999), eps=1e-14)
            opt_d = torch.optim.Adam(self.dis_net.parameters(), lr=self.lr_d,betas=(0., 0.999), eps=1e-14)#
        elif self.opt == "AdamW":
            opt_g = torch.optim.Adam(self.gen_net.parameters(), lr=self.lr_g, betas=(0., 0.999), eps=1e-14)
            opt_d = torch.optim.AdamW(self.dis_net.parameters(), lr=self.lr_d,betas=(0., 0.999), eps=1e-14, weight_decay=0.01)#
        else:
            raise
        sched_d,sched_g=self.schedulers(opt_d,opt_g)
        return [opt_d, opt_g],[sched_d,sched_g]

    def schedulers(self,opt_d,opt_g):
        sched_d=CosineWarmupScheduler(opt_d, 20, 2000*1000)
        sched_g=CosineWarmupScheduler(opt_g, 20, 2000*1000)
        return sched_d,sched_g

    def train_disc(self,batch,mask,opt_d):

        with torch.no_grad():
            fake = self.sampleandscale(batch, mask, scale=False)
        opt_d.zero_grad()
        self.dis_net.zero_grad()
        batch=batch*(~mask.bool()).unsqueeze(-1).float()
        fake=fake*(~mask.bool()).unsqueeze(-1).float()
        if self.mean_field_loss:
            pred_real,mean_field,m_t = self.dis_net(batch, mask=mask, weight=False)
            pred_fake,_,m_f = self.dis_net(fake.detach(), mask=mask, weight=False)
        else:
            mean_field=None
            pred_real,_,_ = self.dis_net(batch, mask=mask, weight=False)#mass=m_true
            pred_fake,_,_ = self.dis_net(fake.detach(), mask=mask, weight=False)
        pred_fake=pred_fake.reshape(-1)
        pred_real=pred_real.reshape(-1)
        if self.gan=="ls":
            target_fake=torch.zeros_like(pred_fake)
            target_real=torch.ones_like(pred_real)
            d_loss = self.mse(pred_fake, target_fake ).mean()+self.mse(pred_real,target_real).mean()
            if self.global_step<2:
                    self.d_loss_mean=d_loss
            self.d_loss_mean=d_loss.detach()*0.01+0.99*self.d_loss_mean
        elif self.gan=="hinge":
            d_loss=F.relu(1-pred_real).mean()+F.relu(1+pred_fake).mean()
            if self.global_step<2:
                    self.d_loss_mean=d_loss
            self.d_loss_mean=d_loss.detach()*0.01+0.99*self.d_loss_mean
        else:
            gp=self._gradient_penalty(batch, fake,mask=mask)
            d_loss=-pred_real.mean()+pred_fake.mean()
            if self.global_step<2:
                    self.d_loss_mean=d_loss
            self.d_loss_mean=d_loss.detach()*0.01+0.99*self.d_loss_mean
            d_loss+=gp
            self._log_dict["Training/gp"]=gp

        if self.mass and self.mean_field_loss:
            m_t=m_t.reshape(-1)
            m_f=m_f.reshape(-1)
            b_t=mass(batch).reshape(-1)
            b_f=mass(fake.detach()).reshape(-1)
            m_loss=1e-5*self.mse(m_f,b_f)+1e-5*self.mse(b_t,m_t)
            d_loss+=m_loss
            self._log_dict["Training/m_loss"]= m_loss.detach()

        self.manual_backward(d_loss)
        opt_d.step()
        self._log_dict["Training/lr_d"]=opt_d.param_groups[0]["lr"]

        self._log_dict["Training/d_loss"]=self.d_loss_mean
        return mean_field

    def train_gen(self,batch,mask,opt_g,mean_field=None):

        opt_g.zero_grad()
        self.gen_net.zero_grad()
        fake= self.sampleandscale(batch, mask=mask, scale=False)
        if mask is not None:
            fake=fake*(~mask).unsqueeze(-1)
        if mean_field is not None:
            pred,mean_field_gen,_ = self.dis_net(fake, mask=mask, weight=False)
            assert mean_field.shape==mean_field_gen.shape

            mean_field = self.mse(mean_field_gen,mean_field.detach()).mean()
            self._log_dict["Training/mean_field"]= mean_field

        else:
            pred,_,_ = self.dis_net(fake, mask=mask, weight=False )
        pred=pred.reshape(-1)
        if self.gan=="ls":
            target=torch.ones_like(pred)
            g_loss=0.5*self.mse(pred,target).mean()
        else:
            g_loss=-0.5*pred.mean()
        if self.g_loss_mean is None:
            self.g_loss_mean=g_loss
        self.g_loss_mean=g_loss.detach()*0.01+0.99*self.g_loss_mean
        if self.mean_field_loss:
            g_loss+=mean_field
        self.manual_backward(g_loss)
        opt_g.step()
        self._log_dict["Training/lr_g"]=opt_g.param_groups[0]["lr"]
        self._log_dict["Training/g_loss"]= self.g_loss_mean

    def training_step(self, batch):

        self._log_dict={}
        if self.global_step>30000 and self.stop_mean:
            self.mean_field_loss=False
        if len(batch)==1:
            return None
        mask = batch[:, :self.n_part,self.n_dim].bool()
        batch = batch[:, :self.n_part,:self.n_dim]

        opt_d, opt_g= self.optimizers()

        sched_d, sched_g = self.lr_schedulers()
        if self.continue_training and self.global_step==0 and self.ckpt is not None:
            opt_d.optimizer.load_state_dict(torch.load(self.ckpt)["optimizer_states"][0])
            opt_g.optimizer.load_state_dict(torch.load(self.ckpt)["optimizer_states"][1])
        ### GAN PART
        mean_field=self.train_disc(batch,mask,opt_d)
        if self.global_step%(FREQ)==0:
            self.train_gen(batch,mask,opt_g,mean_field)
            self.i+=1
            if self.i%(100//FREQ)==0:
                self.logger.log_metrics(self._log_dict, step=self.global_step)
        sched_d.step()
        sched_g.step()

    def validation_step(self, batch, batch_idx):
        """This calculates some important metrics on the hold out set (checking for overtraining)"""
        self._log_dict={}
        batch = batch.to("cpu")
        mask = batch[:,:self.n_current,-1].bool()
        #mask = torch.cat([mask for i in range(10)],dim=0)
        assert not mask[:,0].all()
        batch = batch[:, :self.n_current,:self.n_dim].cpu()
        with torch.no_grad():
            gen, fake_scaled, true_scaled = self.sampleandscale(batch,mask, scale=True)
            batch=batch*(~mask).unsqueeze(-1).float()
            scores_real = self.dis_net(batch, mask=mask[:len(true_scaled)], weight=False)[0]
            scores_fake = self.dis_net(gen[:len(true_scaled)], mask=mask[:len(true_scaled)], weight=False )[0]
        fake_scaled[mask]=0
        true_scaled[mask[:len(true_scaled)]]=0
        for i in range(self.n_current):
            fake_scaled[fake_scaled[:, i,2] < 0, i,2] = true_scaled[:,:,2].min()
        fake_scaled[:,:,:3] = center_jets_tensor(fake_scaled[:,:,:3][...,[2,0,1]])[...,[1,2,0]]
        fake_scaled[mask]=0
        true_scaled[mask[:len(true_scaled)]]=0
        self.calc_log_metrics(fake_scaled,true_scaled,scores_real,scores_fake)

    def calc_log_metrics(self, fake_scaled,true_scaled,scores_real,scores_fake):
        # data_ms = mass(true_scaled).numpy()
        # i=0
        # w_dist_list = []
        # for _ in range(10):
        #     gen_ms = mass(fake_scaled[i:i+len(true_scaled)]).numpy()
        #     i += len(true_scaled)
        #     w_dist_ms = wasserstein_distance(data_ms, gen_ms)
        #     w_dist_list.append(w_dist_ms)
        w1m_ = w1m(fake_scaled,true_scaled, num_eval_samples= 25000,num_batches=5)[0]
        if  w1m_<0.0006 and self.n_current<self.n_part :#
            self.n_current+=self.part_increase
            self.fadein=0
            self.data_module.n_part=self.n_current
            self.w1m_best=100
            self.mean_field_loss=True
            print("number particles increased to {}".format(self.n_current))

        try:
            cov, mmd = cov_mmd( true_scaled,fake_scaled[:len(true_scaled)], use_tqdm=False)
            self._log_dict = {"mmd": mmd,"cov": cov,}
        except:
            self._log_dict = {}
            print("cov mmd failed")

        self.log("w1m",w1m_, on_step=False, prog_bar=False, logger=True)
        #self._log_dict["w1m"]=w1m_
        if w1m_<self.w1m_best:
            self.w1m_best=w1m_
            self._log_dict["n_current"]=self.n_current
            self._log_dict["best_w1m"]=w1m_
            self.plot = plotting_point_cloud(model=self,gen=fake_scaled[:len(true_scaled)],true=true_scaled,n_dim=self.n_dim, n_part=self.n_current, step=self.global_step,logger=self.logger, n=self.n_current,p=self.parton)
            try:
                self.plot.plot_scores(scores_real.reshape(-1).numpy(),scores_fake.reshape(-1).numpy(),False,self.global_step)
                self.plot.plot_mass(save=None, bins=50)
            except Exception as e:
                traceback.print_exc()
        if self.n_part==30 and w1m_<0.001 or self.w1m_best==w1m_ and self.n_part==30 :
                fpnd_best=fpnd(fake_scaled,jet_type="t",use_tqdm=False)
                self._log_dict["fpnd"]=fpnd_best
        self._log_dict["n_current"]=self.n_current
        self.logger.log_metrics(self._log_dict, step=self.global_step)
        self._log_dict["w1m"]=w1m_
        print("epoch {}: ".format(self.current_epoch), self._log_dict)