import datetime
import os
print(os.system('hostname'))
import sys
import time
import traceback

sys.path.insert(1, "/home/kaechben/ProGamer")
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import yaml
# from pytorch_lightning.plugins.environments import SLURMEnvironment
from helpers import *
from jetnet_dataloader import JetNetDataloader
from progamer import ProGamer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import (CometLogger, TensorBoardLogger,
                                       WandbLogger)
from pytorch_lightning.tuner.tuning import Tuner
from scipy import stats
from torch.nn import functional as FF

# from plotting import plotting

# from comet_ml import Experiment
def lcm(a,b):
  return (a * b) // math.gcd(a,b)

def train(config,  load_ckpt=False):
    # This function is a wrapper for the hyperparameter optimization module called ray
    # Its parameters hyperopt and load_ckpt are there for convenience
    # Config is the only relevant parameter as it sets the trainings hyperparameters
    # hyperopt:whether to optimizer hyper parameters - load_ckpt: path to checkpoint if used
    
    # if  config["load_ckpt_trafo"]:
    
    #     model = ProGamer.load_from_checkpoint(config["load_ckpt_trafo"])
    
    
    # Callbacks to use during the training, we  checkpoint our models

    logger = WandbLogger(save_dir="/beegfs/desy/user/kaechben/pf_"+config["parton"],sync_tensorboard=True,
                project="progamer_top",)# TensorBoardLogger(root)#,version=version_name
    callbacks = [
    EarlyStopping(monitor="w1m", min_delta=0.00, patience=40,  mode="min",divergence_threshold=10,verbose=True),
    ModelCheckpoint(monitor="w1m",save_top_k=3,mode="min",filename="{epoch}-{fpnd:.3f}-{w1m:.4f}--{w1efp:.6f}",every_n_epochs=10),
    ModelCheckpoint(monitor="fpnd",save_top_k=3,mode="min",filename="{epoch}-{fpnd:.3f}-{w1m:.4f}--{w1efp:.6f}",every_n_epochs=10)]
    if config["n_part"]<30:
        config["load_ckpt"]="/beegfs/desy/user/kaechben/pf_t/pointflow/1pn8k3fp/checkpoints/epoch=109-w1p=0.0010768-w1m=0.0172--w1efp=0.001217.ckpt"
    else:
        config["load_ckpt"]= "/beegfs/desy/user/kaechben/pointflow_t/epoch=5549-val_fpnd=57.51-val_w1m=0.0094-val_w1efp=0.000221-val_w1p=0.00085.ckpt"
    if len(logger.experiment.config.keys())>0:
        config.update(**logger.experiment.config)
        config["l_dim"]=lcm(config["l_dim"],config["heads"])
        config["l_dim_gen"]=lcm(config["l_dim_gen"],config["heads_gen"])
        config["lr_d"]=config["lr_g"]*config["ratio"]
        print(config["lr_d"],config["lr_g"],config["ratio"])
    print(logger.experiment.dir)
    # if not config["load_ckpt_trafo"]:
    data_module = JetNetDataloader(config)
    data_module.setup("training")
    print("config:", config)
    model = ProGamer(num_batches= data_module.num_batches,**config)
    # else:
    #     print("model loaded")
    #     ckpt=config["load_ckpt_trafo"]
    #     model=ProGamer.load_from_checkpoint(config["load_ckpt_trafo"]).eval()

      # this loads the data
    model.data_module = data_module
      # the sets up the model,  config are hparams we want to optimize
    trainer = pl.Trainer(
        gpus=1,
        logger=logger,
        log_every_n_steps=data_module.num_batches//2,  # auto_scale_batch_size="binsearch",
        max_epochs=config["max_epochs"]*10,
        callbacks=callbacks,
        progress_bar_refresh_rate=0,
        check_val_every_n_epoch=config["val_check"],
        num_sanity_val_steps=0,  # gradient_clip_val=.02, 
        fast_dev_run=False,
        track_grad_norm=1,
        default_root_dir="/beegfs/desy/user/kaechben/pf_"+config["parton"],
        reload_dataloaders_every_n_epochs=0,#,config["val_check"] if not config["smart_batching"] else 0,
        #profiler="pytorch"
    )
    print(trainer.default_root_dir)
    # This calls the fit function which trains the model
    print("This is run: ",logger.experiment.name)
    trainer.fit(model, datamodule=data_module,)#,ckpt_path=ckpt,ckpt_path=config["load_ckpt_trafo"],

if __name__ == "__main__":


    parton=np.random.choice(["t"])#"q","g",
  


    config = { 
        "val_check": 25,
        "parton": parton,
        "warmup": 1200,
        "sched": "linear",
        "freq": 5,
        "batch_size": 2048,
        "dropout": 0.01,
        "opt": "Adam",
        "lr_g": 0.001,
        "lr_d": 0.001,
        "l_dim": 100,
        'l_dim_gen':20, 
        'hidden_gen':False,
        'heads_gen':4,
        "no_hidden_gen": False,
        "hidden": 512,
        "max_epochs": 1200,
        "name": "ProGamer",
        "n_part": 30,
        "n_start":30,
        "n_dim": 3,
        "heads": 2,
        "flow_prior": True,
        #"load_ckpt_trafo":True,#'/home/kaechben/ProGamer/start_fpnd_022_w1m_08.ckpt',
        "add_corr":True,
        "num_layers":4,
        "latent":None,
        "activation":"gelu",
        "activation_gen":"gelu",
        "smart_batching":True,
        "aux":False,
        "proj":True,
        "dropout_gen":0.3,
        "proj_gen":False,
        "num_layers_gen":4,
        "swa":False,
        "swa_gen":False,
        "affine":True,
        "slope":0.2,
        "aux":False,
        "bias":True,
        "beta1":0.0,
        "beta2":0.0
        }
    config["parton"] =parton
    train(config, )#load_ckpt=ckptroot=root,
