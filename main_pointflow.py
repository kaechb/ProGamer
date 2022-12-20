import datetime
import os
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
from pointflow import PF
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import (CometLogger, TensorBoardLogger,
                                       WandbLogger)
from pytorch_lightning.tuner.tuning import Tuner
from scipy import stats
from torch.nn import functional as FF

def train(config,  load_ckpt=False, i=0, root=None):
    # This function is a wrapper for the hyperparameter optimization module called ray
    # Its parameters hyperopt and load_ckpt are there for convenience
    # Config is the only relevant parameter as it sets the trainings hyperparameters
    # hyperopt:whether to optimizer hyper parameters - load_ckpt: path to checkpoint if used
    
    callbacks = [ModelCheckpoint(monitor="w1p",save_top_k=3,mode="min",filename="{epoch}-{w1p:.7f}-{w1m:.4f}--{w1efp:.6f}",every_n_epochs=10,)
    ]
    logger = WandbLogger(save_dir="/beegfs/desy/user/kaechben/pf_"+config["parton"],sync_tensorboard=True,
                project="pointflow")# TensorBoardLogger(root)#,version=version_name
    for key in logger.experiment.config.keys():
        config[key]=logger.experiment.config[key]

    print(logger.experiment.dir)
    data_module = JetNetDataloader(config)
    data_module.setup("training")
   
    config["lr_nf"]
    model = PF(
        config,  data_module.num_batches
    )
    config=model.config
    model.config=config
      # this loads the data
    model.data_module = data_module
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
        default_root_dir=root,
        reload_dataloaders_every_n_epochs=config["val_check"] if not config["smart_batching"] else 0,
    )
    # This calls the fit function which trains the model
    print("This is run: ",logger.experiment.name)
    trainer.fit(model, datamodule=data_module)#,ckpt_path=ckpt

if __name__ == "__main__":


    parton=np.random.choice(["t"])#"q","g",
    hyperopt=True
    config = { 
        "val_check": 5,
        "parton": parton,
        "warmup": 1200,
        "sched": "linear",
        "batch_size": 2048,
        "dropout": 0.01,
        "opt": "Adam",
        "lr_nf": 0.001,
        "max_epochs":120,
        "hidden": 80,
        "name": "PointFlow",
        "n_part": 30,
        "n_start":30,
        "n_dim": 3,
        "swa":True,
        "swagen":True,
        "network_nodes_nf":128,
        "coupling_layers":8,
        "network_layers_nf":2,
        "tail_bound":6,
        "bins":8,
        "smart_batching":True,
        "context_features":5
        }
    config["parton"] =parton
    root = "/beegfs/desy/user/" + os.environ["USER"] + "/"+ config["name"]
    train(config, root=root,)#load_ckpt=ckpt
   