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
from progamer import ProGamer
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
    
    callbacks = [
        EarlyStopping(monitor="w1m", min_delta=0.00, patience=60,  mode="min",divergence_threshold=10,verbose=True),
        
        ModelCheckpoint(
            monitor="w1m",
            save_top_k=3,
            mode="min",
            filename="{epoch}-{fpnd:.3f}-{w1m:.4f}--{w1efp:.6f}",
            #dirpath=root,
            every_n_epochs=10,
        ),
        ModelCheckpoint(
            monitor="fpnd",
            save_top_k=3,
            mode="min",
            filename="{epoch}-{fpnd:.3f}-{w1m:.4f}--{w1efp:.6f}",
            #dirpath=root,
            every_n_epochs=10,
        )
    ]
    logger = WandbLogger(save_dir="/beegfs/desy/user/kaechben/pf_"+config["parton"],sync_tensorboard=True,project="progamer_top")# TensorBoardLogger(root)#,version=version_name
    # for key in logger.experiment.config.keys():
    #     config[key]=logger.experiment.config[key]
    # tags=[]
    # if config["cls"]:
    #     tags=tags+["cls"]
    # if config["swa"]:
    #     tags=tags+["swa"]
    # tags=tags+[str(config["opt"])]+[str(config["sched"])]+[str(config["n_part"])]
    # print(logger.experiment.dir)
    data_module = JetNetDataloader(config)
    data_module.setup("training")
    config["l_dim"]=config["l_dim"]*config["heads"]
    config["l_dim_gen"]=config["l_dim_gen"]*config["heads_gen"]
    config["lr_d"]=config["lr_g"]
    if config["n_part"]==30:
        config["load_ckpt"]="/beegfs/desy/user/kaechben/pf_t/pointflow/1pn8k3fp/checkpoints/epoch=109-w1p=0.0010768-w1m=0.0172--w1efp=0.001217.ckpt"
    else:
        config["load_ckpt"]= "/beegfs/desy/user/kaechben/pointflow_t/epoch=5549-val_fpnd=57.51-val_w1m=0.0094-val_w1efp=0.000221-val_w1p=0.00085.ckpt"
    print("config:", logger.experiment.config)
    model = ProGamer(
        config,  data_module.num_batches
    )
    # else:
    #     print("model loaded")
    #     ckpt=config["load_ckpt_trafo"]
    #     model=ProGamer.load_from_checkpoint(config["load_ckpt_trafo"]).eval()
    # config=model.config

    # model.config=config
    #   # this loads the data
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
        num_sanity_val_steps=1,  # gradient_clip_val=.02, 
        fast_dev_run=False,
        # track_grad_norm=0,
        
        default_root_dir=root,
        reload_dataloaders_every_n_epochs=0#,config["val_check"] if not config["smart_batching"] else 0,
    )
    # This calls the fit function which trains the model
    print("This is run: ",logger.experiment.name)
    trainer.fit(model, datamodule=data_module,)#,ckpt_path=ckpt,ckpt_path=config["load_ckpt_trafo"],

if __name__ == "__main__":


    parton=np.random.choice(["t"])#"q","g",
  

    hyperopt=True
    config = { 
        "val_check": 100,
        "parton": parton,
        "warmup": 1200,
        "sched": "linear",
        "freq": 5,
        "batch_size": 1024,
        "dropout": 0.01,
        "opt": "Adam",
        "lr_g": 0.001,
        "ratio": 1,
        "l_dim": 100,
        "no_hidden_gen": False,
        "hidden": 512,
        "max_epochs": 1200,
        "name": "ProGamer",
        "n_part": 150,
        "n_start":150,
        "n_dim": 3,
        "heads": 2,
        "flow_prior": False,
        "load_ckpt_trafo":"/beegfs/desy/user/kaechben/pf_t/ProGamer/y4osepxa/checkpoints/epoch=2349-fpnd=1000.000-w1m=0.0049--w1efp=0.000251.ckpt",#'/home/kaechben/ProGamer/start_fpnd_022_w1m_08.ckpt',
        "swa":True,
        "swagen":True,
        "add_corr":True,
        "frac_pretrain":0.05,
        "cls":True,
        "num_layers":4,
        "latent":50,
        "activation":"gelu",
        "smart_batching":True,
        "aux":False,
        "proj":True,
        "latent":True,
        "l_dim_gen":5,
        "heads_gen":1,
        "hidden_gen":4,
        "num_layers_gen":5,
        "dropout_gen":0,
        "activation_gen":"gelu",
        "proj_gen":True,
        }
    # c={'activation': 'gelu', 'activation_gen': 'gelu', 'add_corr': True, 'dropout': 0.4, 'dropout_gen': 0.05, 'flow_prior': True, 'heads': 4, 'heads_gen': 8, 'hidden': 256, 'hidden_gen': 256, 'l_dim': 10, 'l_dim_gen': 5, 'latent': 50, 'lr_g': 0.0001, 'n_part': 150, 'num_layers': 4, 'num_layers_gen': 4, 'opt': 'Adam', 'proj': True, 'proj_gen': False, 'sched': 'linear'}
    # for key in c.keys():
    #     config[key]=c[key]
    config["parton"] =parton
    root = "/beegfs/desy/user/" + os.environ["USER"] + "/"+ config["name"]
    train(config, root=root,)#load_ckpt=ckpt
   