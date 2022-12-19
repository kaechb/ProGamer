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

# from plotting import plotting

# from comet_ml import Experiment


def train(config,  load_ckpt=False, i=0, root=None):
    # This function is a wrapper for the hyperparameter optimization module called ray
    # Its parameters hyperopt and load_ckpt are there for convenience
    # Config is the only relevant parameter as it sets the trainings hyperparameters
    # hyperopt:whether to optimizer hyper parameters - load_ckpt: path to checkpoint if used
    
    # if  config["load_ckpt_trafo"]:
    
    #     model = PF.load_from_checkpoint(config["load_ckpt_trafo"])
    

    # Callbacks to use during the training, we  checkpoint our models

    callbacks = [
        
        ModelCheckpoint(
            monitor="w1p",
            save_top_k=3,
            mode="min",
            filename="{epoch}-{w1p:.7f}-{w1m:.4f}--{w1efp:.6f}",
            #dirpath=root,
            every_n_epochs=10,
        )
    ]
    logger = WandbLogger(save_dir="/beegfs/desy/user/kaechben/pf_"+config["parton"],sync_tensorboard=True,
                project="pointflow")# TensorBoardLogger(root)#,version=version_name
    for key in logger.experiment.config.keys():
        config[key]=logger.experiment.config[key]
    # tags=[]
    # if config["cls"]:
    #     tags=tags+["cls"]

    # if config["swa"]:
    #     tags=tags+["swa"]
    
    # tags=tags+[str(config["opt"])]+[str(config["sched"])]+[str(config["n_part"])]

   
    
    print(logger.experiment.dir)
    # if not config["load_ckpt_trafo"]:
    
    data_module = JetNetDataloader(config)
    data_module.setup("training")
   
    config["lr_nf"]
    model = PF(
        config,  data_module.num_batches
    )
    # else:
    #     print("model loaded")
    #     ckpt=config["load_ckpt_trafo"]
    #     model=PF.load_from_checkpoint(config["load_ckpt_trafo"]).eval()
    config=model.config

    model.config=config
      # this loads the data
    model.data_module = data_module
      # the sets up the model,  config are hparams we want to optimize
    # log every n steps could be important as it decides how often it should log to tensorboard
    # Also check val every n epochs, as validation checking takes some time
    
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
    
    # best_hparam="/home/kaechben/JetNet_NF/LitJetNet/LitNF/bestever_{}/hparams.yaml".format(parton)
    # with open(best_hparam, 'r') as stream:
    #     config=yaml.load(stream,Loader=yaml.Loader)
    #     config=config["config"]

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
        "n_part": 150,
        "n_start":150,
        "n_dim": 3,
        "swa":True,
        "swagen":True,
        "network_nodes_nf":128,
        "coupling_layers":8,
        "network_layers_nf":2,
        "tail_bound":6,
        "bins":8,
        "smart_batching":True
        
        }
    config["parton"] =parton

    # start a run
    # wandb.init(config=config,project="PF_top")
    # update any values not set by sweep

    # if hyperopt:

    #     # config["no_hidden"]=np.random.choice([True,False,"more"])
    #     # config["no_hidden"]=config["no_hidden"]=="True" or config["no_hidden"]=="more"

        
    #     #config["max_epochs"]=int(config["max_epochs"])#*np.random.choice([2]))
        
    #     config["sched"]=np.random.choice(["cosine2","linear"])
        
    #     config["freq"]=np.random.choice([5])    # config["opt"]="Adam"
    #     config["batch_size"]=int(np.random.choice([128,512]))    # config["opt"]="Adam"
    #     config["dropout"]=np.random.choice([0.15,0.05,0.01])    
    #     config["opt"]=np.random.choice(["Adam","RMSprop"])#"AdamW","mixed"
    #     config["lr_g"]=np.random.choice([0.0003,0.0001])  
    #     config["ratio"]=np.random.choice([0.9,1,1.1,])
    #     # config["cls"]=np.random.choice([False,True])
    #     config["normfirst"]=np.random.choice([True])
        
    #     config["fc_disc"]=np.random.choice([True])
    #     config["swa"]=np.random.choice([True,False])
    #     config["swagen"]=np.random.choice([True,False])

    #     config["num_layers"]=np.random.choice([4])
    #     config["l_dim"]=np.random.choice([25,15,30])
    #     config["hidden"]=np.random.choice([512,756,1024])
    #     config["n_part"]=np.random.choice([30,50,80,150])

    #     config["heads"]=np.random.choice([4,5])
    #     config["val_check"]=5

    #     config["lr_d"]=config["lr_g"]*config["ratio"]
    #     config["l_dim"] = config["l_dim"] * config["heads"]      
    #     config["name"] = config["name"]+config["parton"]
    #     config["no_hidden_gen"]=np.random.choice([True,False,"more"])
    #     config["max_epochs"]=np.random.choice([1200])  
    #     config["warmup"]=int(np.random.choice([0.4,0.6,0.8])*config["max_epochs"])
    #     config["name"]="PF_"+parton
    #     config["frac_pretrain"]=0.05
    #     config["load_ckpt"]= "/beegfs/desy/user/kaechben/pointflow_t/epoch=5549-val_fpnd=57.51-val_w1m=0.0094-val_w1efp=0.000221-val_w1p=0.00085.ckpt"
    # else:

    #     print("hyperopt off"*100)
    #     config["name"]="bestever_"+parton#config["parton"]

    # if len(sys.argv) > 2:
    #     root = "/beegfs/desy/user/"+ os.environ["USER"]+"/"+config["name"]+"/"+config["parton"]+"_" +"run"+sys.argv[1]+"_"+str(sys.argv[2])
    # else:
    root = "/beegfs/desy/user/" + os.environ["USER"] + "/"+ config["name"]


    train(config, root=root,)#load_ckpt=ckpt
   