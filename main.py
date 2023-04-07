import datetime
import os
import sys
import time
import traceback

import wandb

sys.path.insert(1, "/home/kaechben/ProGamer")
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger, WandbLogger
from pytorch_lightning.tuner.tuning import Tuner
from scipy import stats
from torch.nn import functional as FF
import torch
# from pytorch_lightning.plugins.environments import SLURMEnvironment
from helpers import *
from jetnet_dataloader import JetNetDataloader
from progamer_re import ProGamer
from tqdm import tqdm
# from plotting import plotting


# from comet_ml import Experiment
def lcm(a, b):
    return (a * b) // math.gcd(a, b)


def train(config, ckpt=False,logger=None):
    torch.set_float32_matmul_precision('medium' )
    # This function is a wrapper for the hyperparameter optimization module called ray
    # Its parameters hyperopt and ckpt are there for convenience
    # Config is the only relevant parameter as it sets the trainings hyperparameters
    # hyperopt:whether to optimizer hyper parameters - ckpt: path to checkpoint if used
    # Callbacks to use during the training, we  checkpoint our models



    if not ckpt:
        model = ProGamer( config=config, **config)
        # if model.n_part<150:
        #     model.fine_tune=False
        # else:
        #     model.fine_tune = True
        # model.fine_tune = config["fine_tune"]
        model.continue_training=False
        # model.mean_field_loss=config["mean_field_loss"]

    else:
        print("model loaded")
        model=ProGamer.load_from_checkpoint(ckpt)
        model.parton=config["parton"]
        model.ckpt=ckpt
        model.d_loss_mean=0
        model.g_loss_mean=0
        model.mean_field_loss=False#config["mean_field_loss"]
        model.fine_tune=True#config["fine_tune"]
        model.continue_training=True
        #torch.nn.init.normal(model.dis_net.out.weight)

        logger.log_hyperparams({"mean_field_loss":model.mean_field_loss,"fine_tune":model.fine_tune,"parton":model.parton,"ckpt":model.ckpt})

    # this loads the data
    data_module = JetNetDataloader(config,finetune=model.fine_tune)
    data_module.setup("validation")
    model.data_module = data_module

    # the sets up the model,  config are hparams we want to optimize
    trainer = pl.Trainer(
        devices=1,accelerator="gpu",
        logger=logger,
        log_every_n_steps=100,  # auto_scale_batch_size="binsearch",
        max_epochs=20000,
        callbacks=callbacks,
        # progress_bar_refresh_rate=0,
        val_check_interval=5000,  # config["val_check"],
        check_val_every_n_epoch=None,
        num_sanity_val_steps=0,  # gradient_clip_val=.02,
        enable_progress_bar=False,
        # profiler="simple",
       # reload_dataloaders_every_n_epochs=1,
        #
        # fast_dev_run=False,
        # track_grad_norm=1,
        default_root_dir="/beegfs/desy/user/kaechben/pf_" + config["parton"],
        # reload_dataloaders_every_n_epochs=0,#,config["val_check"] if not config["smart_batching"] else 0,
        #profiler="pytorch"
    )
    print(trainer.default_root_dir)
    #model.mean_field_loss=False
    # This calls the fit function which trains the model
    print("This is run: ", logger.experiment.name)
    # model.gen_net= torch.compile(model.gen_net, )
    # model.dis_net= torch.compile(model.dis_net, )
    trainer.fit(model,datamodule=data_module,ckpt_path=ckpt)  # ckpt_path=config["ckpt_trafo"],,,


if __name__ == "__main__":

    parton = np.random.choice(["t"])  # "q","g",

    config = {
        # "activation_gen": "leaky",
        # "activation": "leaky",
        "batch_size": 256,
        "part_increase": 10,
        # "beta1": 0.0,
        # "beta2": 0.999,
        "dropout_gen": 0,
        "dropout": 0.1,
        "gan": "ls",
        "heads": 8,
        "heads_gen": 16,
        "hidden_gen": 128,
        "hidden": 64,
        "l_dim_gen": 32,
        "l_dim": 32,
        "lr_d": 0.0001,
        "lr_g": 0.0001,
        "max_epochs": 2000,
        "n_dim": 3,
        "n_part":150,
        "n_start": 150,
        "name": "ProGamer",
        "num_layers_gen": 6,
        "num_layers": 2,
        "opt": "Adam",
        "parton": parton,
        "checkpoint":True,
        "mean_field_loss":False,
        "fine_tune":False,
        "act": "leaky",
        "stop_mean":True,
        "mass":False,
        "ckpt":None



    }
    # "load_ckpt_trafo":True,#'/home/kaechben/ProGamer/start_fpnd_022_w1m_08.ckpt',
    wandb.init( )
    logger = WandbLogger(
        save_dir="/beegfs/desy/user/kaechben/pf_" + config["parton"],
        sync_tensorboard=False,
        project="linear",
    )  #
    logger.experiment.log_code(".")
    callbacks = [
        ModelCheckpoint(
            monitor="w1m",
            save_top_k=3,
            mode="min",
            filename="{epoch}-{fpnd:.3f}-{w1m:.4f}--{w1efp:.6f}",
            every_n_epochs=1,
        ),
    ]

    if len(logger.experiment.config.keys()) > 0:
        config.update(**logger.experiment.config)
    print(logger.experiment.dir)
    print("config:", config)
    #ckpt ="/beegfs/desy/user/kaechben/pf_t/linear/n4wdcpj0/checkpoints/epoch=406-fpnd=0.000-w1m=0.0013--w1efp=0.000000.ckpt"
     #ckpt="/beegfs/desy/user/kaechben/pf_t/linear/qobm3g4x/checkpoints/epoch=49-fpnd=0.000-w1m=0.0010--w1efp=0.000000.ckpt"
    # ckpt="/beegfs/desy/user/kaechben/pf_t/linear/rimc1ylb/checkpoints/epoch=74-fpnd=0.000-w1m=0.0017--w1efp=0.000000.ckpt"
    if config["ckpt"]:
        ckpt="/beegfs/desy/user/kaechben/pf_t/linear/nqxv9qxz/checkpoints/epoch=1221-fpnd=0.000-w1m=0.0006--w1efp=0.000000.ckpt"
    ckpt=None
    train(config, ckpt=ckpt,logger=logger)  # load_ckpt=ckptroot=root,
