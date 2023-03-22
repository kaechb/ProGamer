import datetime
import os

print(os.system("hostname"))
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

# from pytorch_lightning.plugins.environments import SLURMEnvironment
from helpers import *
from jetnet_dataloader import JetNetDataloader
from progamer_re import ProGamer

# from plotting import plotting


# from comet_ml import Experiment
def lcm(a, b):
    return (a * b) // math.gcd(a, b)


def train(config, load_ckpt=False):

    # This function is a wrapper for the hyperparameter optimization module called ray
    # Its parameters hyperopt and load_ckpt are there for convenience
    # Config is the only relevant parameter as it sets the trainings hyperparameters
    # hyperopt:whether to optimizer hyper parameters - load_ckpt: path to checkpoint if used
    # Callbacks to use during the training, we  checkpoint our models

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
            every_n_epochs=10,
        ),
    ]

    if len(logger.experiment.config.keys()) > 0:
        config.update(**logger.experiment.config)
        # config["l_dim"] = lcm(config["l_dim"], config["heads"])
        # config["l_dim_gen"] = lcm(config["l_dim_gen"], 1)
        # config["lr_d"] = config["lr_g"]
        # print(config["lr_d"], config["lr_g"])
    # if load_ckpt:
    #     model = ProGamer.load_from_checkpoint(
    #         load_ckpt, strict=False, num_prog=config["num_prog"]
    #     )
    print(logger.experiment.dir)
    # if not config["load_ckpt_trafo"]:
    data_module = JetNetDataloader(config)
    #data_module.setup("training")
    print("config:", config)
    if not load_ckpt:
        config["spectral"] = False
        model = ProGamer( config=config, **config)
    else:
        model.n_part = config["n_part"]
        model.n_current = config["n_start"]

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
        log_every_n_steps=100,  # auto_scale_batch_size="binsearch",
        max_epochs=config["max_epochs"] * 10,
        callbacks=callbacks,
        # progress_bar_refresh_rate=0,
        val_check_interval=1000,  # config["val_check"],
        check_val_every_n_epoch=None,
        num_sanity_val_steps=1,  # gradient_clip_val=.02,
        enable_progress_bar=False,
        track_grad_norm=1,
        reload_dataloaders_every_n_epochs=1,
        #
        # fast_dev_run=False,
        # track_grad_norm=1,
        default_root_dir="/beegfs/desy/user/kaechben/pf_" + config["parton"],
        # reload_dataloaders_every_n_epochs=0,#,config["val_check"] if not config["smart_batching"] else 0,
        # profiler="pytorch"
    )
    print(trainer.default_root_dir)
    # This calls the fit function which trains the model
    print("This is run: ", logger.experiment.name)
    trainer.fit(
        model,
        datamodule=data_module
    )  # ,ckpt_path=ckpt,ckpt_path=config["load_ckpt_trafo"],


if __name__ == "__main__":

    parton = np.random.choice(["t"])  # "q","g",

    config = {
        # "activation_gen": "leaky",
        # "activation": "leaky",
        "batch_size": 128,
        # "beta1": 0.0,
        # "beta2": 0.999,
        "dropout_gen": 0,
        "dropout": 0,
        "gan": "ls",
        "heads": 2,
        "heads_gen": 8,
        "hidden_gen": 128,
        "hidden": 128,
        "l_dim_gen": 128,
        "l_dim": 128,
        "lr_d": 0.0001,
        "lr_g": 0.0001,
        "max_epochs": 1200,
        "n_dim": 3,
        "n_part":30,
        "n_start": 30,
        "name": "ProGamer",
        "num_layers_gen": 6,
        "num_layers": 3,
        "opt": "Adam",
        "parton": parton,
        "cond_dim": 6,
        "swa": False,
        "swagen": False,
        "epic":False

    }
    # "load_ckpt_trafo":True,#'/home/kaechben/ProGamer/start_fpnd_022_w1m_08.ckpt',

    config["parton"] = parton
    ckpt = "/beegfs/desy/user/kaechben/pf_t/ProGamer/clhdrd17/checkpoints/epoch=3599-fpnd=0.249-w1m=0.0006--w1efp=0.000014.ckpt"
    train(config, load_ckpt=False)  # load_ckpt=ckptroot=root,
