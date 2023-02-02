import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.preprocessing import QuantileTransformer,PowerTransformer
from helpers import *
import jetnet
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset,Dataset
def custom_collate(data): #(2)
        # x=torch.cat(torch.unsqueeze(data,0),)
        
        data=torch.stack(data)
        n=(~data[:,:,-1].bool()).sum(1).max()

        return data[:,:int(n)]
class StandardScaler:
    def __init__(self, mean=None, std=None, epsilon=1e-7):
        """Standard Scaler.
        The class can be used to normalize PyTorch Tensors using native 
        functions. The module does not expect the tensors to be of any specific shape;
         as long as the features are the last dimension in the tensor, the module
        will work fine.
        :param mean: The mean of the features. The property will be set after a call to fit.
        :param std: The standard deviation of the features. The property will be set after a call to fit.
        :param epsilon: Used to avoid a Division-By-Zero exception.
        """
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def fit(self, values):
        dims = list(range(values.dim() - 1))
        self.mean = torch.mean(values, dim=dims)
        self.std = torch.std(values, dim=dims)

    def transform(self, values):
        return (values - self.mean) / (self.std + self.epsilon)

    def inverse_transform(self, values):
        return (values * self.std) + self.mean

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

    def to(self, dev):
        self.std = self.std.to(dev)
        self.mean = self.mean.to(dev)
        return self

class PowerScaler:
    def __init__(self):
        """Standard Scaler.
        The class can be used to normalize PyTorch Tensors using native 
        functions. The module does not expect the tensors to be of any specific shape;
         as long as the features are the last dimension in the tensor, the module
        will work fine.
        :param mean: The mean of the features. The property will be set after a call to fit.
        :param std: The standard deviation of the features. The property will be set after a call to fit.
        :param epsilon: Used to avoid a Division-By-Zero exception.
        """
        self.scaler=PowerTransformer("box-cox")
    def transform(self,values):
        dev=values.device
        values=values.detach().cpu().numpy()
        values=torch.tensor(self.scaler.transform(values)).float().to(dev)
        return values
    def inverse_transform(self,values):
        dev=values.device

        values=values.detach().cpu().numpy()
        
        values=np.clip(values,-1/abs(self.scaler.lambdas_)+1e-10,1/abs(self.scaler.lambdas_)-1e-10)
        values=torch.tensor(self.scaler.inverse_transform(values)).float().to(dev)
        return values
    def fit(self,values):
        self.scaler.fit(values.detach().cpu().numpy())
        
        print("lambdas:",self.scaler.lambdas_)
    def fit_transform(self,values):
        self.fit(values)
        return self.transform(values)


    
class JetNetDataloader(pl.LightningDataModule):
    """This is more or less standard boilerplate coded that builds the data loader of the training
    one thing to note is the custom standard scaler that works on tensors
   """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_dim = config["n_dim"]

        self.n_part = config["n_part"]
        self.batch_size = config["batch_size"]
        
        
    def setup(self, stage):
        # This just sets up the dataloader, nothing particularly important. it reads in a csv, calculates mass and reads out the number particles per jet
        # And adds it to the dataset as variable. The only important thing is that we add noise to zero padded jets
        data,_=jetnet.datasets.JetNet.getData(jet_type=self.config["parton"],split="train",num_particles=self.n_part,data_dir="/beegfs/desy/user/kaechben/datasets")
        test_set,_=jetnet.datasets.JetNet.getData(jet_type=self.config["parton"],split="valid",num_particles=self.n_part,data_dir="/beegfs/desy/user/kaechben/datasets")
        data=torch.tensor(data)
        self.num_batches=len(data)/self.config["batch_size"]
        test_set=torch.tensor(test_set)
        self.data=torch.cat((data,test_set),dim=0)
        masks=~self.data[:,:,-1].bool()
        self.n = self.data[:,:,-1].sum(axis=1)       
        self.data[:,:,-1] = ~(self.data[:,:,-1].bool())
        self.scalers=[]
        self.scaler=StandardScaler()
        temp=self.data[:,:,:-1].reshape(-1,self.n_dim)
        temp[masks.reshape(-1)==0]=self.scaler.fit_transform(temp[masks.reshape(-1)==0,:])
        self.data[:,:,:-1]=temp.reshape(-1,self.n_part,self.n_dim)
        self.test_set = self.data[-len(test_set):].float()
        self.data = self.data[:-len(test_set)].float()



    def scale(self,data,mask):
        return self.scaler.inverse_transform(data)

        
    def train_dataloader(self):
        
        # batch_size={"0":[128],"1":[1024],"31":[512],"50":[128]}

        # if self.n_current<=30:
        #     n="1"
        # elif self.n_current>=50:
        #     n="1"
        # elif self.n_current>=31:
        #     n="1"
        
        
        # smart_batching(self.data)
       
        return DataLoader(self.data, collate_fn=custom_collate,batch_size=self.config["batch_size"],num_workers=40)

    def val_dataloader(self):
        return DataLoader(self.test_set[:,:self.n_part], batch_size=len(self.test_set), drop_last=True,num_workers=40)
    def custom_collate(data): #(2)
        masks=~(data[:,:,-1].bool())
        sel=torch.ones(len(data),data.sum(1).max())
        inputs = data[sel]
        return inputs
        

if __name__=="__main__":
    config = { 
        "n_part": 150,
        "n_dim": 3,
        "batch_size": 1024,
        "parton": "t",
        "smart_batching":True
     }
    x=JetNetDataloader(config)
    x.setup("train")