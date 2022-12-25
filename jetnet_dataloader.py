import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.preprocessing import QuantileTransformer
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
        


    # def smart_batching(self,data):
    #     n=self.data[:,:,-1].sum(1)


    #     for i in range(len(data[:])):
    #         del data[i][self.n_part-int(n[i]):]

    #     data=sorted(data,key=lambda x: len(x))
    #     sorted_lengths = [len(s) for s in data]



    #     # List of batches that we'll construct.
    #     batch_ordered_sentences = []
    #     batch_size=self.batch_size
    #     print('Creating training batches of size {:}'.format(batch_size))

    #     # Loop over all of the input samples...    
    #     while len(data) > 0:
            
    #         # Report progress.
    #         if ((len(batch_ordered_sentences) % 500) == 0):
    #             print('  Selected {:,} batches.'.format(len(batch_ordered_sentences)))

    #         # `to_take` is our actual batch size. It will be `batch_size` until 
    #         # we get to the last batch, which may be smaller. 
    #         to_take = min(batch_size, len(data))

    #         # Pick a random index in the list of remaining samples to start
    #         # our batch at.
    #         select = random.randint(0, len(data) - to_take)

    #         # Select a contiguous batch of samples starting at `select`.
    #         batch = data[select:(select + to_take)]

    #         # Each sample is a tuple--split them apart to create a separate list of 
    #         # sequences and a list of labels for this batch.
    #         batch_ordered_sentences.append([s for s in batch])
    #         # batch_ordered_labels.append([s[1] for s in batch])

    #         # Remove these samples from the list.
    #         del data[select:select + to_take]
    #     py_inputs = []
    #     # For each batch...
    #     for batch_inputs in batch_ordered_sentences:

    #         # New version of the batch, this time with padded sequences and now with
    #         # attention masks defined.
    #         batch_padded_inputs = []
    #         batch_attn_masks = []
            
    #         # First, find the longest sample in the batch. 
    #         # Note that the sequences do currently include the special tokens!
    #         max_size = max([len(sen) for sen in batch_inputs])

    #         #print('Max size:', max_size)

    #         # For each input in this batch...
    #         for sen in batch_inputs:
    #             # How many pad tokens do we need to add?
    #             num_pads = max_size - len(sen)
                
    #             # Add `num_pads` padding tokens to the end of the sequence.
    #             padded_input = sen + [[0,0,0,1]]*num_pads

    #             # Add the padded results to the batch.
    #             batch_padded_inputs.append(padded_input)
    #     # break
    #         # Our batch has been padded, so we need to save this updated batch.
    #         # We also need the inputs to be PyTorch tensors, so we'll do that here.
    #         py_inputs.append(torch.tensor(batch_padded_inputs))
    #         # py_attn_masks.append(torch.tensor(batch_attn_masks))
    #     return py_inputs
    #     # py_attn_masks.append(torch.tensor(batch_attn_masks))
    #     # Get the new list of lengths after sorting.
    #     # max_len=150
    #     # padded_lengths = []

    #     # # For each batch...
    #     # for batch in py_inputs:
            
    #     #     # For each sample...
    #     #     for s in batch:
            
    #     #         # Record its length.
    #     #         padded_lengths.append(len(s))

    #     # # Sum up the lengths to the get the total number of tokens after smart batching.
    #     # smart_token_count = np.sum(padded_lengths)

    #     # # To get the total number of tokens in the dataset using fixed padding, it's
    #     # # as simple as the number of samples times our `max_len` parameter (that we
    #     # # would pad everything to).
    #     # fixed_token_count = len(data) * max_len

    #     # # Calculate the percentage reduction.
    #     # prcnt_reduced = (fixed_token_count - smart_token_count) / float(fixed_token_count) 

    #     print('Total tokens:')
    #     print('   Fixed Padding: {:,}'.format(fixed_token_count))
    # print('  Smart Batching: {:,}  ({:.1%} less)'.format(smart_token_count, prcnt_reduced)
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