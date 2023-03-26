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
from torch.utils.data import DataLoader,TensorDataset,Dataset,Sampler
import numpy as np
from random import shuffle
import numpy as np
from torch.utils.data import IterableDataset

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
        self.std = torch.std(values, dim=dims)#/5

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



from torch.utils.data import Sampler, Dataset
from collections import OrderedDict
from random import shuffle

# class BucketDataset(Dataset):

#     def __init__(self, inputs, targets):
#         self.inputs = inputs
#         self.targets = targets

#     def __len__(self):
#         return len(self.inputs)

#     def __getitem__(self, index):
#         if self.targets is None:

#             return self.inputs[index]
#         else:
#             return self.inputs[index], self.targets[index]


# class BucketBatchSampler(Sampler):
#     # want inputs to be an array
#     def __init__(self, inputs, batch_size):
#         self.batch_size = batch_size
#         ind_n_len = []
#         for i, p in enumerate(inputs):
#             ind_n_len.append((i, p.shape[0]))
#         self.ind_n_len = ind_n_len
#         self.batch_list = self._generate_batch_map()
#         self.num_batches = len(self.batch_list)

#     def _generate_batch_map(self):
#         # shuffle all of the indices first so they are put into buckets differently
#         shuffle(self.ind_n_len)
#         # Organize lengths, e.g., batch_map[10] = [30, 124, 203, ...] <= indices of sequences of length 10
#         batch_map = OrderedDict()
#         for idx, length in self.ind_n_len:
#             if length not in batch_map:
#                 batch_map[length] = [idx]
#             else:
#                 batch_map[length].append(idx)
#         # Use batch_map to split indices into batches of equal size
#         # e.g., for batch_size=3, batch_list = [[23,45,47], [49,50,62], [63,65,66], ...]
#         batch_list = []
#         for length, indices in batch_map.items():
#             for group in [indices[i:(i + self.batch_size)] for i in range(0, len(indices), self.batch_size)]:
#                 batch_list.append(group)
#         return batch_list

#     def batch_count(self):
#         return self.num_batches

#     def __len__(self):
#         return len(self.ind_n_len)

#     def __iter__(self):
#         self.batch_list = self._generate_batch_map()
#         # shuffle all the batches so they arent ordered by bucket size
#         shuffle(self.batch_list)
#         for i in self.batch_list:
#             yield i
class Dataset_Bucketing(IterableDataset):

    def __init__(self, data, batch_size_max, shuffle=True):
        self.whole_set = data
        self.batch_size_max = batch_size_max
        self.shuffle = shuffle

    def __len__(self):
        ls = self.calc_lengths(self.whole_set)
        _, uni_frqs = np.unique(ls, return_counts=True)
        divisors = uni_frqs // self.batch_size_max
        return divisors.sum() + np.count_nonzero(uni_frqs % self.batch_size_max)

    def calc_lengths(self, data):   # how many particles per jet
        lengths = np.count_nonzero(data[...,0], axis=1)
        return lengths

    def make_bucket_dict(self, data, batch_size=128):
        ls = self.calc_lengths(data)
        uni_ls = np.unique(ls)
        bucket_dict = {}
        bucket_ids = []
        for i in range(len(uni_ls)):
            bucket = data[ls == uni_ls[i]]
            bucket = bucket[:,0:uni_ls[i],:]  # drop all zero padded values, assuming [batch, points, feats]
            sub_bucket_dict, bucket_id_ary = self.make_sub_buckets(bucket, batch_size=batch_size, bucket_id=i)
            bucket_dict[i] = sub_bucket_dict
            bucket_ids.append(bucket_id_ary)  # tupel of bucket and sub_bucket index
        bucket_ids = np.vstack(bucket_ids)
        return bucket_dict, bucket_ids

    def make_sub_buckets(self, data, batch_size=128, bucket_id=0):
        n_samples = len(data)
        n_batches = n_samples // batch_size
        if n_samples % batch_size != 0: # for the case of excess events not filling the whole batch_size
            n_batches +=1
        sub_bucket_dict = {}
        i = 0
        bucket_id_list = []
        for j in range(n_batches):
            sub_bucket = data[i:i+batch_size]
            sub_bucket_dict[j] = sub_bucket
            i += batch_size
            bucket_id_list.append([bucket_id, j])
        bucket_id_ary = np.vstack(bucket_id_list)
        return sub_bucket_dict, bucket_id_ary

    def get_batch(self):
        # get unique lengths of dataset and shuffle them
        if self.shuffle:
            permutation_dataset = np.random.permutation(len(self.whole_set))
            data = self.whole_set[permutation_dataset]
        else:
            data = self.whole_set

        bucket_dict, bucket_ids = self.make_bucket_dict(data, batch_size=self.batch_size_max)
        if self.shuffle:
            permutation = np.random.permutation(len(bucket_ids))
            bucket_ids = bucket_ids[permutation]

        i = 0  # bucket/sub_bucket index
        while True:
            if i >= len(bucket_ids):    # resetting the generator --> shuffel dataset, shuffle lengths
                if self.shuffle:
                    permutation_dataset = np.random.permutation(len(data))
                    data = data[permutation_dataset]

                bucket_dict, bucket_ids = self.make_bucket_dict(data, batch_size=self.batch_size_max)
                if self.shuffle:
                    permutation = np.random.permutation(len(bucket_ids))
                    bucket_ids = bucket_ids[permutation]
                i = 0
            else:    # normally, not resetting mode
                while i < len(bucket_ids):
                    j,k = bucket_ids[i]
                    batch = bucket_dict[j][k]
                    i += 1
                    yield batch

    def __iter__(self):
        return self.get_batch()

class JetNetDataloader(pl.LightningDataModule):
    """This is more or less standard boilerplate coded that builds the data loader of the training
    one thing to note is the custom standard scaler that works on tensors
   """

    def __init__(self, config,finetune=False):
        super().__init__()
        self.finetune=finetune
        self.config = config
        self.n_dim = config["n_dim"]
        self.n_part = config["n_part"]
        self.batch_size = config["batch_size"]
        self.n_start = config["n_start"]

    def setup(self, stage ,n=None ):
        # This just sets up the dataloader, nothing particularly important. it reads in a csv, calculates mass and reads out the number particles per jet
        # And adds it to the dataset as variable. The only important thing is that we add noise to zero padded jets

        if self.n_part==30:
            data=jetnet.datasets.JetNet.getData(jet_type=self.config["parton"],split="train",num_particles=self.n_part,data_dir="/beegfs/desy/user/kaechben/datasets")[0]
            test_set=jetnet.datasets.JetNet.getData(jet_type=self.config["parton"],split="valid",num_particles=self.n_part,data_dir="/beegfs/desy/user/kaechben/datasets")[0]
        else:
            data= np.load("/beegfs/desy/user/kaechben/datasets/{}_{}_train.npy".format(self.config["parton"],self.config["n_part"]))
            test_set= np.load("/beegfs/desy/user/kaechben/datasets/{}_{}_valid.npy".format(self.config["parton"],self.config["n_part"]))
            #test_set,_=jetnet.datasets.JetNet.getData(jet_type=self.config["parton"],split="valid",num_particles=self.n_part,data_dir="/beegfs/desy/user/kaechben/datasets")
        data=torch.tensor(data)[:,:,:]
        test_set=torch.tensor(test_set)
        self.data=torch.cat((data,test_set),dim=0)
        if self.n_part>30:
            self.data[:,:,-1]=~self.data[:,:,-1].bool()
        self.n = self.data[:,:,-1].sum(axis=1)
        masks=~(self.data[:,:,-1]).bool()
        self.scalers=[]
        self.scaler=StandardScaler()
        temp=self.data[:,:,:-1].reshape(-1,self.n_dim)
        temp[masks.reshape(-1)==0]=self.scaler.fit_transform(temp[masks.reshape(-1)==0,:])
        self.data[:,:,:-1]=temp.reshape(-1,self.n_part,self.n_dim)
        self.data[:,:,-1]=masks
        self.test_set = self.data[-len(test_set):].float()
        self.data = self.data[:-len(test_set)].float()

        # self.data_=[]
        # for i,m in zip(self.data,150-(self.data[:,:,-1].sum(1))):
        #     self.data_.append(i[:int(m),:].numpy())
        # if not self.finetune:
        #     self.data=np.array(self.data,dtype=object)

    def scale(self,data,mask):
        return self.scaler.inverse_transform(data)

    def train_dataloader(self):
        if not self.finetune:
            dataset = Dataset_Bucketing(self.data[:,:self.n_part].numpy(), self.config["batch_size"])
            dataloader = DataLoader(dataset, batch_size=None)
            return dataloader
            # bucket_dataset = BucketDataset(self.data, None)
            # bucket_batch_sampler = BucketBatchSampler(bucket_dataset, self.config["batch_size"]) # <-- does not store X
            # return DataLoader(bucket_dataset,batch_sampler=bucket_batch_sampler, shuffle=False, num_workers=40, drop_last=False)#DataLoader(self.data, collate_fn=custom_collate,batch_size=self.config["batch_size"],num_workers=40)
        else:

            return DataLoader(self.data[:,:self.n_part], batch_size=self.batch_size, shuffle=False, num_workers=40, drop_last=False)

    def val_dataloader(self):
        return DataLoader(self.test_set[:,:self.n_part], batch_size=len(self.test_set), drop_last=True,num_workers=40)



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
    for i in x.val_dataloader():
        print(i.shape)