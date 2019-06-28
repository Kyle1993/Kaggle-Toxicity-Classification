import pandas as pd
import numpy as np
import sys,os
import pickle
from sklearn.model_selection import KFold
from torch.utils.data import Dataset,DataLoader
from keras.preprocessing import sequence
import gc
import random
import torch

from global_variable import *

def generate_kfold():
    ids = list(range(train_num))
    kfolder = KFold(n_splits=cv,shuffle=True,random_state=1993)

    kfold = []
    for train_idx,validate_idx in kfolder.split(ids):
        # print(type(train_idx),train_idx.shape,validate_idx.shape)
        kfold.append([train_idx,validate_idx])

    with open('kfold_{}.pkl'.format(cv),'wb') as f:
        pickle.dump(kfold,f)

class ToxicityDataset(Dataset):
    def __init__(self,x,y,weight,idx):
        super(ToxicityDataset,self).__init__()

        x = np.asarray(x)
        self.x = x[idx]
        self.y = y[idx]
        self.idx = idx
        self.weight = weight[idx]

    def __getitem__(self, id):
        return self.x[id], self.y[id], self.weight[id], self.idx[id]

    def __len__(self):
        return len(self.idx)

class ToxicityTestDataset(Dataset):
    def __init__(self,x,ids):
        super(ToxicityTestDataset,self).__init__()

        self.x = x
        self.ids = ids

        assert len(x) == len(ids)

    def __getitem__(self, id):
        return self.x[id],self.ids[id]

    def __len__(self):
        return len(self.ids)


class SequenceBucketCollator():
    def __init__(self,percentile=100):
        self.percentile = percentile

    def __call__(self, batch):
        batch_ = list(zip(*batch))
        data_num = len(batch[0])
        assert data_num >= 1
        # data = [item[0] for item in batch]
        # target = [item[1] for item in batch]
        # weight = [item[2] for item in batch]
        # idx = [item[3] for item in batch]

        lens = [len(x) for x in batch_[0]]
        max_len = np.percentile(lens, self.percentile)

        batch_[0] = sequence.pad_sequences(batch_[0], maxlen=int(max_len))
        batch_[0] = np.asarray(batch_[0],dtype=np.int)

        for i in range(1,data_num):
            batch_[i] = np.asarray(batch_[i])

        return batch_

if __name__ == '__main__':
    x = np.asarray([[1,2,3],[2,3],[2,3,4,5,6,7],[1]])
    w = np.random.random(size=(4,))
    y = np.random.random(size=(4,))
    id = np.asarray([0,1,2,])

    ds = ToxicityTestDataset(x,['asd','qwe','s','r'])
    dl = DataLoader(ds,2,collate_fn=SequenceBucketCollator())

    for d in dl:
        print('-------------')
        print(d[0])
        print(d[1])
        # print(d[2])
        # print(d[3])