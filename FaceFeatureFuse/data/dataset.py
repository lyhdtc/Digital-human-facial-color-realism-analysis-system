'''
descripttion: 
version: 
Author: LJ
Date: 2022-03-28 09:38:40
LastEditors: LJ
LastEditTime: 2022-03-28 19:37:06
'''
from re import S
import torch, h5py, os, cv2
from torch.utils.data import Dataset
import numpy as np
# import pandas as pd

# train_idx = np.arange(700)
# test_idx = np.arange(700, 1000)

class DatasetAll(Dataset):
    def __init__(self, h5_data, h5_label) -> None:
        super(DatasetAll, self).__init__()
        self.data = h5_data
        self.label = h5_label
  
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = np.asarray(self.data[index])
        label = np.asarray(self.label[index])

        min = np.nanmin(data)
        max = np.nanmax(data)
        data = (data-min)/(max-min)

        data = torch.tensor(data).float()
        label = torch.tensor(label).long()
        return data, label

class Dataset(Dataset):
    def __init__(self, path, start, end) -> None:
        super(Dataset, self).__init__()
        self.root = path

        f = h5py.File(path, 'r')
        data = f['data'][start:end]
        self.data = np.asarray(data)

        label = f['label'][start:end]
        self.label = np.asarray(label)

        f.close()
  
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]

        min = np.nanmin(data)
        max = np.nanmax(data)
        data = (data-min)/(max-min)

        data = torch.tensor(data).float()
        label = torch.tensor(label).long()
        return data, label

# def normalization(data):
#     _range = np.max(data) - np.min(data)
#     return (data - np.min(data)) / _range
 

# class Dataset(Dataset):
#     def __init__(self, subset) -> None:
#         super(Dataset, self).__init__()
#         self.root = 'data/001DataTest'

#         label = pd.read_csv('data/GroundTruth.csv', header=None)
#         label = np.array(label)

#         dirs = os.listdir('data/001DataTest')
#         dirs.sort()

#         if subset == 'train':
#             self.dirs = dirs[:700]
#             self.label = label[:700]
#         else:
#             self.dirs = dirs[700: 1000]
#             self.label = label[700: 1000]

#         pics = os.listdir('data/001DataTest/00000/o_p0')
#         pics.sort()
#         self.pics = pics

#     def __len__(self):
#         return len(self.dirs)

#     def __getitem__(self, index):
#         _dir = self.dirs[index]
#         p0, p1 = [], []
#         for pic in self.pics:
#             op0  = os.path.join(self.root, _dir, 'o_p0', pic)
#             op1  = os.path.join(self.root, _dir, 'o_p1', pic)
#             op0, _, _ = cv2.split(cv2.imread(op0))
#             op1, _, _ = cv2.split(cv2.imread(op1))
#             p0.append([op0])
#             p1.append([op1])
        
#         p0 = np.concatenate(p0, 0)
#         p1 = np.concatenate(p1, 0)
#         data = np.concatenate([p0, p1], 0)
#         data = normalization(data)

#         data = torch.tensor(data).float()
#         label = torch.tensor(self.label[index]).long()
#         return data, label