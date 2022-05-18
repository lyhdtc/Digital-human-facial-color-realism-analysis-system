'''
Descripttion: 
version: 
Author: LJ
Date: 2022-03-28 10:04:49
LastEditors: LJ
LastEditTime: 2022-03-28 14:48:02
'''
import os
class Logger:
    def __init__(self, path):
        self.path = path
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.fw = open(self.path+'/log','a')

    def write(self, text):
        print(text)
        self.fw.write(text)
        self.fw.flush()

    def close(self):
        self.fw.close()


class AverageValueMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



















        '''
# Descripttion: 
# version: 
# Author: LJ
# Date: 2022-03-28 09:38:40
# LastEditors: LJ
# LastEditTime: 2022-03-28 14:41:28
# '''
# import torch, h5py
# from torch.utils.data import Dataset
# import numpy as np

# train_idx = np.arange(700)
# test_idx = np.arange(700, 1000)

# class Dataset(Dataset):
#     def __init__(self, subset) -> None:
#         super(Dataset, self).__init__()
#         self.root = '/media/lj/My Passport/000_h5_split'
#         if subset == 'train':
#             self.idx = train_idx
#         else:
#             self.idx = test_idx
        

    
#     def __len__(self):
#         return len(self.idx)

#     def __getitem__(self, index):
#         n = '00000'
#         idx = str(self.idx[index])
#         n[-len(idx):] = idx
#         file_path = f'{self.root}/{n}.h5'
#         f = h5py.File(file_path, 'r')

#         data = np.array(f['data'])
#         label = np.array(f['label'])

#         f.close()

    
#         min = np.nanmin(data)
#         max = np.nanmax(data)
#         data = (data-min)/(max-min)
       

#         data = torch.tensor(data).float()
#         label = torch.tensor(label).long()
#         return data, label

