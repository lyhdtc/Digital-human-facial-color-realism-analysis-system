import os
import h5py
import numpy as np


# 生成的TrainData位置
train_dataset_path = '/home/lyh/002Experiment/00000.h5'
# 生成的TestData位置
test_dataset_path  = '/home/lyh/results/002DataTest/Test_data.h5'


f_train = h5py.File(train_dataset_path, 'r')
data_train = np.array(f_train['data'])
label_train= np.array(f_train['label'])
f_train.close()

# f_test = h5py.File(test_dataset_path, 'r')
# data_test = np.array(f_test['data'])
# label_test= np.array(f_test['label'] )
# f_test.close()

print('1')