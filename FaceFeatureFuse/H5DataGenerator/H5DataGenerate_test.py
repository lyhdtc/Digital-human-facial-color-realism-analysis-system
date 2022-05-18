from cProfile import label
from email.mime import base
import os
import cv2
import csv
import h5py
import numpy as np
from sklearn.model_selection import train_test_split

# GroundTruth位置
ground_truth_path = '/home/lyh/results/002DataTest/GroundTruth.csv'
# 数据集根目录，目录下应该为编号数字的文件夹，例如00000， 00001
base_path = '/home/lyh/results/002DataTest/1'
# 生成的TrainData位置
train_dataset_path = '/home/lyh/results/002DataTest/Train_data.h5'
# 生成的TestData位置
test_dataset_path  = '/home/lyh/results/002DataTest/Test_data.h5'


folder_len = len(os.listdir(base_path))

ans_data = np.zeros((10, 428, 400, 400))
ans_label= []
with open(ground_truth_path, 'r') as f:
    for line in f:
        ans_label.append(int(line))

for i in range(folder_len):
    cur_patch = np.zeros((428,400,400))
    # 单个图片组的目录，目录下应为o_p0, o_p1两个文件夹
    folder_path = os.path.join(base_path, str(i).zfill(5))
    o_p0_path   = os.path.join(folder_path, 'o_p0')
    o_p1_path   = os.path.join(folder_path, 'o_p1')
    
    o_p0_filelist = os.listdir(o_p0_path)
    o_p0_filelist.sort()
    o_p1_filelist = os.listdir(o_p1_path)
    o_p1_filelist.sort()
    
    for j, filename in enumerate(o_p0_filelist):
        cur_path = os.path.join(o_p0_path, filename)
        b,_,_ = cv2.split(cv2.imread(cur_path))
        b = b.astype(int)
        # b = np.asarray(b)
        cur_patch[j] = b
    for j, filename in enumerate(o_p1_filelist):
        cur_path = os.path.join(o_p1_path, filename)
        b,_,_ = cv2.split(cv2.imread(cur_path))
        b = b.astype(int)
        cur_patch[j+214] = b
    # cur_patch = cur_patch.tolist()
    ans_data[i] = cur_patch
# ans_data = np.array(ans_data)
# ans_label = np.array(ans_label)
# print(np.shape(ans_data))
# print(np.shape(ans_label))
# ans_data = ans_data.tolis
data_train, data_test, label_train, label_test = train_test_split(ans_data, ans_label, test_size=0.3)



f_train = h5py.File(train_dataset_path, 'w')

dataset = f_train.create_dataset('data', [7,428,400,400],maxshape=[None, 428,400,400], chunks=True, compression='gzip', compression_opts=7)
dataset[0:7] = data_train
dataset.resize((14,428,400,400))
dataset[7:14]=data_train

# f_train['data'] = data_train
f_train['label']= label_train
f_train.close()

f_test = h5py.File(test_dataset_path, 'w')
f_test['data'] = data_test
f_test['label']= label_test
f_test.close()



# ans_data = np.zeros((10, 428, 400, 400))
# ans_label= []
# with open(ground_truth_path, 'r') as f:
#     for line in f:
#         ans_label.append(int(line))

# for i in range(folder_len):
#     cur_patch = np.zeros((428,400,400))
#     # 单个图片组的目录，目录下应为o_p0, o_p1两个文件夹
#     folder_path = os.path.join(base_path, str(i).zfill(5))
#     o_p0_path   = os.path.join(folder_path, 'o_p0')
#     o_p1_path   = os.path.join(folder_path, 'o_p1')
    
#     o_p0_filelist = os.listdir(o_p0_path)
#     o_p0_filelist.sort()
#     o_p1_filelist = os.listdir(o_p1_path)
#     o_p1_filelist.sort()
    
#     for j, filename in enumerate(o_p0_filelist):
#         cur_path = os.path.join(o_p0_path, filename)
#         b,_,_ = cv2.split(cv2.imread(cur_path))
#         b = b.astype(int)
#         # b = np.asarray(b)
#         cur_patch[j] = b
#     for j, filename in enumerate(o_p1_filelist):
#         cur_path = os.path.join(o_p1_path, filename)
#         b,_,_ = cv2.split(cv2.imread(cur_path))
#         b = b.astype(int)
#         cur_patch[j+214] = b
#     # cur_patch = cur_patch.tolist()
#     ans_data[i] = cur_patch
# # ans_data = np.array(ans_data)
# # ans_label = np.array(ans_label)
# # print(np.shape(ans_data))
# # print(np.shape(ans_label))
# # ans_data = ans_data.tolis
# data_train, data_test, label_train, label_test = train_test_split(ans_data, ans_label, test_size=0.3)



# f_train = h5py.File(train_dataset_path, 'w')
# dataset.resize((14,428,400,400))
# dataset[7:14]=f_train
# # f_train.update('data', data=data_train)
# f_train.create_dataset('label', data=label_train)

# f_train.close()

# f_test = h5py.File(test_dataset_path, 'w')

# f_test.create_dataset('data', data = data_test)
# f_test.create_dataset('label', data= label_test)

# f_test.close()