from cProfile import label
from email.mime import base
import os
import cv2
import pandas as pd
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import multiprocessing
from tqdm import tqdm


# 10个一组
# for i in range(int(100*test_data_percent)):
def test_Generate(i):
    left = int(i*10)
    right = int(i*10+10)
    
    # with open(ground_truth_path, 'r') as f:
    #     raws = csv.reader(f)
    #     for line in range(10):
    #         ans_label[line] = raws[line+left]
    f = pd.read_csv(ground_truth_path,header=None)
    ans_label=[]
    for j in range(left, right):
        ans_label.append(f.iloc[j])
    # ans_label = f.iloc[left:right,0].astype(np.int64)
    # print(ans_label[3])
    for j in range(10):
        cur_patch = np.zeros((386,400,400))
        # 单个图片组的目录，目录下应为o_p0, o_p1两个文件夹
        folder_path = os.path.join(base_path, str(j+left).zfill(5))
        print(folder_path)
        o_p0_path   = os.path.join(folder_path, 'o_p0')
        o_p1_path   = os.path.join(folder_path, 'o_p1')
        
        o_p0_filelist = os.listdir(o_p0_path)
        o_p0_filelist.sort()
        o_p1_filelist = os.listdir(o_p1_path)
        o_p1_filelist.sort()
        
        for k, filename in enumerate(o_p0_filelist):
            cur_path = os.path.join(o_p0_path, filename)
            b,_,_ = cv2.split(cv2.imread(cur_path))
            b = b.astype(int)
            # b = np.asarray(b)
            cur_patch[k] = b
        for k, filename in enumerate(o_p1_filelist):
            cur_path = os.path.join(o_p1_path, filename)
            b,_,_ = cv2.split(cv2.imread(cur_path))
            b = b.astype(int)
            cur_patch[k+193] = b
        # cur_patch = cur_patch.tolist()
        ans_data[j] = cur_patch    
    # lock.acquire()
    if i==0:    
        
        
        test_dataset[0:10] = ans_data
        
        
        test_labelset[0:10] = ans_label    
        # f_train.close()
    else:        
        test_dataset.resize((10*i+10, 386, 400, 400))
        test_dataset[10*i:10*i+10] = ans_data
        test_labelset.resize((10*i+10,1))    
        test_labelset[10*i:10*i+10] = ans_label
    # lock.release()   
# for i in range(int(100*(1-test_data_percent))):
def train_Generate(i):
    left = int(i*10+1000*test_data_percent)
    right = int(i*10+10+1000*test_data_percent)
    
    # with open(ground_truth_path, 'r') as f:
    #     raws = csv.reader(f)
    #     for line in range(10):
    #         ans_label[line] = raws[line+left]
    f = pd.read_csv(ground_truth_path,header=None)
    ans_label=[]
    for j in range(left, right):
        ans_label.append(f.iloc[j])
    # ans_label = f.iloc[left:right,0].astype(np.int64)
    # print(ans_label[3])
    for j in range(10):
        cur_patch = np.zeros((386,400,400))
        # 单个图片组的目录，目录下应为o_p0, o_p1两个文件夹
        folder_path = os.path.join(base_path, str(j+left).zfill(5))
        print(folder_path)
        o_p0_path   = os.path.join(folder_path, 'o_p0')
        o_p1_path   = os.path.join(folder_path, 'o_p1')
        
        o_p0_filelist = os.listdir(o_p0_path)
        o_p0_filelist.sort()
        o_p1_filelist = os.listdir(o_p1_path)
        o_p1_filelist.sort()
        
        for k, filename in enumerate(o_p0_filelist):
            cur_path = os.path.join(o_p0_path, filename)
            b,_,_ = cv2.split(cv2.imread(cur_path))
            b = b.astype(int)
            # b = np.asarray(b)
            cur_patch[k] = b
        for k, filename in enumerate(o_p1_filelist):
            cur_path = os.path.join(o_p1_path, filename)
            b,_,_ = cv2.split(cv2.imread(cur_path))
            b = b.astype(int)
            cur_patch[k+193] = b
        # cur_patch = cur_patch.tolist()
        ans_data[j] = cur_patch   
    # lock.acquire() 
    if i==0:    
        
       
        train_dataset[0:10] = ans_data
        
        
        train_labelset[0:10] = ans_label    
        # f_train.close()
    else:        
        train_dataset.resize((10*i+10, 386, 400, 400))
        train_dataset[10*i:10*i+10] = ans_data
        train_labelset.resize((10*i+10,1))    
        train_labelset[10*i:10*i+10] = ans_label  
    # lock.release()

# GroundTruth位置
ground_truth_path = '/home/lyh/000Dataset/DataSet_0325_lit/GroundTruth.csv'
# 数据集根目录，目录下应该为编号数字的文件夹，例如00000， 00001
base_path = '/home/lyh/000Dataset/New_litCal/results'
# 生成的TrainData位置
train_dataset_path = '/home/lyh/000Dataset/0403_train_400_400.h5'
# 生成的TestData位置
test_dataset_path  = '/home/lyh/000Dataset/0403_test_400_400.h5'


folder_len = len(os.listdir(base_path))

ans_data = np.zeros((10, 386, 400, 400))
# ans_label= np.zeros(10)
f_test = h5py.File(test_dataset_path, 'w')
f_train = h5py.File(train_dataset_path, 'w')
test_data_percent = 0.3

test_dataset = f_test.create_dataset('data', [10,386,400,400],maxshape=[None, 386,400,400], chunks=True, compression='gzip', compression_opts=7)
test_labelset = f_test.create_dataset('label', [10,1], maxshape=[None,1], chunks=True, compression='gzip', compression_opts=7)
train_dataset = f_train.create_dataset('data', [10,386,400,400],maxshape=[None, 386,400,400], chunks=True, compression='gzip', compression_opts=7)
train_labelset = f_train.create_dataset('label', [10,1], maxshape=[None,1], chunks=True, compression='gzip', compression_opts=7)

len_test = int(100*test_data_percent)
# manager = multiprocessing.Manager()
# lock = manager.Lock()
# pool = multiprocessing.Pool(4)

# list(tqdm(pool.imap(test_Generate, range(len_test))))
# pool.close()
# pool.join()
for i in range(len_test):
    test_Generate(i)

len_train = int(100*(1-test_data_percent))
for i in range(len_train):
    train_Generate(i)
# pool = multiprocessing.Pool(4)
# list(tqdm(pool.imap(train_Generate, range(len_train))))
# pool.close()
# pool.join()



    
    
# for i in range(100):
#     left = i*10
#     right = i*10+10
    
#     # with open(ground_truth_path, 'r') as f:
#     #     raws = csv.reader(f)
#     #     for line in range(10):
#     #         ans_label[line] = raws[line+left]
#     f = pd.read_csv(ground_truth_path,header=None)
#     ans_label=[]
#     for j in range(left, right):
#         ans_label.append(f.iloc[j])
#     # ans_label = f.iloc[left:right,0].astype(np.int64)
#     # print(ans_label[3])
#     for j in range(10):
#         cur_patch = np.zeros((386,400,400))
#         # 单个图片组的目录，目录下应为o_p0, o_p1两个文件夹
#         folder_path = os.path.join(base_path, str(j+left).zfill(5))
#         print(folder_path)
#         o_p0_path   = os.path.join(folder_path, 'o_p0')
#         o_p1_path   = os.path.join(folder_path, 'o_p1')
        
#         o_p0_filelist = os.listdir(o_p0_path)
#         o_p0_filelist.sort()
#         o_p1_filelist = os.listdir(o_p1_path)
#         o_p1_filelist.sort()
        
#         for k, filename in enumerate(o_p0_filelist):
#             cur_path = os.path.join(o_p0_path, filename)
#             b,_,_ = cv2.split(cv2.imread(cur_path))
#             b = b.astype(int)
#             # b = np.asarray(b)
#             cur_patch[k] = b
#         for k, filename in enumerate(o_p1_filelist):
#             cur_path = os.path.join(o_p1_path, filename)
#             b,_,_ = cv2.split(cv2.imread(cur_path))
#             b = b.astype(int)
#             cur_patch[k+193] = b
#         # cur_patch = cur_patch.tolist()
#         ans_data[j] = cur_patch

#     data_train, data_test, label_train, label_test = train_test_split(ans_data, ans_label, test_size=0.3)


#     if i==0:
        

#         train_dataset = f_train.create_dataset('data', [7,386,400,400],maxshape=[None, 386,400,400], chunks=True, compression='gzip', compression_opts=7)
        
#         train_dataset[0:7] = data_train
#         train_labelset = f_train.create_dataset('label', [7,1], maxshape=[None,1], chunks=True, compression='gzip', compression_opts=7)
        
#         train_labelset[0:7] = label_train        
#         # f_train.close()
        
        
#         test_dataset = f_test.create_dataset('data', [3,386,400,400],maxshape=[None, 386,400,400], chunks=True, compression='gzip', compression_opts=7)
#         test_dataset[0:3] = data_test
#         test_labelset = f_test.create_dataset('label', [3,1], maxshape=[None,1], chunks=True, compression='gzip', compression_opts=7)
        
#         test_labelset[0:3] = label_test    
#         # f_train.close()
#     else:
#         train_dataset.resize((7*i+7, 386,400,400))
#         train_dataset[7*i:7*i+7] = data_train
#         train_labelset.resize((7*i+7,1))
#         train_labelset[7*i:7*i+7]= label_train
        
#         test_dataset.resize((3*i+3, 386, 400, 400))
#         test_dataset[3*i:3*i+3] = data_test
#         test_labelset.resize((3*i+3,1))    
#         test_labelset[3*i:3*i+3] = label_test
    
f_train.close()
f_test.close()



