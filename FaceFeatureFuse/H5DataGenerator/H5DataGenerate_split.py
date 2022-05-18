from cProfile import label
from email.mime import base
import os
import cv2
import pandas as pd
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import tqdm

# GroundTruth位置
ground_truth_path = '/home/lyh/Chapter4Experiment/res/Test_GroundTruth.csv'
# 数据集根目录，目录下应该为编号数字的文件夹，例如00000， 00001
base_path = '/home/lyh/Chapter4Experiment/res/H5pyGenFolder'
# h5py文件生成时的位置
h5_data_path = '/home/lyh/Chapter4Experiment/res'

folder_len = len(os.listdir(base_path))


for i in tqdm.trange(folder_len):

    

    f_csv = pd.read_csv(ground_truth_path,header=None)
    
    patch_label = f_csv.iloc[i]
    
    patch_data = np.zeros((386,400,400))
    # 单个图片组的目录，目录下应为o_p0, o_p1两个文件夹
    folder_path = os.path.join(base_path, str(i).zfill(5))
    o_p0_path   = os.path.join(folder_path, 'o_p0')
    o_p1_path   = os.path.join(folder_path, 'o_p1')
    
    o_p0_filelist = os.listdir(o_p0_path)
    o_p0_filelist.sort()
    o_p1_filelist = os.listdir(o_p1_path)
    o_p1_filelist.sort()
    
    for k, filename in enumerate(o_p0_filelist):
        cur_path = os.path.join(o_p0_path, filename)
        b,_,_ = cv2.split(cv2.imread(cur_path))
        # b = b.astype(int)
        # b = np.asarray(b)
        patch_data[k] = b
    for k, filename in enumerate(o_p1_filelist):
        cur_path = os.path.join(o_p1_path, filename)
        b,_,_ = cv2.split(cv2.imread(cur_path))
        # b = b.astype(int)
        patch_data[k+193] = b

    patch_data = patch_data.astype(np.uint8)
    
    f_h5_path = os.path.join(h5_data_path, str(i).zfill(5)+'.h5')
    f_h5 = h5py.File(f_h5_path, 'w') 
    f_h5['label'] = patch_label
    f_h5['data']  = patch_data
    f_h5.close()




