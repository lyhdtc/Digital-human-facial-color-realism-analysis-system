import cv2
import tqdm
import os

# 数据集根目录，目录下应该为编号数字的文件夹，例如00000， 00001
base_path = '/home/lyh/000Dataset/Lit_calDate/001DataTest'

# 生成的新数据集的目录
new_base_path = '/home/lyh/results/ResizeTest'
folder_len = len(os.listdir(base_path))


for i in tqdm.trange(folder_len):
    folder_path = os.path.join(base_path, str(i).zfill(5))
    new_folder_path = os.path.join(new_base_path, str(i).zfill(5))
    o_p0_path   = os.path.join(folder_path, 'o_p0')
    o_p1_path   = os.path.join(folder_path, 'o_p1')
    
    new_o_p0_path = os.path.join(new_folder_path, 'o_p0')
    os.makedirs(new_o_p0_path, exist_ok=True)
    new_o_p1_path = os.path.join(new_folder_path, 'o_p1')
    os.makedirs(new_o_p1_path, exist_ok=True)
    o_p0_filelist = os.listdir(o_p0_path)
    o_p0_filelist.sort()
    o_p1_filelist = os.listdir(o_p1_path)
    o_p1_filelist.sort()
    
    for k, filename in enumerate(o_p0_filelist):
        cur_path = os.path.join(o_p0_path, filename)
        new_path = os.path.join(new_o_p0_path, filename)
        img = cv2.resize(cv2.imread(cur_path), (200,200))
        cv2.imwrite(new_path, img)
        
        
    for k, filename in enumerate(o_p1_filelist):
        cur_path = os.path.join(o_p1_path, filename)
        new_path = os.path.join(new_o_p1_path, filename)
        img = cv2.resize(cv2.imread(cur_path), (200,200))
        cv2.imwrite(new_path, img)