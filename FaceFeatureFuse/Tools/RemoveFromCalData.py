import os
import shutil
from tqdm import trange

# 根目录，下面应该是00000， 00001...的文件夹
base_path = '/home/lyh/002Experiment/NormTest'

# 把要移除的照片存放的位置
save_path = '/home/lyh/000Dataset/New_litCal/results'

# 要删除的照片的名字
# pic_list = ['Color_SpecularShadow.jpg', 'Color_Saturation.jpg', 'Color_Exposure.jpg', 'Color_Constract.jpg']
# pic_list = ['Color_CoherenceVector_a_0-31.jpg', 'Color_CoherenceVector_a_32-63.jpg', 
#             'Color_CoherenceVector_a_64-95.jpg', 'Color_CoherenceVector_a_96-127.jpg',
#             'Color_CoherenceVector_a_128-159.jpg', 'Color_CoherenceVector_a_160-191.jpg',
#             'Color_CoherenceVector_a_192-223.jpg', 'Color_CoherenceVector_a_224-255.jpg',
#             'Color_CoherenceVector_b_0-31.jpg', 'Color_CoherenceVector_b_32-63.jpg', 
#             'Color_CoherenceVector_b_64-95.jpg', 'Color_CoherenceVector_b_96-127.jpg',
#             'Color_CoherenceVector_b_128-159.jpg', 'Color_CoherenceVector_b_160-191.jpg',
#             'Color_CoherenceVector_b_192-223.jpg', 'Color_CoherenceVector_b_224-255.jpg',
#             'Color_CoherenceVector_l_0-31.jpg', 'Color_CoherenceVector_l_32-63.jpg', 
#             'Color_CoherenceVector_l_64-95.jpg', 'Color_CoherenceVector_l_96-127.jpg',
#             'Color_CoherenceVector_l_128-159.jpg', 'Color_CoherenceVector_l_160-191.jpg',
#             'Color_CoherenceVector_l_192-223.jpg', 'Color_CoherenceVector_l_224-255.jpg']
pic_list = ['Color_CoherenceVector_a.jpg', 'Color_CoherenceVector_b.jpg', 'Color_CoherenceVector_l.jpg']
label_list = ['o_p0', 'o_p1']
for i in trange(1000):
    for j in label_list:
        folder_path = os.path.join(base_path, str(i).zfill(5))
        folder_path = os.path.join(folder_path, j)
        target_path = os.path.join(save_path, str(i).zfill(5))
        target_path = os.path.join(target_path, j)
        os.makedirs(target_path, exist_ok=True)
        for k in pic_list:
            file_path = os.path.join(folder_path, k)
            target_file_path = os.path.join(target_path, k)
            if os.path.exists(file_path):
                shutil.move(file_path, target_file_path)
            # else:
                # print(f'{file_path} does not exist! please check out!')
        
        
        
    