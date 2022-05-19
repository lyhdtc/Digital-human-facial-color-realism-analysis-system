from errno import ELIBBAD
import cv2, sys, os, h5py, torch
sys.path.append('.')

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.model import model
from data.dataset import DatasetAll



save_dir = '/home/lyh/Chapter4Experiment/4_3Exp/NewCal_Data/O'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

network = model(193, [128,64,1]).cuda()
state = torch.load('log/sample5/model.pth')
network.load_state_dict(state)

network.eval()

# '''计算测试集'''
# test_path = '/home/lyh/000Dataset/0403_test_400_400.h5'
# test_h5 = h5py.File(test_path, 'r')
# test_data, test_label = test_h5['data'], test_h5['label']
# test_dataset = DatasetAll(test_data, test_label)
# test_loader = DataLoader(test_dataset, 1, False, num_workers=0)
# with torch.no_grad():
#     for i, (data, target) in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
        
#         print(i, target)
#         data = data.cuda()
#         target = target.cuda()
#         output, heatmap = network(data)
#         pred = output.data.max(1, keepdim=True)[1][0]
#         # acc = pred.eq(target.data.view_as(pred)).sum()
#         heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
#         # heatmap = heatmap/heatmap.max()
#         op0 = heatmap[0, :, :]
#         op0 = torch.relu(op0)
#         # op0 = (op0 - op0.min()) / (op0.max() - op0.min())

#         op0 = op0.cpu().data.numpy()
#         op0 = np.uint8(255 * op0)  # 将热力图转换为RGB格式
#         op0 = cv2.applyColorMap(op0, cv2.COLORMAP_JET) 

#         op1 = heatmap[1,:,:]
#         op1 = torch.relu(op1)
#         # op1 = (op1 - op1.min()) / (op1.max() - op1.min())
#         op1 = op1.cpu().data.numpy()
#         op1 = np.uint8(255 * op1)  # 将热力图转换为RGB格式
#         op1 = cv2.applyColorMap(op1, cv2.COLORMAP_JET) 

#         # if pred == 0:
#         #     op0_path = f'{save_dir}/op0_{i}*.png'
#         #     op1_path = f'{save_dir}/op1_{i}.png'

#         # else:
#         #     op0_path = f'{save_dir}/op0_{i}.png'
#         #     op1_path = f'{save_dir}/op1_{i}*.png'
            
            
#         if pred == 0:
#             # op0_path = f'{save_dir}/op0_{i}*.png'
#             op0_path = os.path.join(save_dir, 'o_p0')
#             if not os.path.exists(op0_path):
#                 os.makedirs(op0_path)
#             op0_path = os.path.join(op0_path, str(i).zfill(5)+'*.png')
#             # op1_path = f'{save_dir}/op1_{27}.png'
#             op1_path = os.path.join(save_dir, 'o_p1')
#             if not os.path.exists(op1_path):
#                 os.makedirs(op1_path)
#             op1_path = os.path.join(op1_path, str(i).zfill(5)+'.png')

#         else:
#             # op0_path = f'{save_dir}/op0_{27}.png'
#             op0_path = os.path.join(save_dir, 'o_p0')
#             if not os.path.exists(op0_path):
#                 os.makedirs(op0_path)
#             op0_path = os.path.join(op0_path, str(i).zfill(5)+'.png')
#             # op1_path = f'{save_dir}/op1_{27}*.png'
#             op1_path = os.path.join(save_dir, 'o_p1')
#             if not os.path.exists(op1_path):
#                 os.makedirs(op1_path)
#             op1_path = os.path.join(op1_path, str(i).zfill(5)+'*.png')


#         cv2.imwrite(op0_path, op0)
#         cv2.imwrite(op1_path, op1)

# test_h5.close()



'''
lyh 对整个数据集计算
'''

# h5_path_base = '/home/lyh/000Dataset/New_h5_split'
# for i in range(1000):
#     h5_path = os.path.join(h5_path_base, str(i).zfill(5)+'.h5')
#     test_h5 = h5py.File(h5_path, 'r')
#     data, target = np.asarray(test_h5['data'])[None, :, :,:], np.asarray(test_h5['label'])[None, :]
#     data = torch.tensor(data).float()
#     target = torch.tensor(target).long()
#     # save_dir = '.'
#     with torch.no_grad():
#         data = data.cuda()
#         target = target.cuda()
#         output, heatmap = network(data)
#         pred = output.data.max(1, keepdim=True)[1][0]
#         # acc = pred.eq(target.data.view_as(pred)).sum()
#         # heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
#         heatmap = heatmap / heatmap.max()
#         op0 = heatmap[0, :, :]
#         # op0 = (op0 - op0.min()) / (op0.max() - op0.min())

#         op0 = op0.cpu().data.numpy()
#         op0 = np.uint8(255 * op0)  # 将热力图转换为RGB格式
#         op0 = cv2.applyColorMap(op0, cv2.COLORMAP_JET) 

#         op1 = heatmap[1,:,:]
#         # op1 = (op1 - op1.min()) / (op1.max() - op1.min())
#         op1 = op1.cpu().data.numpy()
#         op1 = np.uint8(255 * op1)  # 将热力图转换为RGB格式
#         op1 = cv2.applyColorMap(op1, cv2.COLORMAP_JET) 

#         if pred == 0:
#             # op0_path = f'{save_dir}/op0_{i}*.png'
#             op0_path = os.path.join(save_dir, 'o_p0')
#             if not os.path.exists(op0_path):
#                 os.makedirs(op0_path)
#             op0_path = os.path.join(op0_path, str(i).zfill(5)+'*.png')
#             # op1_path = f'{save_dir}/op1_{27}.png'
#             op1_path = os.path.join(save_dir, 'o_p1')
#             if not os.path.exists(op1_path):
#                 os.makedirs(op1_path)
#             op1_path = os.path.join(op1_path, str(i).zfill(5)+'.png')

#         else:
#             # op0_path = f'{save_dir}/op0_{27}.png'
#             op0_path = os.path.join(save_dir, 'o_p0')
#             if not os.path.exists(op0_path):
#                 os.makedirs(op0_path)
#             op0_path = os.path.join(op0_path, str(i).zfill(5)+'.png')
#             # op1_path = f'{save_dir}/op1_{27}*.png'
#             op1_path = os.path.join(save_dir, 'o_p1')
#             if not os.path.exists(op1_path):
#                 os.makedirs(op1_path)
#             op1_path = os.path.join(op1_path, str(i).zfill(5)+'*.png')

#         cv2.imwrite(op0_path, op0)
#         cv2.imwrite(op1_path, op1)
#         print(op0_path)
#         print(op1_path)

'''计算单组数据'''
h5_path = '/home/lyh/Chapter4Experiment/4_3Exp/NewCal_Data/O/00001.h5'
test_h5 = h5py.File(h5_path, 'r')
data, target = np.asarray(test_h5['data'])[None, :, :,:], np.asarray(test_h5['label'])[None, :]
data = torch.tensor(data).float()
target = torch.tensor(target).long()
# save_dir = '.'
with torch.no_grad():
    data = data.cuda()
    target = target.cuda()
    output, heatmap = network(data)
    pred = output.data.max(1, keepdim=True)[1][0]
    # acc = pred.eq(target.data.view_as(pred)).sum()
    # print(type(heatmap))
    print(heatmap.max())
    print(heatmap.min())
    mx = torch.tensor(20000).float()
    heatmap = (heatmap - heatmap.min()) / (mx)
    
    # heatmap = heatmap / heatmap.norm2d()
    op0 = heatmap[0, :, :]
    # op0 = (op0 - op0.min()) / (op0.max() - op0.min())

    op0 = op0.cpu().data.numpy()
    op0 = np.uint8(255 * op0)  # 将热力图转换为RGB格式
    # op0 = cv2.applyColorMap(op0, cv2.COLORMAP_JET) 

    op1 = heatmap[1,:,:]
    # op1 = (op1 - op1.min()) / (op1.max() - op1.min())
    op1 = op1.cpu().data.numpy()
    op1 = np.uint8(255 * op1)  # 将热力图转换为RGB格式
    # op1 = cv2.applyColorMap(op1, cv2.COLORMAP_JET) 

    num = '00001g'
    if pred == 0:
        op0_path = f'{save_dir}/op0_{num}*.png'
        op1_path = f'{save_dir}/op1_{num}.png'

    else:
        op0_path = f'{save_dir}/op0_{num}.png'
        op1_path = f'{save_dir}/op1_{num}*.png'

    cv2.imwrite(op0_path, op0)
    cv2.imwrite(op1_path, op1)