import cv2
import os

# folder_name = ['albedo', 'normal', '']
label = ['o', 'p0', 'p1']
img_num = 29
folder = '/home/lyh/000Dataset/DataSet_0325_lit/shading'
folder_mask = '/home/lyh/000Dataset/DataSet_0325_lit/mask'
save_folder = '/home/lyh/002Experiment'


for i,l in enumerate(label):
    img_path = os.path.join(folder, l)
    img_path = os.path.join(img_path, str(img_num).zfill(5)+'.png')
    mask_path = os.path.join(folder_mask, l)
    mask_path = os.path.join(mask_path, str(img_num).zfill(5)+'.png')
    img = cv2.imread(img_path)
    mask= cv2.imread(mask_path)
    mask = mask/255
    img = img*mask
    save_path = os.path.join(save_folder, l)
    os.makedirs(save_path)
    save_path = os.path.join(save_path, str(0).zfill(5)+'.png')
    print(save_path)
    cv2.imwrite(save_path, img)