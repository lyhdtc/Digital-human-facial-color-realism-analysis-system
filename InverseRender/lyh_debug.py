import cv2
import os


'''使用的命令是这个'''
#python3 main.py --N 5 --checkpoint ../model/ --dataDir ../lyh_inData/DeepInverseRenderingnew/Face --logDir ../lyh_outData/Face --initDir ../example_data/example_init --network network_ae_fixBN --init_method rand --input_type image --wlv_type random --wlvDir ../example_data/example_wlv


'''自己需要用的一些小工具'''
# folder = '/home/lyh/Chapter4Experiment/4_3Exp/Res/Pic'
# file = os.listdir(folder)

# for f in file:
#     f = os.path.join(folder, f)
#     img = cv2.imread(f)
#     img = cv2.resize(img, (400,600))
#     cv2.imwrite(f, img)
# for i in range(15):
#     f = 'lyh_inData/Camera1/PBR_4.png'
#     img =cv2.imread(f)
#     f = 'PBR_'+str(i+5)+'.png'
#     f = os.path.join(folder,f)
#     cv2.imwrite(f, img)
    
# split textures
# 裁剪贴图，输出的时候是四个拼在一起的
pic = '/home/lyh/Chapter4Experiment/4_3Exp/Res/Pic/output_199.png'
savepath = 'lyh_outData/Face'
name=['normal', 'diffuse','roughness','specular']
for i in range(4):
    p = cv2.imread(pic)
    img = p[0:256,256*i:256*(i+1)]
    f = name[i]+'.png'
    f = os.path.join(savepath, f)
    cv2.imwrite(f, img)