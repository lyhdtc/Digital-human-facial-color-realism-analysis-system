<!--
 * @Author: lyh
 * @Date: 2022-05-19 16:02:31
 * @LastEditors: lyh
 * @LastEditTime: 2022-05-19 16:10:53
 * @FilePath: /Digital-human-facial-color-realism-analysis-system/FaceFeatureExtract/README.md
 * @Description: 
 * 
 * Copyright (c) 2022 by lyh, All Rights Reserved. 
-->
# Face Feature Extract
# 面部特征提取

将```InverseRender```解耦得到的贴图组进行特征提取，获得193张特征图像

## 上手指南
使用```main.py```即可，注意修改代码中的路径
* 每次计算两张图像间的特征，使用```general_run()```
* 计算数据集所有图像的特征，使用```dataset_run()```   

## 提取的特征种类
详见```Color/ColorAlgorithrm.py```与```Texture/TextureAlgorithrm.py```

## 结果样例
待补充