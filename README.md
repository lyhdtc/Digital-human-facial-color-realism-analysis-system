<!--
 * @Author: lyh
 * @Date: 2022-05-19 22:17:19
 * @LastEditors: lyh
 * @LastEditTime: 2022-05-19 22:17:20
 * @FilePath: /Digital-human-facial-color-realism-analysis-system/README.md
 * @Description: 
 * 
 * Copyright (c) 2022 by lyh, All Rights Reserved. 
-->
# Digital-human-facial-color-realism-analysis-system
# 数字人面部色彩真实度分析系统

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]

一种符合人类认知的数字人面部色彩真实度分析系统


## 主要流程
待补充
## 文件目录说明
共包含四个模块，每个模块的使用请参考各自的README.md文件
```
Modules
├── DatasetGenerator
├── InverseRender
├── FaceFeatureExtract
└── FaceFeatureFuse
```

## 上手指南
使用时参照流程依次选取对应模块运行即可   
```DatasetGenerator:```生成```FaceFeatureFuse```训练使用的数据集    
```InverseRender:```逆向渲染，将图片解耦成贴图   
```FaceFeatureExtract:```特征提取，使用传统图像处理方式提取特征灰度图像 
```FaceFeatureFuse:```特征融合，使用神经网络将灰度图像融合成伪彩色图像

<!-- links -->
[your-project-path]:lyhdtc/Digital-human-facial-color-realism-analysis-system
[contributors-shield]: https://img.shields.io/github/contributors/lyhdtc/Digital-human-facial-color-realism-analysis-system.svg?style=flat-square
[contributors-url]: https://github.com/lyhdtc/Digital-human-facial-color-realism-analysis-system/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/lyhdtc/Digital-human-facial-color-realism-analysis-system.svg?style=flat-square
[forks-url]: https://github.com/lyhdtc/Digital-human-facial-color-realism-analysis-system/network/members
[stars-shield]: https://img.shields.io/github/stars/lyhdtc/Digital-human-facial-color-realism-analysis-system.svg?style=flat-square
[stars-url]: https://github.com/lyhdtc/Digital-human-facial-color-realism-analysis-system/stargazers
[issues-shield]: https://img.shields.io/github/issues/lyhdtc/Digital-human-facial-color-realism-analysis-system.svg?style=flat-square
[issues-url]: https://img.shields.io/github/issues/lyhdtc/Digital-human-facial-color-realism-analysis-system.svg

