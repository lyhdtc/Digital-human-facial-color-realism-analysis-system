<!--
 * @Author: lyh
 * @Date: 2022-05-19 16:11:42
 * @LastEditors: lyh
 * @LastEditTime: 2022-05-20 17:16:00
 * @FilePath: /Digital-human-facial-color-realism-analysis-system/DatasetGenerator/README.md
 * @Description: 
 * 
 * Copyright (c) 2022 by lyh, All Rights Reserved. 
-->
# Dataset Generator
# 数据集生成

本模块修改自 [GIF: Generative Interpretable Faces](https://github.com/ParthaEth/GIF)，通过控制渲染过程中法线、albedo、光照实现数据集生成

## 上手指南
* [下载预训练模型](https://pan.baidu.com/s/13dkSCxBxBIWgwJBCgNvvSw?pwd=0000)(22G)，并将整个文件夹放至根目录
* 使用```lyh_dataset_generate.py```生成数据集，注意修改28-40行的参数

## 数据集下载（png格式）
* [修改法线](https://pan.baidu.com/s/1ZbwazWAxsLGyhlHDvJbAZw?pwd=0000)(811.1M)
* [修改albedo](https://pan.baidu.com/s/1t6A63dobXA7zCGdjBtLlow?pwd=0000)(801.4M)
* [修改光照](https://pan.baidu.com/s/1swi_8oxTODt13WZ0XoPc4w?pwd=0000)(825.5M)

## 测试实验工具
为了验证生成的数据集与人类认知相似，制作了验证实验程序，代码见```experiment.py```，[exe格式程序下载](https://pan.baidu.com/s/1yiUv2c64SiZYJ2M1KXuPvw?pwd=0000)，使用时替换```Test_Pics```文件夹中的图像即可，程序界面如下   
![images](/ReadMePics/2_ExperimentUI.png)