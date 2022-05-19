<!--
 * @Author: lyh
 * @Date: 2022-05-19 16:11:42
 * @LastEditors: lyh
 * @LastEditTime: 2022-05-19 16:28:51
 * @FilePath: /Digital-human-facial-color-realism-analysis-system/DatasetGenerator/README.md
 * @Description: 
 * 
 * Copyright (c) 2022 by lyh, All Rights Reserved. 
-->
# Dataset Generator
# 数据集生成

本模块修改自 [GIF: Generative Interpretable Faces](https://github.com/ParthaEth/GIF)，通过控制渲染过程中法线、albedo、光照实现数据集生成

## 上手指南
* 下载预训练模型，并将整个文件夹放至根目录
* 使用```lyh_dataset_generate.py```生成数据集，注意修改28-40行的参数

## 数据集下载（png格式）
* 修改法线
* 修改albedo
* 修改光照