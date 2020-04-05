
---
layout:     post
title:      [论文阅读]
subtitle:   对抗检测
date:       2020-04-03
author:     Mr.Huang
header-img: img/2020.04.03.jpg
catalog: true
tags:
    - 对抗检测
    - 机器学习
    - 论文阅读
    
---

# 前言

本期阅读论文《CNN-generated images are surprisingly easy to spot... for now》，并整理笔记如下：

论文名称：《CNN-generated images are surprisingly easy to spot... for now》

作者信息：Sheng-Yu Wang

[论文地址](https://arxiv.org/abs/1912.11035)

[源码地址](https://peterwang512.github.io/CNNDetection/)

# 文章大意

即使是在一种 CNN 生成的图像所训练的分类器，也能够跨数据集、网络架构和训练任务，展现出惊人的泛化能力.


在本文中，使用单个 CNN 模型（使用 ProGAN，一种高性能的非条件式 GAN 模型）生成大量伪造图像，并训练一个二分类器来检测伪造图像，将模型使用的真实训练图像作为负例。然后在创建的CNN生成图像的新数据集[ForenSynths]进行测试泛化性，甚至可以打败新出的StyleGAN2和DeepFake。。

	
## 优缺点

亮点：
	
	该工作意图探究深度生成取证泛化能力可能性。从训练数据的多样性、数据增强技术做探讨，说明了一定强先验条件下，泛化能力是可能的（AP衡量）。该文章做了偏基础的探究性实验工作。

缺点：

	1.这种泛化效果的先验假设是，测试的样本都是同一种伪造技术，这是不满足实际应用环境的。

        2.对衡量指标AP的阈值threshold，和准确率Acc.未做讨论，未说明是否会存在较大的数据集之间的bias偏差。
	
思考：

	1.训练用的是ProGAN，内容是场景，非人脸
	2.SAN 超分辨率很难检测，说明其有较强的篡改噪声能力，是否能加以利用？
	3.对Photoshop类似的算法无效。

# 文章结构

## Introduction

通用的检测器、用11种CNN生成技术的数据集、预处理和后处理、数据增强。

“针对生成技术的共有特征”为出发点，公布了以11种生成方法生成的的测试集“ForenSynths”，包含GAN生成、超分辨率、手工设计的Deepfakes。以ProGAN用作训练，采用特定数据增强的前/后处理方法，能带来泛化能力和鲁棒性。”

## Related work

分三部分：检测CNN生成的技术、图像取证、CNN生成的共有特征

## 检测CNN生成的图像

### 训练分类器

用ProGAN训练，探究泛化能力的上界；

采用ProGAN训练的原因：生成图像分辨率高、网络结构简单

ProGAN训练集用了20种LSUN类别：相当于20种生成的类型，airplane、cat...

每类别36K训练图片，20种一共720k训练图片

网络结构：预训练的ResNet-50+二分类

数据增强：随机翻转+224x224随机裁剪；同时Blur和JPEG压缩划分了几个等级。

衡量指标：AP，对不同数据集单独衡量，而非混合在一起（存在潜在数据集偏差问题，未做探究）

### 数据增强的作用

泛化性（训练数据增强，测试不处理）

![GdEhaq.jpg](https://s1.ax1x.com/2020/04/03/GdEhaq.jpg)

鲁棒性（训练和测试同时处理）

![GdEzi6.jpg](https://s1.ax1x.com/2020/04/03/GdEzi6.jpg)


# 结语

 > 本文首次发布于 [Mr.Huang Blog](http://www.huangsz.xyz), 作者 [@(Mr.Huang)](http://github.com/EmotionalXX) ,转载请保留原文链接.







