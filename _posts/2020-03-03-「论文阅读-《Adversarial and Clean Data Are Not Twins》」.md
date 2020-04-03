
---
layout:     post
title:      [AI工程师]
subtitle:   对抗检测
date:       2020-03-03
author:     Mr.Huang
header-img: img/2020.03.03.jpg
catalog: true
tags:
    - 对抗检测
    - 机器学习
    - 论文阅读
    
---
# 前言

本期阅读论文《Adversarial and Clean Data Are Not Twins》，并整理笔记如下：

论文名称：《Adversarial and Clean Data Are Not Twins》

作者信息：Zhitao Gong

摘要：Zhitao Gong等在论文中将基于DNN的二进制分类器训练为检测器，对输入样本进行对抗样本识别，
同时证明该检测器在第二轮对抗攻击中具有鲁棒性。实验表明，对于测试的二分类检测器，
在由MNIST、CIFAR10和SVHN数据集上通过FGSM/TGSM产生的对抗样本识别准确率达到超过99%，
同时对第二轮对抗攻击是具有鲁棒性。

源码地址：https://github.com/gongzhitaao/adversarial-classifier （Tensrflow1.4+Python2.7）

# 学习内容

	算法思路与结论分析
	
## 算法思路

![3hujgO.jpg](https://s2.ax1x.com/2020/03/03/3hujgO.jpg)

## 算法结论

![3huM9K.jpg](https://s2.ax1x.com/2020/03/03/3huM9K.jpg)

## 算法迷惑之处

![3hud9f.jpg](https://s2.ax1x.com/2020/03/03/3hud9f.jpg)

我认为对X_test进行adv{f2}后，其正确识别结果应该是对抗样本，但是作者给出的解释是high false negative,
我的理解是检测器f2错误识别为干净样本，检测器f2放行,故与原文中的‘it tends to recognize them all
as adversarials’相悖。

## 解惑

感谢作者的悉心回答，我对迷惑之处有了大致的想法：

我认为文章主要疑惑的数据为{X_test^adv(f2)} 和 {X_test^adv(f1)}^adv(f2).前者的实验测得数据为→0，而后者测得数据为→1。
我们以手写识别数字分类器为例，其f2 label为0（干净样本），f1 label为‘1’（手写体为数字1），此时对f2进行对抗攻击，使其f2 label→**1**，但是我们识别的是针对f1的对抗样本，所以→0，说明是识别为干净样本的准确率，即大部分是识别为对抗样本，这对二分器的性能来说并没有影响。此外第二个数据指标说明，二分类器具有第二轮对抗攻击的鲁棒性，即无法制作对抗样本使其fool f1 and dypass f2.

再次感谢作者 Zhitao Gong

# 结语

 > 本文首次发布于 [Mr.Huang Blog](http://www.huangsz.xyz), 作者 [@(Mr.Huang)](http://github.com/EmotionalXX) ,转载请保留原文链接.







