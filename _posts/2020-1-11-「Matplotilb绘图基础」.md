---
layout:     post
title:      [Matplotlib绘图基础]
subtitle:   Numpy学习
date:       2020-1-11
author:     Mr.Huang
header-img: img/2020.1.11.jpg
catalog: true
tags:
    - python
    - numpy
    - matplotlib
    
---
# 前言

观看视频覃秉丰《Python进阶-Matplotlib绘图基础》学习视频整理笔记

## 学习内容

	Matplotlib基础、figure图像、设置坐标轴、图例、标注、散点图、直方图、等高线图、3D绘图、subplot、动态图
		
### matplotlib基础

~~~
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-1,1,100) #生成范围为-1~1范围的均匀分布的100点
y = 2*x+1
plt.plot(x,y)
plt.show()
~~~

[![l4xmp8.md.jpg](https://s2.ax1x.com/2020/01/11/l4xmp8.md.jpg)](https://imgchr.com/i/l4xmp8)


### array创建

~~~
a = np.array([1,2,3],dtype = np.int32)
zero = np.zeros((2,3)) #生成(2,3)的全为0的矩阵 
empty = np.empty((2,3)) #生成(2,3)的接近0但不为0的矩阵，可用作除数
one = np.ones((2,3)) #生成(2,3)的全为1的矩阵
e = np.arrange(10) #生成包含0~9的数列，左闭右开
f = np.arrange(4,12) #生成4~11的数列
f1 = np.arrange(4,12,3) #生成4~11的数列，间隔为3
g = np.arrange(8).reshape(2,4) #将包含0~7的数组重新定义为(2,4)矩阵形式 
~~~

### matplotlib figure图像

~~~

import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-1,1,100)
y1 = 2*x +1
y2 = x ** 2

分别绘制两个图像，figure 1的大小为(8,5)，figure 2大小不变
plt.figgure()
plt.plot(x,y1)
plt.figure(figsize=(8,5)) #该图像设置为宽度为8，长度为5的figure
plt.plot(x,y2)
plt.show()
~~~

[![l4xjBj.md.jpg](https://s2.ax1x.com/2020/01/11/l4xjBj.md.jpg)](https://imgchr.com/i/l4xjBj)

~~~
#在一个figure中显示两张图片
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-1,1,100)
y1 = 2*x +1
y2 = x ** 2
plt.plot(x,y1,color='red',linewidth=1.0,linestyle='--') #颜色为红色，线宽为1，‘--’:表示虚线
plt.plot(x,y2,color='blue',linewidth=5.0,linestyle='-') #颜色为蓝色，线宽为5，‘-’:表示实线
plt.show()
~~~

[![l4zlvD.md.jpg](https://s2.ax1x.com/2020/01/11/l4zlvD.md.jpg)](https://imgchr.com/i/l4zlvD)

### matplotlib设置坐标轴

~~~
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-2,2,100)
y1 = 2*x +1
y2 = x ** 2

#设置x，y范围限制
plt.xlim((-2,2)) #限制x的范围为(-2,2)
plt.ylim((-2,3)) #限制y的范围为(-2,3)


~~~

### numpy的索引

~~~
for i in arr2:
    print(i) #按行读取，若for i in arr2.T 则是按列读取

for i in arr2.flat:
    print(i) #迭代输出每个元素
~~~

### array合并

~~~
import numpy as np
arr1 = np.array([1,2,3])
arr2 = np.array([4,5,6])

arr3 = np.vstack((arr1,arr2)) #垂直合并 得到[[1,2,3];[4,5,6]]

arr3 = np.hstack((arr1,arr2)) #水平合并 得到[1,2,3,4,5,6]

arr = np.concatenate((arr1,arr2),axis=0) #垂直合并 
#注意：被合并的矩阵维度需要保持一致  axis = 0:纵向合并；axis = 1:横向合并

print(arr1.T)  #一维的array不能转置，输出与转置前无差别
import numpy as np
arr1 = np.arange(3)
print(arr1) #原形状为(3,)
arr1_1 = arr1[np.newaxis,:] #增加维度,使其改变为(1,3)
arr1_2 = arr1[:,np.newaxis] #增加维度，使其改变为(3,1)
arr1_3 = np.atleast_2d(arr1) #增加维度，改变为2维,成为(1,3)
~~~


### array分割
~~~
import numpy as np
arr1 = np.arrange(12).reshape(3,4)

arr2,arr3 = np.split(arr1,2,axis=1) #水平方向分割，分成两份，需满足条件：列数%2=0

arr4,arr5,arr6 = np.array_split(arr1,3,axis=1) 
#水平方向完成三份分割
#注意'_split'的使用，此时忽略列数条件

arrv1,arrv2,arrv3 = np.vsplit(arr1,3) #垂直分割 等价于 axis = 0

arrh1,arrh2 = np.hsplit(arr1,2) #水平分割 等价于 axis = 1
~~~



### numpy的浅拷贝和深拷贝

~~~
import numpy as np
arr1 = np.array([1,2,3])
arr2 = arr1
arr2[0] = 5
print(arr1)  #arr1 => [5,2,3]
print(arr2)  #arr1 => [5,2,3]  arr1与arr2指向同一块内存  浅拷贝
arr3 = arr1.copy() #           arr1和arry3独立内存     深拷贝
~~~


# 结语

关于**python-numpy基础语法**的学习差不多在此结尾，后续有补充，将会继续更新。

 
 > 本文首次发布于 [Mr.Huang Blog](http://www.huangsz.xyz), 作者 [@(Mr.Huang)](http://github.com/EmotionalXX) ,转载请保留原文链接.







