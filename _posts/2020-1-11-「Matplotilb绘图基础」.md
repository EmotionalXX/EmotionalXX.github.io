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

#对x,y描述
plt.xlabel('i am x')
plt.ylabel('i am y')

#x,y的曲线形式
plt.plot(x,y1,color='red',linewidth=1.0,linestyle='--') #‘--’表示虚线
plt.plot(x,y2,color='blue',linewidth=5.0,linestyle='-') #'-'表示实线

#修改x的间距
new_ticks = np.linespace(-2,2,11)

#更换x间隔和y轴别名
plt.xticks(new_ticks)
plt.yticks([-1,0,1,2,3],['level1','level2','level3','level4','level5'])

#设置figure的边框
ax = plt.gca() #  gca ：获取坐标轴
ax.spines['right'].set_color('none') #右边的边框设置为红色
ax.spines['top'].set_color('none') #上面的边框设置为透明色

ax.xaxis.set_ticks_position('bottom') #把x轴的刻度设置为‘bottom’
ax.yaxis.set_ticks_position('left')   #把y轴的刻度设置为‘left’
ax.spines['bottom'].set_position(('data',0)) #设置bottom对应的零点
ax.spines['left'].set_position(('data',0)) #设置left对应的零点

plt.show()

~~~
![lIoWvR.jpg](https://s2.ax1x.com/2020/01/12/lIoWvR.jpg)

### matplotlib图例

~~~
#接上文
from matplotlib.legend_handler import HandlerLine2D
l1, = plt.plot(x,y1,color = 'red',linewidth=1.0,linestyle='--',label='test1')
l2, = plt.plot(x,y2,color='blue',linewidth=1.0,linestyle='-',label='test2') 
plt.legend(handler_map={l1: HandlerLine2D(numpoints=1)})

~~~

![lITPPg.jpg](https://s2.ax1x.com/2020/01/12/lITPPg.jpg)

### matplotlib标注

~~~
import matplotlib.pyplot as plt
#from matplotlib.legend_handler import HandlerLine2D
import numpy as np
x = np.linspace(-1,1,100)
y1 = 2*x +1
plt.plot(x,y1,color='red',linewidth=1.0,linestyle='--')
ax = plt.gca() #gca ：获取坐标轴
ax.spines['right'].set_color('none') #右边的边框设置为透明色
ax.spines['top'].set_color('none') #上面的边框设置为透明色
ax.xaxis.set_ticks_position('bottom') #把x轴的刻度设置为‘bottom’
ax.yaxis.set_ticks_position('left')   #把y轴的刻度设置为‘left’
ax.spines['bottom'].set_position(('data',0)) #设置bottom对应的零点
ax.spines['left'].set_position(('data',0)) #设置left对应的零点

#定义需要标注的点
x0 = 0.5
y0 = 2*x0 + 1

#瞄点与垂直虚线
plt.scatter(x0,y0,s=50,edgecolors='b') #'b'=blue
plt.plot([x0,x0],[y0,0],'k--',lw=2) #(x0,y0),(x0,0)，‘k'表示black，黑色，lw：线宽

#箭头与描述
plt.annotate(r'$2x+1=%s$' % y0,xy=(x0,y0),xytext=(+30,-30),textcoords='offset points',fontsize=16 \
,arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2'))
#xytext ：文字描述的位置，以蓝点为起点，x方向+30，y方向-30，形成距离感
#connectstyle=arc3,rad=.2  作带弧度的箭头，其中0.2表示弧度，不加该语句，则为直线箭头

#作红色文字描述
plt.text(-1,2,r'$this\ is\ the\ text$',fontdict={'size':16,'color':'r'})  #不能直接使用空格
#字体大小为16，颜色为红色
plt.show()
~~~

![lITuIU.jpg](https://s2.ax1x.com/2020/01/12/lITuIU.jpg)


### matplotlib作散点图
~~~
import matplotlib.pyplot as plt
import numpy as np

# plt.scatter(np.arange(5),np.arange(5))  #scatter作散点图(0,0),(1,1)...
# plt.show()
x = np.random.normal(0,1,500)
y = np.random.normal(0,1,500)
plt.scatter(x,y,s=50,c='b',alpha=0.5) #’s'表示点的大小，'c'表示颜色,’alpha'指透明度
plt.xlim((-2,2))
plt.ylim((-2,2))
plt.xticks(())    #取消边框，无坐标值
plt.yticks(())    #取消边框，无坐标值
plt.show()
~~~

![lITGs1.jpg](https://s2.ax1x.com/2020/01/12/lITGs1.jpg)


### matplotlib直方图

~~~
import matplotlib.pyplot as plt
import numpy as np
x = np.arange(10)
y = 2**x +10
plt.bar(x,y,facecolor = '#9999ff',edgecolor='white') #设置颜色和边框

for x,y in zip(x,y):  #zip结合x,y,形成整体
    plt.text(x,y,'%.2f' %y,ha='center',va='bottom') #ha 和 va表示显示的位置
plt.show()
~~~

![lITWFS.jpg](https://s2.ax1x.com/2020/01/12/lITWFS.jpg)


### matplotlib作等高线图

~~~
import matplotlib.pyplot as plt
import numpy as np

def f(x,y):
    return (1-x/2+x**5+y**3)*np.exp(-x**2-y**2)
x = np.linspace(-3,3,100)
y = np.linspace(-3,3,100)

X,Y = np.meshgrid(x,y)  #将x,y坐标传入网格中，成为X.Y
plt.contourf(X,Y,f(X,Y),8,alpha=0.75,cmap=plt.cm.hot) #8指图中的8+根线，绘制等温线，其中cmap指颜色
plt.xticks()
plt.yticks()
C = plt.contour(X,Y,f(X,Y),8,colors='black',linewidth=.5)
plt.clabel(C,inline=True,fontsize = 10)
plt.show()
~~~

![lITbwV.jpg](https://s2.ax1x.com/2020/01/12/lITbwV.jpg)

### matplotlib 3D绘图

~~~
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)
x = np.arange(-4,4,0.25)
y = np.arange(-4,4,0.25) #-4~4，间隔为0.25
X,Y = np.meshgrid(x,y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)
ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap=plt.get_cmap('rainbow'))
#restride、cstride表示色块的大小
# cmap 表示颜色，此处用到的是彩虹颜色


#映射
ax.contourf(X,Y,Z,zdir = 'z',offset=-2,cmap='rainbow')
ax.set_zlim(-2,2)
plt.show()
~~~

![lITjW4.jpg](https://s2.ax1x.com/2020/01/12/lITjW4.jpg)

### matplotlib.subplot

~~~
import matplotlib.pyplot as plt
import numpy as np
plt.figure()
plt.subplot(2,1,1)  #在figure中创建2行2列的绘图，该图像位于第一个位置
plt.plot([0,1],[0,1])
plt.subplot(2,3,4) #也可写成222
plt.plot([0,1],[0,1])
plt.subplot(2,3,5)
plt.plot([0,1],[0,1])
plt.subplot(2,3,6)
plt.plot([0,1],[0,1])
plt.show()
~~~

![lI7pO1.jpg](https://s2.ax1x.com/2020/01/12/lI7pO1.jpg)

### matplotlib动态图

~~~
from matplotlib import animation#动态图所需要的包
import numpy as np
import matplotlib.pyplot as plt

fig,ax = plt.subplots() #子图像
x = np.arange(0,2*np.pi,0.01)
line, = ax.plot(x,np.sin(x))

def animate(i):
    line.set_ydata(np.sin(x+i/10))#用来改变的y对应的值
    return line,
def init():
    line.set_ydata(np.sin(x))#动态图初始图像
    return line,

ani = animation.FuncAnimation(fig=fig,func=animate,init_func=init,interval=20)#动态作图的方法，func动态图函数，init_func初始化函数，interval指图像改变的时间间隔
plt.show()
~~~

![lI7ZSH.jpg](https://s2.ax1x.com/2020/01/12/lI7ZSH.jpg)


# 结语

关于**python-matplotlib**的学习差不多在此结尾，后续有补充，将会继续更新。

 
 > 本文首次发布于 [Mr.Huang Blog](http://www.huangsz.xyz), 作者 [@(Mr.Huang)](http://github.com/EmotionalXX) ,转载请保留原文链接.







