---
layout:     post
title:      利用已训练好的InceptionV3模型进行测试自己的图像数据集
subtitle:   环境:Ubuntu18.04 TensorFlow1.12-CPU python3.7 
date:       2019-12-109
author:     Mr.Huang
header-img: img/post-bg-hacker.jpg
catalog: true
tags:
    - 深度学习
    - InceptionV3
    - tensorflow
    - python
---

# 前言
本节内容主要是利用虚拟机内的 **Ubuntu18.04** 和 **tensorflow1.12** 进行操作。

网络模型是inceptionV3，由于是穷汉学生，没有设备自己跑一遍，就用的别人已经训练好的。

# 正文


## InceptionV3基础理论

说到InceptionV3，就不得不提到GoogLeNet，也就是inceptionV1。2014年，GoogLeNet和VGG是当年ImageNet挑战赛(ILSVRC14)的冠亚军，GoogLeNet获得了第一名、VGG获得了第二名，这两类模型结构的共同特点是层次更深了。当时的GoogLeNet是22层，比当时的第二名VGG19还多3层（层数一般指的是卷积层和全连接层的层数），但是其参数量只有500万个，AlexNet参数个数是GoogleNet的12倍，VGG参数又是AlexNet的3倍，GoogleNet节约了内存空间，同时从模型结果来看，GoogLeNet的性能也 更加优越。

>GoogLeNet是谷歌（Google）研究出来的深度网络结构，为什么不叫“GoogleNet”，而叫“GoogLeNet”，据说是为了向“LeNet”致敬，因此取名为“GoogLeNet”

GoogLeNet网络模型的最重要的部分就是Inception模块，可以说，GoogLeNet就是由多个Inception模块堆叠而成。在GoogLeNet模型中，Inception模块是Inception V1。它的网络模型如下：





## InceptionV3实战
感谢@[超自然祈祷的整理](https://blog.csdn.net/sinat_27382047/article/details/80534234/)

### 获取训练好的模型文件

**模型和pbtxt文件：** <https://github.com/taey16/tf/tree/master/imagenet/>

[![Qd04VP.md.jpg](https://s2.ax1x.com/2019/12/09/Qd04VP.md.jpg)](https://imgse.com/i/Qd04VP)

其中

classify_image_graph_def.pb 文件就是训练好的Inception-v3模型; 

imagenet_synset_to_human_label_map.txt是类别文件

### 加载模型

先创建一个类NodeLookup来将softmax概率值映射到标签上,然后创建一个函数create_graph()来读取模型。详细代码如下：

```
# -*- coding: utf-8 -*-
 
import tensorflow as tf
import numpy as np
import re
import os
 
model_dir='D:/tf/model/'
image='d:/cat.jpg'
 
 
#将类别ID转换为人类易读的标签
class NodeLookup(object):
  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(
          model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
          model_dir, 'imagenet_synset_to_human_label_map.txt')
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)
 
  def load(self, label_lookup_path, uid_lookup_path):
    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)
 
    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string
 
    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]
 
    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name
 
    return node_id_to_name
 
  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]
 
#读取训练好的Inception-v3模型来创建graph
def create_graph():
  with tf.gfile.FastGFile(os.path.join(
      model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')
 
 
#读取图片
image_data = tf.gfile.FastGFile(image, 'rb').read()
 
#创建graph
create_graph()
 
sess=tf.Session()
#Inception-v3模型的最后一层softmax的输出
softmax_tensor= sess.graph.get_tensor_by_name('softmax:0')
#输入图像数据，得到softmax概率值（一个shape=(1,1008)的向量）
predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0': image_data})
#(1,1008)->(1008,)
predictions = np.squeeze(predictions)
 
# ID --> English string label.
node_lookup = NodeLookup()
#取出前5个概率最大的值（top-5)
top_5 = predictions.argsort()[-5:][::-1]
for node_id in top_5:
  human_string = node_lookup.id_to_string(node_id)
  score = predictions[node_id]
  print('%s (score = %.5f)' % (human_string, score))
 
sess.close()
```

>由于我是在虚拟机里跑的代码，我将之前下载好的文件夹放在share文件夹内， `注意`model_dir和image对应的位置应该修改，我的是

	model_dir = '/mnt/hgfs/share/tf/model/'
	image = '/mnt/hgfs/share/tf/test_image/cat.jpg'

### 测试数据

测试图片：

[![QYHIoQ.md.jpg](https://s2.ax1x.com/2019/12/07/QYHIoQ.md.jpg)](https://imgse.com/i/QYHIoQ)

测试结果如下：

![QdD3pF.png](https://s2.ax1x.com/2019/12/09/QdD3pF.png)


`测试文件夹内的多张图片`

1. 将image指定到文件夹位置，修改为

	image = '/mnt/hgfs/share/tf/test_image/'

	
2. 读取文件夹内的图像，依次读取一张，并测试分类效果。
```
pathDir = os.listdir(image_local)
for image in pathDir:
    image_data = tf.gfile.FastGFile(image_local+image, 'rb').read()

    sess = tf.Session()
    # Inception-v3模型的最后一层softmax的输出
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    # 输入图像数据，得到softmax概率值（一个shape=(1,1008)的向量）
    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    # (1,1008)->(1008,)
    predictions = np.squeeze(predictions)

    # ID --> English string label.
    node_lookup = NodeLookup()
    # 取出前5个概率最大的值（top-5)
    top_5 = predictions.argsort()[-1:][::-1]
    for node_id in top_5:
            human_string = node_lookup.id_to_string(node_id)
            score = predictions[node_id]
            print('%s (score = %.5f)' % (human_string, score))
    sess.close()
```


测试图片：

![QdDUTx.png](https://s2.ax1x.com/2019/12/09/QdDUTx.png)

测试结果如下：

[![QdD0fO.md.jpg](https://s2.ax1x.com/2019/12/09/QdD0fO.md.jpg)](https://imgse.com/i/QdD0fO)

>我修改成了TOP1的类别概率值

# 结语

关于**IncptionV3**的下载与测试就这样简单的介绍完了。

 
 > 本文首次发布于 [Mr.Huang Blog](http://www.huangsz.xyz), 作者 [@(Mr.Huang)](http://github.com/EmotionalXX) ,转载请保留原文链接.
