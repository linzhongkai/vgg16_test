# -*- coding:utf-8 -*-
"""
基本流程是：
1：载入相关模块
2：下载训练好的模型文件
3：导入测试图像
4：应用模型文件进行分类
5：显示
"""


#导入支持包
import matplotlib.pyplot as plt

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions

import numpy as np

def percent(value):
    return '%.2f%%' % (value * 100)

model = VGG16(weights='imagenet', include_top=True)
#下载模型权重文件，True表示会载入完整的VGG16模型，包括加在最后的三层卷积层
# false表示会载入模型，但是不会包括最后的三层卷积层

img_path='test.jpp'

img = image.load_img(img_path, target_size=(224, 224))
#转换图像尺寸

x = image.img_to_array(img)
#转换为浮点型
x = np.expand_dims()
#转化为张量(1,224,224,3)
x = preprocess_input(x)

features = model.predict(x)
#预测，取得features，维度为(1, 1000)

pred = decode_predictions(features, top=5)[0]
#取得前5个最可能的类别和概率

#整理预测结果
values = []
bar_label = []
for element in pred:
    values.append(element[2])
    bar_label.append(element[1])

#绘图并且保存
fig = plt.figure(u"Top-5 预测结果")
ax = fig.add_subplot(111)
ax.bar(range(len(values)), values, tick_label=bar_label, width=0.5, fc='g')
ax.set_ylabel(u'probability')
ax.set_title(u'Top-5')
for a, b in zip(range(len(values)), values):
    ax.text(a, b+0.0005, percent(b), ha='center', va='bottom', fontsize=7)

fig = plt.gcf()
plt.show()

name = img_path[0:-4] + '_pred'
fig.savefig(name, dpi=200)




