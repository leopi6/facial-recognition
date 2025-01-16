
import copy
import os
import random
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from data_process_astringent import load_data_0
from data_process_sweet import load_data_1
#from sweet1 import load_data_1
from data_process_sour import load_data_2
from data_process_bitter import load_data_5
from data_process_umami import load_data_4
from data_process_salt import load_data_3

import torch.nn.functional as F
import sklearn.metrics as metrics
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix







# 首先定义一个 分类数*分类数 的空混淆矩阵
conf_matrix = torch.zeros(3, 3)
# 使用torch.no_grad()可以显著降低测试用例的GPU占用
"""with torch.no_grad():
    for step, (imgs, targets) in enumerate(Dte):
        # imgs:     torch.Size([50, 3, 200, 200])   torch.FloatTensor
        # targets:  torch.Size([50, 1]),     torch.LongTensor  多了一维，所以我们要把其去掉
        # targets = targets.squeeze()  # [50,1] ----->  [50]

        # 将变量转为gpu
        targets = targets.cuda()
        imgs = imgs.cuda()
        # print(step,imgs.shape,imgs.type(),targets.shape,targets.type())

        out = model(imgs)
        # 记录混淆矩阵参数
        conf_matrix = confusion_matrix(out, targets, conf_matrix)
        conf_matrix = conf_matrix.cpu()
conf_matrix = np.array(conf_matrix.cpu())  # 将混淆矩阵从gpu转到cpu再转到np
corrects = conf_matrix.diagonal(offset=0)  # 抽取对角线的每种分类的识别正确个数
per_kinds = conf_matrix.sum(axis=0)  # 抽取每个分类数据总的测试条数

# print("混淆矩阵总元素个数：{0},测试集总个数:{1}".format(int(np.sum(conf_matrix)), test_num))
print(conf_matrix)
"""
# 获取每种Emotion的识别准确率
conf_matrix = np.array([[2366, 2890, 3723],
                        [654, 859, 884],
                        [0, 0, 0]])

# 绘制混淆矩阵
Emotion = 3  # 这个数值是具体的分类数，大家可以自行修改
labels = ['0', '1', '2']  # 每种类别的标签

# 显示数据
plt.imshow(conf_matrix, cmap=plt.cm.Blues)

# 在图中标注数量/概率信息
thresh = 3723 / 2  # 数值颜色阈值，如果数值超过这个，就颜色加深。
for x in range(3):
    for y in range(3):
        # 注意这里的matrix[y, x]不是matrix[x, y]
        info = int(conf_matrix[y, x])
        plt.text(x, y, info,
                 verticalalignment='center',
                 horizontalalignment='center',
                 color="white" if info > thresh else "black")

plt.tight_layout()  # 保证图不重叠
plt.yticks(range(3), labels)
plt.xticks(range(3), labels, rotation=45)  # X轴字体倾斜45°
plt.show()
plt.close()