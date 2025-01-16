

def get_predictions_and_labels(model, Dte):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    predictions_probabilities_list = []  # 保存模型预测的概率
    true_labels_list = []  # 保存真实标签

    with torch.no_grad():
        for imgs, labels in Dte:
            imgs = imgs.to(device)
            labels = labels.to(device)

            output = model(imgs)
            predictions_probabilities = torch.softmax(output, dim=1)

            predictions_probabilities_list.append(predictions_probabilities.cpu().numpy())
            true_labels_list.append(labels.cpu().numpy())

    # 将 predictions_probabilities_list 和 true_labels_list 转换为 NumPy 数组
    predictions_probabilities_all = np.concatenate(predictions_probabilities_list)
    true_labels_all = np.concatenate(true_labels_list)

    return predictions_probabilities_all, true_labels_all



def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix
"""
@Time ： 2024/03/26 11:34
@Author ：李颢
@File ：CNN.py
@Motto：GO UP 

"""
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import roc_curve, auc


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


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(20)


class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=2,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        #
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        #
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc1 = nn.Linear(3 * 3 * 64, 64)
        self.fc2 = nn.Linear(64, 18)
        self.out = nn.Linear(18, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # print(x.size())
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # x = self.sigmoid(self.out(x))
        x = F.log_softmax(x, dim=1)
        return x




def get_val_loss(model, Val):
    model.eval()
    criterion = nn.CrossEntropyLoss().to(device)
    val_loss = []
    for (data, target) in Val:
        data, target = Variable(data).to(device), Variable(target.long()).to(device)
        output = model(data)
        loss = criterion(output, target)
        val_loss.append(loss.cpu().item())

    return np.mean(val_loss)







def train(input,b):
    if input == '0':
        Dtr, Val, Dte = load_data_0(b)
    if input == '1':
        Dtr, Val, Dte = load_data_1(b)
    if input == '2':
        Dtr, Val, Dte = load_data_2(b)
    if input == '3':
        Dtr, Val, Dte = load_data_3(b)
    if input == '4':
        Dtr, Val, Dte = load_data_4(b)
    if input == '5':
        Dtr, Val, Dte = load_data_5(b)




    print('training...')
    epoch_num = 5
    best_model = None
    min_epochs = 1
    min_val_loss = 5
    model = cnn().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)#51,NOW ADD A 0 HERE
    criterion = nn.CrossEntropyLoss().to(device)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.2)  # 定义StepLR调度程序
    # criterion = nn.BCELoss().to(device)
    for epoch in tqdm(range(epoch_num), ascii=True):
        train_loss = []
        for batch_idx, (data, target) in enumerate(Dtr, 0):
            data, target = Variable(data).to(device), Variable(target.long()).to(device)
            # target = target.view(target.shape[0], -1)
            # print(target)
            optimizer.zero_grad()
            output = model(data)
            # print(output)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.cpu().item())
        # validation
        scheduler.step()
        val_loss = get_val_loss(model, Val)
        model.train()
        if epoch + 1 > min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)

        tqdm.write('Epoch {:03d} train_loss {:.5f} val_loss {:.5f}'.format(epoch, np.mean(train_loss), val_loss))

    torch.save(best_model.state_dict(), f"model/cnn{input}.pkl")


def test(input,b):
    if input == '0':
        Dtr, Val, Dte = load_data_0(b)
    if input == '1':
        Dtr, Val, Dte = load_data_1(b)
    if input == '2':
        Dtr, Val, Dte = load_data_2(b)
    if input == '3':
        Dtr, Val, Dte = load_data_3(b)
    if input == '4':
        Dtr, Val, Dte = load_data_4(b)
    if input == '5':
        Dtr, Val, Dte = load_data_5(b)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cnn().to(device)
    model.load_state_dict(torch.load(f"model/cnn{input}.pkl"), False)
    model.eval()
    total = 0
    current = 0
    for (data, target) in Dte:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        predicted = torch.max(outputs.data, 1)[1].data
        total += target.size(0)
        current += (predicted == target).sum()

    print('Accuracy:%d%%' % (100 * current / total))





    predictions_probabilities_all, true_labels_all = get_predictions_and_labels(model, Dte)

    num_classes = 3

    predictions_probabilities_list = []  # 保存模型预测的概率
    true_labels_list = []  # 保存真实标签

    with torch.no_grad():
        for imgs, labels in Dte:
            imgs = imgs.to(device)
            labels = labels.to(device)

            output = model(imgs)
            predictions_probabilities = torch.softmax(output, dim=1)

            predictions_probabilities_list.append(predictions_probabilities.cpu().numpy())
            true_labels_list.append(labels.cpu().numpy())

    predictions_probabilities_all = np.concatenate(predictions_probabilities_list)
    true_labels_all = np.concatenate(true_labels_list)

    # 获取测试集中实际存在的类别
    existing_classes = np.unique(true_labels_all)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in existing_classes:
        # 如果测试集中存在当前类别的标签
        if np.any(true_labels_all == i):
            fpr[i], tpr[i], _ = roc_curve(true_labels_all == i, predictions_probabilities_all[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

    # 计算平均 ROC 曲线和 AUC
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(mean_fpr)
    for i in existing_classes:
        mean_tpr += np.interp(mean_fpr, fpr[i], tpr[i])
    mean_tpr /= len(existing_classes)

    # 计算平均 AUC
    mean_auc = auc(mean_fpr, mean_tpr)
    print("Mean AUC:", mean_auc)

    # 绘制平均 ROC 曲线
    plt.figure()
    plt.plot(mean_fpr, mean_tpr, label='Mean ROC curve (AUC = {0:0.2f})'.format(mean_auc))
    plt.plot([0, 1], [0, 1], 'k--')  # 对角线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()










    # 首先定义一个 分类数*分类数 的空混淆矩阵
    conf_matrix = torch.zeros(3, 3)
    # 使用torch.no_grad()可以显著降低测试用例的GPU占用
    with torch.no_grad():
        for step, (imgs, targets) in enumerate(Dte):
            # imgs:     torch.Size([50, 3, 200, 200])   torch.FloatTensor
            # targets:  torch.Size([50, 1]),     torch.LongTensor  多了一维，所以我们要把其去掉
            #targets = targets.squeeze()  # [50,1] ----->  [50]

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

    #print("混淆矩阵总元素个数：{0},测试集总个数:{1}".format(int(np.sum(conf_matrix)), test_num))
    print(conf_matrix)

    # 获取每种Emotion的识别准确率
    print("测试集中，分类总个数：", per_kinds)
    print("每种评分预测正确的个数：", corrects)
    print("每种评分的识别准确率为：{0}".format([rate * 100 for rate in corrects / per_kinds]))

    # 绘制混淆矩阵
    Emotion = 3  # 这个数值是具体的分类数，大家可以自行修改
    labels = ['0','1', '2']  # 每种类别的标签

    # 显示数据
    plt.imshow(conf_matrix, cmap=plt.cm.Blues)

    # 在图中标注数量/概率信息
    thresh = conf_matrix.max() / 2  # 数值颜色阈值，如果数值超过这个，就颜色加深。
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


if __name__ == '__main__':
    print('0代表astringent')
    print('1代表sweet')
    print('2代表sour')
    print('3代表salt')
    print('4代表umami')
    print('5代表bitter')
    a = input('请问需要输入那种口味的数据:')
    print('0代表普通数据处理')
    print('1代表图数据处理')
    b = input('这决定使用的是图数据的处理或者是普通的数据处理')

    train(a,b)
    test(a,b)
