"""
@Time   : 2024/03/26 11:34
@Author : 李颢
@File   : CNN.py
@Motto  : GO UP 
"""

import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
from tqdm import tqdm

# 根据需求自行修改这些导入方式
from data_process_astringent import load_data_0
from data_process_sweet import load_data_1
from data_process_sour import load_data_2
from data_process_bitter import load_data_5
from data_process_umami import load_data_4
from data_process_salt import load_data_3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seed(seed: int):
    """
    设置随机数种子，保证实验结果可复现。
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(20)


def get_predictions_and_labels(model: nn.Module, Dte):
    """
    给定模型和测试集迭代器，返回模型预测概率与真实标签的组合。
    
    :param model: 已训练好的 PyTorch 模型
    :param Dte: 测试集 DataLoader
    :return: (predictions_probabilities_all, true_labels_all) 都是 numpy 数组
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    predictions_probabilities_list = []
    true_labels_list = []

    with torch.no_grad():
        for imgs, labels in Dte:
            imgs = imgs.to(device)
            labels = labels.to(device)

            output = model(imgs)
            # 使用 softmax 得到每个类别的概率分布
            predictions_probabilities = torch.softmax(output, dim=1)

            predictions_probabilities_list.append(predictions_probabilities.cpu().numpy())
            true_labels_list.append(labels.cpu().numpy())

    # 将列表拼接成整体的 numpy 数组
    predictions_probabilities_all = np.concatenate(predictions_probabilities_list)
    true_labels_all = np.concatenate(true_labels_list)

    return predictions_probabilities_all, true_labels_all


def confusion_matrix(preds: torch.Tensor, labels: torch.Tensor, conf_matrix: torch.Tensor):
    """
    使用预测结果与真实标签，更新混淆矩阵。
    preds: 模型的预测输出 (batch_size, num_classes)
    labels: 真实标签向量 (batch_size)
    conf_matrix: 当前的混淆矩阵 (num_classes, num_classes)
    """
    # 取出每个样本预测出的概率最高的类别索引
    preds = torch.argmax(preds, 1)
    # 更新混淆矩阵
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


class cnn(nn.Module):
    """
    定义 CNN 模型的网络结构，包含 3 个卷积模块 + 3 个全连接层输出。
    """
    def __init__(self):
        super(cnn, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        
        self.fc1 = nn.Linear(3 * 3 * 64, 64)
        self.fc2 = nn.Linear(64, 18)
        self.out = nn.Linear(18, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # 将卷积输出拉平
        x = x.view(x.shape[0], -1)
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # 使用 Log Softmax 作为输出
        x = F.log_softmax(x, dim=1)
        return x


def get_val_loss(model: nn.Module, Val):
    """
    计算验证集上的平均 Loss，用于在训练过程中监控模型的收敛情况。
    
    :param model: 已训练好的 PyTorch 模型
    :param Val: 验证集 DataLoader
    :return: 验证集上的平均 Loss (float)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss().to(device)
    val_loss = []

    for (data, target) in Val:
        data, target = Variable(data).to(device), Variable(target.long()).to(device)
        output = model(data)
        loss = criterion(output, target)
        val_loss.append(loss.cpu().item())

    return np.mean(val_loss)


def train(input_taste: str, b: str):
    """
    训练指定口味（astringent、sweet、sour、salt、umami、bitter）的 CNN 模型。
    
    :param input_taste: 口味类型的标识字符，比如 '0'（astringent），'1'（sweet）等
    :param b: 决定使用何种方式处理数据 ('0'代表普通数据处理, '1'代表图数据处理)
    """
    # 根据传入的口味参数加载对应的数据
    if input_taste == '0':
        Dtr, Val, Dte = load_data_0(b)
    elif input_taste == '1':
        Dtr, Val, Dte = load_data_1(b)
    elif input_taste == '2':
        Dtr, Val, Dte = load_data_2(b)
    elif input_taste == '3':
        Dtr, Val, Dte = load_data_3(b)
    elif input_taste == '4':
        Dtr, Val, Dte = load_data_4(b)
    elif input_taste == '5':
        Dtr, Val, Dte = load_data_5(b)
    else:
        raise ValueError("输入口味无效，请输入[0,1,2,3,4,5]中的一个。")

    print('training...')
    epoch_num = 5          # 训练的总轮数
    best_model = None      # 用于保存在验证集上表现最好的模型
    min_epochs = 1         # 从第几轮开始允许替换 best_model
    min_val_loss = 5       # 用于记录最好的验证集 Loss

    model = cnn().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)  # 学习率可根据需要做调整
    criterion = nn.CrossEntropyLoss().to(device)
    # 定义 StepLR 学习率调度器
    scheduler = StepLR(optimizer, step_size=1, gamma=0.2)

    for epoch in tqdm(range(epoch_num), ascii=True):
        # 训练模式
        model.train()
        train_loss = []

        for batch_idx, (data, target) in enumerate(Dtr, 0):
            data, target = Variable(data).to(device), Variable(target.long()).to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.cpu().item())

        # 每个 epoch 结束后进行一次验证
        scheduler.step()
        val_loss = get_val_loss(model, Val)

        # 如果满足阈值条件，则更新 best_model
        if epoch + 1 > min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)

        tqdm.write('Epoch {:03d} train_loss {:.5f} val_loss {:.5f}'.format(
            epoch, np.mean(train_loss), val_loss))

    # 保存在验证集上表现最好的模型
    torch.save(best_model.state_dict(), f"model/cnn{input_taste}.pkl")


def test(input_taste: str, b: str):
    """
    测试指定口味的 CNN 模型在测试集上的表现，并输出准确率、ROC 曲线、AUC、混淆矩阵等结果。
    
    :param input_taste: 口味类型的标识字符，比如 '0'（astringent），'1'（sweet）等
    :param b: 决定使用何种方式处理数据 ('0'代表普通数据处理, '1'代表图数据处理)
    """
    # 根据传入的口味参数加载对应的数据
    if input_taste == '0':
        Dtr, Val, Dte = load_data_0(b)
    elif input_taste == '1':
        Dtr, Val, Dte = load_data_1(b)
    elif input_taste == '2':
        Dtr, Val, Dte = load_data_2(b)
    elif input_taste == '3':
        Dtr, Val, Dte = load_data_3(b)
    elif input_taste == '4':
        Dtr, Val, Dte = load_data_4(b)
    elif input_taste == '5':
        Dtr, Val, Dte = load_data_5(b)
    else:
        raise ValueError("输入口味无效，请输入[0,1,2,3,4,5]中的一个。")

    # 加载训练好的模型参数
    model = cnn().to(device)
    model.load_state_dict(torch.load(f"model/cnn{input_taste}.pkl"), strict=False)
    model.eval()

    # 计算测试集上整体准确率
    total = 0
    current = 0
    for (data, target) in Dte:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        predicted = torch.max(outputs.data, 1)[1].data
        total += target.size(0)
        current += (predicted == target).sum()
    print('Accuracy: {:.2f}%'.format(100.0 * current / total))

    # ======== 计算 ROC、AUC ========
    predictions_probabilities_all, true_labels_all = get_predictions_and_labels(model, Dte)

    # 统计测试集中实际存在的类别
    existing_classes = np.unique(true_labels_all)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in existing_classes:
        # 只计算在测试集中实际出现的类别
        if np.any(true_labels_all == i):
            fpr[i], tpr[i], _ = roc_curve(true_labels_all == i, predictions_probabilities_all[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

    # 计算平均 ROC 曲线和平均 AUC
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(mean_fpr)
    for i in existing_classes:
        mean_tpr += np.interp(mean_fpr, fpr[i], tpr[i])
    mean_tpr /= len(existing_classes)

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

    # ======== 计算混淆矩阵 ========
    conf_matrix = torch.zeros(3, 3)
    with torch.no_grad():
        for step, (imgs, targets) in enumerate(Dte):
            imgs = imgs.to(device)
            targets = targets.to(device)
            out = model(imgs)
            conf_matrix = confusion_matrix(out, targets, conf_matrix)
    conf_matrix = conf_matrix.cpu().numpy()

    # 计算对角线“预测正确数”和各列“总数”
    corrects = conf_matrix.diagonal(offset=0)
    per_kinds = conf_matrix.sum(axis=0)

    print("混淆矩阵:")
    print(conf_matrix)
    print("各类测试样本总数: ", per_kinds)
    print("各类预测正确个数: ", corrects)
    print("各类预测正确率(%): ", [rate * 100 for rate in corrects / per_kinds])

    # ======== 绘制混淆矩阵 ========
    labels = ['0', '1', '2']  # 仅适合3分类的标签，可根据需要修改
    plt.imshow(conf_matrix, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    # 在图中标注数据
    thresh = conf_matrix.max() / 2
    for x in range(conf_matrix.shape[0]):
        for y in range(conf_matrix.shape[1]):
            info = int(conf_matrix[y, x])
            plt.text(x, y, info,
                     verticalalignment='center',
                     horizontalalignment='center',
                     color="white" if info > thresh else "black")

    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)
    plt.tight_layout()
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

    train(a, b)
    test(a, b)
