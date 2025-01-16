"""
该脚本基于 ResNet(Bottleneck) 架构对指定口味数据进行训练与测试，并输出测试准确率、ROC 曲线(AUC) 以及混淆矩阵等结果。
"""

import os
import math
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from matplotlib import pyplot as plt
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc

# 自定义工具 & 数据加载脚本
import utils
from data_process_astringent import load_data_0
from data_process_sweet import load_data_1
from data_process_sour import load_data_2
from data_process_bitter import load_data_5
from data_process_umami import load_data_4
from data_process_salt import load_data_3


num_classes = 3  # 统一使用3分类

# ======== 1. 混淆矩阵 & 预测输出 ========
def confusion_matrix(preds: torch.Tensor, labels: torch.Tensor, conf_matrix: torch.Tensor):
    """
    基于 preds 和 labels 更新混淆矩阵:
    - preds: (batch_size, num_classes)，需先取 argmax 变成 (batch_size,)
    - labels: (batch_size,)
    - conf_matrix: (num_classes, num_classes)
    """
    preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


def setup_seed(seed: int):
    """
    设置随机种子，以保证实验结果可复现。
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_predictions_and_labels(model: nn.Module, Dte):
    """
    获取模型在测试集上的预测概率与真实标签。
    :param model: 训练好的 PyTorch 模型
    :param Dte: 测试集 DataLoader
    :return: (predictions_probabilities_all, true_labels_all)
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
            preds_probs = torch.softmax(output, dim=1)

            predictions_probabilities_list.append(preds_probs.cpu().numpy())
            true_labels_list.append(labels.cpu().numpy())

    predictions_probabilities_all = np.concatenate(predictions_probabilities_list)
    true_labels_all = np.concatenate(true_labels_list)
    return predictions_probabilities_all, true_labels_all


# ======== 再次定义 confusion_matrix (与上面一致，保持原逻辑) ========
def confusion_matrix(preds: torch.Tensor, labels: torch.Tensor, conf_matrix: torch.Tensor):
    preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix

# ======== 2. 定义 ResNet 相关模块 (Bottleneck) ========
def conv3x3(in_planes, out_planes, stride=1):
    """
    3x3 卷积 + padding
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    """
    ResNet 的 BasicBlock 模块 (非 Bottleneck)
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # 如果网络尺寸不匹配，需要下采样
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    """
    ResNet 的 Bottleneck 模块
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # 1x1 卷积
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1,
                               stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # 3x3 卷积
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # 1x1 卷积
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # 若通道数或步幅不匹配，需要下采样
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """
    自定义 ResNet 类，可通过 block 和 layers 定义具体结构。
    这里以 ResNet50 (Bottleneck, [3,4,6,3]) 为例。
    """
    def __init__(self, block, layers, num_classes=8631, include_top=True):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.include_top = include_top

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2,
                                    padding=0, ceil_mode=True)

        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc      = nn.Linear(512 * block.expansion, num_classes)

        # 初始化参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # kaiming 初始化
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        生成 ResNet 的一层 (由多个 bottleneck / basicblock 组成)。
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 如果尺寸或通道数不匹配，需要下采样
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # 第一个 block
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        # 之后的 block
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 4 大层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        if not self.include_top:
            return x

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def get_val_loss(model: nn.Module, Val):
    """
    获取验证集上的平均 loss
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    训练指定口味 (0~5) 的 ResNet 模型:
    :param input_taste: '0'(astringent), '1'(sweet), '2'(sour), '3'(salt), '4'(umami), '5'(bitter)
    :param b: '0' 代表普通数据处理, '1' 代表图数据处理
    """
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
    epoch_num = 5
    best_model = None
    min_epochs = 1
    min_val_loss = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("当前设备：", device)

    # 初始化模型
    model = ResNet(Bottleneck, [3, 4, 6, 3]).to(device)

    # 加载预训练权重 (resnet50_ft_weight.pkl) 并冻结
    model_path = 'resnet50_ft_weight.pkl'
    model = utils.load_state_dict(model, model_path)  # 根据需求自行修改 utils.load_state_dict

    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    # 替换最后一层全连接层
    model.fc = nn.Linear(8192, num_classes).to(device)
    print('模型结构:')
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss().to(device)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.3)

    for epoch in tqdm(range(epoch_num), ascii=True):
        model.train()
        train_loss = []
        for inputs, labels in Dtr:
            inputs, labels = Variable(inputs).to(device), Variable(labels.long()).to(device)
            optimizer.zero_grad()

            outputs = model(inputs).to(device)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.cpu().item())

        scheduler.step()
        val_loss = get_val_loss(model, Val)
        if epoch + 1 > min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)

        tqdm.write('Epoch {:03d}, train_loss {:.5f}, val_loss {:.5f}'
                   .format(epoch, np.mean(train_loss), val_loss))

    torch.save(best_model.state_dict(), f"model/cnn{input_taste}.pkl")


def test(input_taste: str, b: str):
    """
    测试指定口味 (0~5) 的 ResNet 模型:
    1. 输出准确率
    2. 绘制并输出 ROC、AUC
    3. 计算并绘制混淆矩阵
    """
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("当前设备：", device)

    # 加载模型并替换最后一层
    model = ResNet(Bottleneck, [3, 4, 6, 3]).to(device)
    model.fc = nn.Linear(8192, num_classes).to(device)

    # 加载训练好的参数
    model.load_state_dict(torch.load(f"model/cnn{input_taste}.pkl"), strict=False)
    model.eval()

    # ======== 计算测试集准确率 ========
    total = 0
    current = 0
    for (data, target) in Dte:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        predicted = torch.max(outputs.data, 1)[1].data
        total += target.size(0)
        current += (predicted == target).sum()
    print('Accuracy: {:.2f}%'.format(100.0 * current / total))

    # ======== 计算并绘制 ROC、AUC ========
    predictions_probabilities_all, true_labels_all = get_predictions_and_labels(model, Dte)
    predictions_probabilities_list = []
    true_labels_list = []
    with torch.no_grad():
        for imgs, labels in Dte:
            imgs = imgs.to(device)
            labels = labels.to(device)
            out = model(imgs)
            preds_probs = torch.softmax(out, dim=1)
            predictions_probabilities_list.append(preds_probs.cpu().numpy())
            true_labels_list.append(labels.cpu().numpy())

    predictions_probabilities_all = np.concatenate(predictions_probabilities_list)
    true_labels_all = np.concatenate(true_labels_list)

    existing_classes = np.unique(true_labels_all)
    fpr, tpr, roc_auc = dict(), dict(), dict()

    for i in existing_classes:
        if np.any(true_labels_all == i):
            fpr[i], tpr[i], _ = roc_curve(true_labels_all == i,
                                         predictions_probabilities_all[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(mean_fpr)
    for i in existing_classes:
        mean_tpr += np.interp(mean_fpr, fpr[i], tpr[i])
    mean_tpr /= len(existing_classes)

    mean_auc = auc(mean_fpr, mean_tpr)
    print("Mean AUC:", mean_auc)

    plt.figure()
    plt.plot(mean_fpr, mean_tpr, label='Mean ROC curve (AUC = {:.2f})'.format(mean_auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    # ======== 计算并绘制混淆矩阵 ========
    conf_matrix = torch.zeros(3, 3)
    with torch.no_grad():
        for step, (imgs, targets) in enumerate(Dte):
            imgs = imgs.to(device)
            targets = targets.to(device)
            out = model(imgs)
            conf_matrix = confusion_matrix(out, targets, conf_matrix)
            conf_matrix = conf_matrix.cpu()

    conf_matrix = np.array(conf_matrix.cpu())
    corrects = conf_matrix.diagonal(offset=0)
    per_kinds = conf_matrix.sum(axis=0)

    print(conf_matrix)
    print("测试集中，分类总个数:", per_kinds)
    print("每种评分预测正确的个数:", corrects)
    print("每种评分的识别准确率:", [rate * 100 for rate in corrects / per_kinds])

    labels = ['0', '1', '2']
    plt.imshow(conf_matrix, cmap=plt.cm.Blues)

    thresh = conf_matrix.max() / 2
    for x in range(3):
        for y in range(3):
            info = int(conf_matrix[y, x])
            plt.text(x, y, info,
                     verticalalignment='center',
                     horizontalalignment='center',
                     color="white" if info > thresh else "black")

    plt.tight_layout()
    plt.yticks(range(3), labels)
    plt.xticks(range(3), labels, rotation=45)
    plt.show()
    plt.close()


if __name__ == '__main__':
    print('0代表astringent')
    print('1代表sweet')
    print('2代表sour')
    print('3代表salt')
    print('4代表umami')
    print('5代表bitter')
    taste = input('请问需要输入那种口味的数据:')
    print('0代表普通数据处理')
    print('1代表图数据处理')
    b = input('这决定使用的是图数据的处理或者是普通的数据处理')

    train(taste, b)
    test(taste, b)
