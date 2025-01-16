"""
本脚本使用 LSTM 模型来对指定口味的数据集进行训练与测试，输出测试准确率、ROC 曲线 (AUC) 以及混淆矩阵等结果。
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# 根据需要加载不同的味觉数据处理脚本
from data_process_astringent import load_data_0
from data_process_sweet import load_data_1
from data_process_sour import load_data_2
from data_process_bitter import load_data_5
from data_process_umami import load_data_4
from data_process_salt import load_data_3


# ======== 混淆矩阵 & 预测输出函数 ========
def confusion_matrix(preds: torch.Tensor, labels: torch.Tensor, conf_matrix: torch.Tensor):
    """
    更新混淆矩阵:
    preds: 模型的预测输出 (batch_size, num_classes)
    labels: 真实标签 (batch_size,)
    conf_matrix: 当前的混淆矩阵 (num_classes, num_classes)
    """
    preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


def get_predictions_and_labels(model: nn.Module, Dte: DataLoader):
    """
    获取模型在测试集上的预测概率与真实标签。
    :param model: 已训练好的模型
    :param Dte: 测试集 DataLoader
    :return: (predictions_probabilities_all, true_labels_all) 
             均为 numpy 数组
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
            predictions_probabilities = torch.softmax(output, dim=1)

            predictions_probabilities_list.append(predictions_probabilities.cpu().numpy())
            true_labels_list.append(labels.cpu().numpy())

    # 拼接成整体的 numpy 数组
    predictions_probabilities_all = np.concatenate(predictions_probabilities_list)
    true_labels_all = np.concatenate(true_labels_list)
    return predictions_probabilities_all, true_labels_all


# ======== 超参数设置 ========
BATCH_SIZE = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 2

# ======== 数据预处理 (pipeline) ========
pipeline = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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

# 根据输入 a、b 加载对应的数据集 (训练集、验证集、测试集)
if a == '0':
    train_loader, Val, test_loader = load_data_0(b)
elif a == '1':
    train_loader, Val, test_loader = load_data_1(b)
elif a == '2':
    train_loader, Val, test_loader = load_data_2(b)
elif a == '3':
    train_loader, Val, test_loader = load_data_3(b)
elif a == '4':
    train_loader, Val, test_loader = load_data_4(b)
elif a == '5':
    train_loader, Val, test_loader = load_data_5(b)
else:
    raise ValueError("请输入 [0~5] 中的一个数字。")


# ======== 定义 LSTM 模型 ========
class LSTM_Model(nn.Module):
    """
    使用 LSTM 结构来处理图像数据。注意这里将图像展平后送入 LSTM。
    """
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTM_Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        # LSTM 层
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        # 全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        # 初始化隐藏层和 cell state
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(DEVICE)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(DEVICE)

        # 将图像展平: [batch, channels, height, width] -> [batch, 1, -1]
        x = x.view(x.size(0), 1, -1)

        # 前向传播 LSTM
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # 取最后时间步的输出做分类
        out = self.fc(out[:, -1, :])
        return out


# ======== 初始化 LSTM 模型 & 相关参数 ========
input_dim = 224 * 224 * 3  # 224x224x3
hidden_dim = 15
layer_dim = 4  # LSTM 层数
out_dim = 4
rnn_model = LSTM_Model(input_dim, hidden_dim, layer_dim, out_dim)

# 将模型放到 GPU / CPU
rnn_model = rnn_model.to(DEVICE)

# 查看模型参数
print("模型参数总数: ", len(list(rnn_model.parameters())))

# 定义优化器 & 损失函数
optimizer = optim.Adam(rnn_model.parameters(), lr=1e-3, betas=(0.9, 0.99))
criterion = nn.CrossEntropyLoss()

# 定义学习率衰减策略
scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

# ======== 开始训练 ========
iteration = 0
train_loss_list = []
train_accuracy_list = []
train_iteration_list = []

test_loss_list = []
test_accuracy_list = []
test_iteration_list = []

for epoch in range(EPOCHS):
    train_correct = 0.0
    train_total = 0.0

    for i, (imgs, labels) in enumerate(train_loader):
        rnn_model.train()
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = rnn_model(imgs.cuda())  # 模型放在 cuda
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 计算训练准确率
        _, train_predict = torch.max(outputs.data, 1)
        if torch.cuda.is_available():
            train_correct += (train_predict.cuda() == labels.cuda()).sum()
        else:
            train_correct += (train_predict == labels).sum()
        train_total += labels.size(0)
        accuracy = 100.0 * train_correct / train_total

        # 若是最后一轮 epoch，统计指标
        if (epoch + 1) == EPOCHS:
            iteration += 1
            train_loss_list.append(loss.item())
            train_accuracy_list.append(accuracy)
            train_iteration_list.append(iteration)

        print(f"Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {loss.item():.6f}, "
              f"Correct: {train_correct}/{train_total}, Accuracy: {accuracy:.2f}%")

    scheduler.step()
    print("==========================验证阶段==========================")

# ======== 测试阶段 ========
rnn_model.eval()
correct = 0.0
total = 0.0
with torch.no_grad():
    for j, (datas, targets) in enumerate(test_loader):
        datas, targets = datas.to(DEVICE), targets.to(DEVICE)
        outputs = rnn_model(datas)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        if torch.cuda.is_available():
            correct += (predicted.cuda() == targets.cuda()).sum()
        else:
            correct += (predicted == targets).sum()

accuracy = 100.0 * correct / total
test_accuracy_list.append(accuracy)
test_loss_list.append(loss.item())
test_iteration_list.append(iteration)

print(f"TEST -> Loop: {iteration}, Loss: {loss.item():.6f}, "
      f"Correct: {correct}, Total: {total}, Accuracy: {accuracy:.2f}%")

# 保存模型
torch.save(rnn_model.state_dict(), f"model/lstm{a}.pth")

# ======== 计算并绘制 ROC、AUC ========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
predictions_probabilities_all, true_labels_all = get_predictions_and_labels(rnn_model, test_loader)

# 收集预测概率与真实标签
predictions_probabilities_list = []
true_labels_list = []
rnn_model.eval()
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        output = rnn_model(imgs)
        probs = torch.softmax(output, dim=1)
        predictions_probabilities_list.append(probs.cpu().numpy())
        true_labels_list.append(labels.cpu().numpy())

predictions_probabilities_all = np.concatenate(predictions_probabilities_list)
true_labels_all = np.concatenate(true_labels_list)

# 获取测试集中实际出现的类别
existing_classes = np.unique(true_labels_all)
fpr, tpr, roc_auc = dict(), dict(), dict()

for i in existing_classes:
    if np.any(true_labels_all == i):
        fpr[i], tpr[i], _ = roc_curve(true_labels_all == i, predictions_probabilities_all[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

mean_fpr = np.linspace(0, 1, 100)
mean_tpr = np.zeros_like(mean_fpr)
for i in existing_classes:
    mean_tpr += np.interp(mean_fpr, fpr[i], tpr[i])
mean_tpr /= len(existing_classes)
mean_auc = auc(mean_fpr, mean_tpr)

print("Mean AUC:", mean_auc)

plt.figure()
plt.plot(mean_fpr, mean_tpr, label=f'Mean ROC curve (AUC = {mean_auc:.2f})')
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
    for step, (imgs, targets) in enumerate(test_loader):
        imgs, targets = imgs.cuda(), targets.cuda()
        out = rnn_model(imgs)
        conf_matrix = confusion_matrix(out, targets, conf_matrix)
        conf_matrix = conf_matrix.cpu()

conf_matrix = np.array(conf_matrix)
corrects = conf_matrix.diagonal(offset=0)
per_kinds = conf_matrix.sum(axis=0)

print(conf_matrix)
print("测试集中，每种打分总个数:", per_kinds)
print("每种打分预测正确的个数:", corrects)
print("每种打分的识别准确率:", [rate * 100 for rate in corrects / per_kinds])

labels = ['0', '1', '2']  # 3类标签
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
