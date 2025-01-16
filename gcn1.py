"""
该脚本使用简单的 GCN 模型来对指定口味的数据进行训练与测试，并输出测试准确率、ROC 曲线 (AUC) 和混淆矩阵等结果。
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
import matplotlib.pyplot as plt

from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torchvision import datasets, transforms

# 加载数据的脚本
from data_process_astringent import load_data_0
from data_process_bitter import load_data_5
from data_process_salt import load_data_3
from data_process_sour import load_data_2
from data_process_sweet import load_data_1
from data_process_umami import load_data_4

from sklearn.metrics import roc_curve, auc


# ======== 全局参数 & 数据预处理 ========
batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def confusion_matrix(preds: torch.Tensor, labels: torch.Tensor, conf_matrix: torch.Tensor):
    """
    更新混淆矩阵:
    - preds: (batch_size, num_classes)
    - labels: (batch_size,)
    - conf_matrix: (num_classes, num_classes)
    
    注意这里在取索引时减1，视具体标签编码而定
    """
    preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[p - 1, t - 1] += 1
    return conf_matrix


# ======== 主流程：选择口味 & 处理方式 ========
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

if a == '0':
    train_dataset, test_dataset = load_data_0(b)
elif a == '1':
    train_dataset, test_dataset = load_data_1(b)
elif a == '2':
    train_dataset, test_dataset = load_data_2(b)
elif a == '3':
    train_dataset, test_dataset = load_data_3(b)
elif a == '4':
    train_dataset, test_dataset = load_data_4(b)
elif a == '5':
    train_dataset, test_dataset = load_data_5(b)
else:
    raise ValueError("请输入[0,1,2,3,4,5]中的一个数字。")


# ======== 定义 GCN 模型 ========
class GCN(nn.Module):
    """
    一个简单的 GCN 模型：
    - conv1: (2 -> 64)
    - conv2: (64 -> 3)
    - 通过 global_mean_pool 将节点特征汇聚成图级别特征
    """
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(2, 64)
        self.conv2 = GCNConv(64, 3)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)

        # 将节点特征聚合为图特征
        x = global_mean_pool(x, batch)

        # 使用对数似然输出
        return torch.log_softmax(x, dim=1)


def train_model(model: nn.Module, optimizer, train_loader, num_epochs: int):
    """
    训练 GCN 模型，并打印每个 epoch 的平均 loss。
    :param model: GCN 模型
    :param optimizer: 优化器 (Adam / SGD ...)
    :param train_loader: 训练集 DataLoader (Graph)
    :param num_epochs: 训练轮数
    """
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = nn.functional.nll_loss(output, data.y)  # 对数似然损失
            loss.backward()
            optimizer.step()
            # 累加损失
            total_loss += loss.item() * data.num_graphs

        # 计算平均损失
        average_loss = total_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}')


def test_model(model: nn.Module, test_loader):
    """
    测试模型的准确率。
    :param model: GCN 模型
    :param test_loader: 测试集 DataLoader (Graph)
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += data.y.size(0)
            correct += (predicted == data.y).sum().item()
    print(f'Accuracy: {100.0 * correct / total:.2f}%')


def get_predictions_and_labels(model: nn.Module, loader):
    """
    获取预测概率与真实标签。
    :param model: 已训练好的 GCN 模型
    :param loader: 测试集 DataLoader (Graph)
    :return: (predictions_probabilities_all, true_labels_all)
    """
    model.eval()
    predictions_probabilities_list = []
    true_labels_list = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            predictions_probabilities = torch.softmax(output, dim=1)

            predictions_probabilities_list.append(predictions_probabilities.cpu().numpy())
            true_labels_list.append(data.y.cpu().numpy())

    predictions_probabilities_all = np.concatenate(predictions_probabilities_list)
    true_labels_all = np.concatenate(true_labels_list)
    return predictions_probabilities_all, true_labels_all


# ======== 设备、模型、优化器、数据加载器 ========
device = 'cpu'  # 如需使用 GPU: torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.035)

train_loader = GeoDataLoader(train_dataset, batch_size=64, shuffle=False, drop_last=True)
test_loader = GeoDataLoader(test_dataset, batch_size=8, shuffle=False, drop_last=True)

# ======== 训练 & 测试 ========
train_model(model, optimizer, train_loader, num_epochs=5)
torch.save(model.state_dict(), f"model/GCN{a}.pth")
test_model(model, test_loader)

# ======== 计算并绘制 ROC、AUC ========
predictions_probabilities_all, true_labels_all = get_predictions_and_labels(model, test_loader)

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
plt.plot(mean_fpr, mean_tpr, label='Mean ROC curve (AUC = {0:0.2f})'.format(mean_auc))
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
    for data in test_loader:
        targets = data.y
        out = model(data)
        conf_matrix = confusion_matrix(out, targets, conf_matrix)
        conf_matrix = conf_matrix.cpu()

conf_matrix = np.array(conf_matrix.cpu())
corrects = conf_matrix.diagonal(offset=0)
per_kinds = conf_matrix.sum(axis=1)

print(conf_matrix)
print("测试集中，每种打分总个数:", per_kinds)
print("每种打分预测正确的个数:", corrects)
print("每种打分的识别准确率: ", [rate * 100 for rate in corrects / per_kinds])

labels = ['0', '1', '2']  # 3 类标签
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
