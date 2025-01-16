"""
该脚本使用图神经网络(GCN)对指定口味的数据集进行训练与测试，同时输出测试准确率、ROC曲线(AUC)以及混淆矩阵。
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# 数据加载脚本
from data_process_astringent import load_data_0
from data_process_bitter import load_data_5
from data_process_salt import load_data_3
from data_process_sour import load_data_2
from data_process_sweet import load_data_1
from data_process_umami import load_data_4

# ======== 全局参数设置 ========
batch_size = 64

# ======== 数据预处理 (若有需要可根据项目修改) ========
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def confusion_matrix(preds: torch.Tensor, labels: torch.Tensor, conf_matrix: torch.Tensor):
    """
    更新混淆矩阵：
    preds: 模型的预测输出 (batch_size, num_classes)
    labels: 真实标签向量 (batch_size)
    conf_matrix: 当前的混淆矩阵 (num_classes, num_classes)
    """
    preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


# ======== 主流程：选择口味和处理方式 ========
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

# 根据输入选择不同的训练集与测试集
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


class GCN(nn.Module):
    """
    定义简单的GCN模型结构：
    - 4层GCNConv
    - 经过全局平均池化后输出 (batch_size, num_classes)
    """
    def __init__(self):
        super(GCN, self).__init__()
        # 输入特征2 -> 128
        self.conv1 = GCNConv(2, 128)
        # 128 -> 64
        self.conv2 = GCNConv(128, 64)
        # 64 -> 18
        self.conv3 = GCNConv(64, 18)
        # 18 -> 3 (三类分类)
        self.conv4 = GCNConv(18, 3)

    def forward(self, data):
        """
        data: 包含 x（节点特征）、edge_index（图的边）、batch（图的批次索引）等信息
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        x = self.conv4(x, edge_index)

        # 使用 batch 索引聚合节点（将每张图的节点信息聚合成图级别表示）
        x = global_mean_pool(x, batch)

        # 输出对数概率 (log_softmax)
        return F.log_softmax(x, dim=1)


def train_model(model: nn.Module, optimizer, train_loader, num_epochs: int):
    """
    训练GCN模型并打印每个epoch的平均loss。
    :param model: 要训练的模型
    :param optimizer: 优化器
    :param train_loader: 训练集 DataLoader (Graph DataLoader)
    :param num_epochs: 训练轮数
    """
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            output = model(data)
            loss = F.nll_loss(output, data.y)  # 对数似然损失
            loss.backward()

            optimizer.step()
            # data.num_graphs 表示本批次含有的图数量
            total_loss += loss.item() * data.num_graphs

        average_loss = total_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss}')
        scheduler.step()  # 更新学习率


def test_model(model: nn.Module, test_loader):
    """
    测试模型的准确率。
    :param model: 训练好的模型
    :param test_loader: 测试集 DataLoader (Graph DataLoader)
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


def get_predictions_and_labels(model: nn.Module, Dte) -> (np.ndarray, np.ndarray):
    """
    获取模型在测试集上的预测概率与真实标签。
    :param model: 已训练的 PyTorch 模型
    :param Dte: 测试集 DataLoader (Graph DataLoader)
    :return: (predictions_probabilities_all, true_labels_all)，均为 numpy 数组
    """
    model.eval()
    predictions_probabilities_list = []
    true_labels_list = []

    with torch.no_grad():
        for data in Dte:
            data = data.to(device)
            y = data.y
            output = model(data)
            predictions_probabilities = F.softmax(output, dim=1)

            predictions_probabilities_list.append(predictions_probabilities.cpu().numpy())
            true_labels_list.append(y.cpu().numpy())

    predictions_probabilities_all = np.concatenate(predictions_probabilities_list)
    true_labels_all = np.concatenate(true_labels_list)
    return predictions_probabilities_all, true_labels_all


# ======== 模型及训练参数设定 ========
device = 'cpu'  # 如需使用GPU请改为: torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
optimizer = Adam(model.parameters(), lr=1.2e-6)
scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

# 使用 PyG DataLoader 进行批处理
train_loader = GeoDataLoader(train_dataset, batch_size=32, shuffle=False, drop_last=True)
test_loader = GeoDataLoader(test_dataset, batch_size=8, shuffle=False, drop_last=True)

# ======== 训练模型 ========
train_model(model, optimizer, train_loader, num_epochs=4)

# ======== 保存模型参数 ========
os.makedirs('model', exist_ok=True)
torch.save(model.state_dict(), f"model/GCN{a}.pth")

# ======== 测试模型 ========
test_model(model, test_loader)

# ======== 计算并绘制 ROC、AUC ========
predictions_probabilities_all, true_labels_all = get_predictions_and_labels(model, test_loader)

num_classes = 3
# 再次遍历，避免重复预测，也可以直接使用上面 get_predictions_and_labels 得到的结果
predictions_probabilities_list = []
true_labels_list = []
with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        y = data.y
        output = model(data)
        predictions_probabilities = F.softmax(output, dim=1)
        predictions_probabilities_list.append(predictions_probabilities.cpu().numpy())
        true_labels_list.append(y.cpu().numpy())

predictions_probabilities_all = np.concatenate(predictions_probabilities_list)
true_labels_all = np.concatenate(true_labels_list)

# 统计测试集中实际存在的类别
existing_classes = np.unique(true_labels_all)

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in existing_classes:
    # 对于每个存在的类别，计算fpr、tpr和auc
    if np.any(true_labels_all == i):
        fpr[i], tpr[i], _ = roc_curve(true_labels_all == i, predictions_probabilities_all[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

# 计算平均ROC曲线和AUC
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

# ======== 绘制并输出混淆矩阵 ========
conf_matrix = torch.zeros(3, 3)
with torch.no_grad():
    for step, data in enumerate(test_loader):
        targets = data.y
        out = model(data)
        conf_matrix = confusion_matrix(out, targets, conf_matrix)
        conf_matrix = conf_matrix.cpu()

conf_matrix = np.array(conf_matrix.cpu())
corrects = conf_matrix.diagonal(offset=0)
per_kinds = conf_matrix.sum(axis=0)

print(conf_matrix)
print("测试集中，打分总个数：", per_kinds)
print("每种打分预测正确的个数：", corrects)

# 绘制混淆矩阵图
labels = ['0', '1', '2']  # 根据需要修改
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()

# 在图中标注每个格子对应的数值
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
