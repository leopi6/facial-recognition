import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
import numpy as np
import torch_geometric
from data_process_astringent import load_data_0
from data_process_bitter import load_data_5
from data_process_salt import load_data_3
from data_process_sour import load_data_2
from data_process_sweet import load_data_1
from data_process_umami import load_data_4
from sklearn.metrics import roc_curve, auc

batch_size = 64
# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[p - 1, t - 1] += 1
    return conf_matrix


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
if a == '1':
    train_dataset, test_dataset = load_data_1(b)
if a == '2':
    train_dataset, test_dataset = load_data_2(b)
if a == '3':
    train_dataset, test_dataset = load_data_3(b)
if a == '4':
    train_dataset, test_dataset = load_data_4(b)
if a == '5':
    train_dataset, test_dataset = load_data_5(b)


# 定义GCN模型
class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(2, 64)
        self.conv2 = GCNConv(64, 3)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)

        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch_geometric.nn.global_mean_pool(x, batch)  # 使用batch索引来聚合节点到各自的图
        # 将输出调整为 (batch_size, num_classes) 的形状

        return torch.log_softmax(x, dim=1)


# 训练模型
def train_model(model, optimizer, train_loader, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            # 假设 data.y 的形状为 (8,)
            loss = nn.functional.nll_loss(output, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
        average_loss = total_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss}')


# 初始化模型和优化器
# 测试模型
def test_model(model, test_loader):
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
    print(f'Accuracy: {100 * correct / total}%')

def get_predictions_and_labels(model, Dte):
     # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    predictions_probabilities_list = []  # 保存模型预测的概率
    true_labels_list = []  # 保存真实标签

    with torch.no_grad():
        for data in Dte:
            data = data.to(device)
            y = data.y
            output = model(data)
            predictions_probabilities = torch.softmax(output, dim=1)

            predictions_probabilities_list.append(predictions_probabilities.cpu().numpy())
            true_labels_list.append(y.cpu().numpy())

        # 将 predictions_probabilities_list 和 true_labels_list 转换为 NumPy 数组
    predictions_probabilities_all = np.concatenate(predictions_probabilities_list)
    true_labels_all = np.concatenate(true_labels_list)

    return predictions_probabilities_all, true_labels_all


# 参数:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
model = GCN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.035)
train_loader = GeoDataLoader(train_dataset, batch_size=64, shuffle=False, drop_last=True)
test_loader = GeoDataLoader(test_dataset, batch_size=8, shuffle=False, drop_last=True)
train_model(model, optimizer, train_loader, num_epochs=5)
# 创建一个迭代器，从 train_loader 中获取数
torch.save(model.state_dict(), f"model/GCN{a}.pth")
test_model(model, test_loader)

# ROC----------------------------------------------------

predictions_probabilities_all, true_labels_all = get_predictions_and_labels(model, test_loader)

num_classes = 3

predictions_probabilities_list = []  # 保存模型预测的概率
true_labels_list = []  # 保存真实标签

with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        y = data.y

        output = model(data)
        predictions_probabilities = torch.softmax(output, dim=1)

        predictions_probabilities_list.append(predictions_probabilities.cpu().numpy())
        true_labels_list.append(y.cpu().numpy())

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

# 绘制混淆矩阵：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：
# 首先定义一个 分类数*分类数 的空混淆矩阵
conf_matrix = torch.zeros(3, 3)
# 使用torch.no_grad()可以显著降低测试用例的GPU占用
with torch.no_grad():
    for step, (data) in enumerate(test_loader):
        # imgs:     torch.Size([50, 3, 200, 200])   torch.FloatTensor
        # targets:  torch.Size([50, 1]),     torch.LongTensor  多了一维，所以我们要把其去掉
        # targets = targets.squeeze()  # [50,1] ----->  [50]

        # 将变量转为gpu
        targets = data.y

        out = model(data)
        # 记录混淆矩阵参数
        conf_matrix = confusion_matrix(out, targets, conf_matrix)
        conf_matrix = conf_matrix.cpu()
conf_matrix = np.array(conf_matrix.cpu())  # 将混淆矩阵从gpu转到cpu再转到np
corrects = conf_matrix.diagonal(offset=0)  # 抽取对角线的每种分类的识别正确个数
per_kinds = conf_matrix.sum(axis=1)  # 抽取每个分类数据总的测试条数

# print("混淆矩阵总元素个数：{0},测试集总个数:{1}".format(int(np.sum(conf_matrix)), test_num))
print(conf_matrix)

# 获取每种Emotion的识别准确率
print("测试集中，打分总个数：", per_kinds)
print("每种打分预测正确的个数：", corrects)
print("每种打分的识别准确率为：{0}".format([rate * 100 for rate in corrects / per_kinds]))

# 绘制混淆矩阵
Emotion = 3  # 这个数值是具体的分类数，大家可以自行修改
labels = ['0', '1', '2']  # 每种类别的标签

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
