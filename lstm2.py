def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix




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



import sklearn.metrics as metrics

"""
模型1：Pytorch LSTM 实现流程
    1.图片数据处理，加载数据集
    2.使得数据集可迭代（每次读取一个Batch）
    3.创建模型类
    4.初始化模型类
    5.初始化损失类
    6.训练模型
"""

# 1.加载库
from sklearn.metrics import roc_curve, auc
from torch.optim.lr_scheduler import StepLR
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt
from data_process_astringent import load_data_0
from data_process_sweet import load_data_1
from data_process_sour import load_data_2
from data_process_bitter import load_data_5
from data_process_umami import load_data_4
from data_process_salt import load_data_3

# 2.定义超参数
BATCH_SIZE = 64 # 每批处理的数据
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 放在cuda或者cpu上训练
EPOCHS = 2 # 训练数据集的轮次

# 3.构建pipeline，对图像做处理
pipeline = transforms.Compose([
    # 彩色图像转灰度图像num_output_channels默认1
    # transforms.Grayscale(num_output_channels=1),
    # 分辨率重置为256
    transforms.Resize(256),
    # 对加载的图像作归一化处理， 并裁剪为[224x224x3]大小的图像(因为这图片像素不一致直接统一)
    transforms.CenterCrop(224),
    # 将图片转成tensor
    transforms.ToTensor(),
    # 正则化，模型出现过拟合现象时，降低模型复杂度
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



#train_dataset = datasets.ImageFolder(root=base_dir_train, transform=pipeline)
#print("train_dataset=" + repr(train_dataset[1][0].size()))
#print("train_dataset.class_to_idx=" + repr(train_dataset.class_to_idx))
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
    train_loader, Val, test_loader = load_data_0(b)
if a == '1':
    train_loader, Val, test_loader = load_data_1(b)

if a == '2':
    train_loader, Val, test_loader = load_data_2(b)

if a == '3':
    train_loader, Val, test_loader = load_data_3(b)

if a == '4':
    train_loader, Val, test_loader = load_data_4(b)


if a == '5':
    train_loader, Val, test_loader = load_data_5(b)



# 5.定义函数，显示一批图片
def imShow(inp, title=None):
    # tensor转成numpy,transpose转成(通道数,长,宽)
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])  # 均值
    std = np.array([0.229, 0.224, 0.225])  # 标准差
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)  # 像素值限制在0-1之间
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


# 网格显示
#out = torchvision.utils.make_grid(images)
#imShow(out)


# 6.定义LSTM网络
class LSTM_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTM_Model, self).__init__()  # 初始化父类构造方法
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        # 构建LSTM模型
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        # 全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 初始化隐藏层状态全为0
        # (layer_dim,batch_size,hidden_dim)
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(DEVICE)
        # 初始化cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(DEVICE)
        x = x.view(x.size(0), 1, -1)
        # 分离隐藏状态 避免梯度爆炸
        '''
            RNN只有一个状态，而LSTM有两个状态，所以两个状态都要分离
        '''
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # 只需要最后一层隐层的状态
        out = self.fc(out[:, -1, :])
        return out

# 7.初始化模型
input_dim = 196608  # 输入维度(输入的节点数）256*256*3（三通道三个batch）
hidden_dim = 15  # 隐藏层的维度(每个隐藏层的节点数)
layer_dim = 4 # 2层LSTM(隐藏层的数量 2层)
out_dim = 4  # 输出维度
rnn_model = LSTM_Model(input_dim, hidden_dim, layer_dim, out_dim)

# 8.输出模型参数信息
length = len(list(rnn_model.parameters()))
print(length)

# 9.输出模型参数信息
length = len(list(rnn_model.parameters()))
print(length)

# 优化器
# optimizer = optim.SGD(rnn_model.parameters(), lr=1e-3, momentum=0.9)
optimizer = optim.Adam(rnn_model.parameters(), lr=1e-3, betas=(0.9, 0.99))

# 损失函数,交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 把损失，准确度，迭代都记录出list，然后讲loss和准确度画出图像
sequence_dim = 53
train_loss_list = []
train_accuracy_list = []
train_iteration_list = []

test_loss_list = []
test_accuracy_list = []
test_iteration_list = []

iteration = 0

# 优化器
# optimizer = optim.SGD(rnn_model.parameters(), lr=1e-3, momentum=0.9)
optimizer = optim.Adam(rnn_model.parameters(), lr=1e-3, betas=(0.9, 0.99))
# 定义学习率衰减策略
# step_size是每经过多少个epoch执行一次学习率调整，gamma是学习率衰减的乘数因子
scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

# 训练
# """
for epoch in range(EPOCHS):
    # 用来显示训练的loss correct等
    train_correct = 0.0
    train_total = 0.0

    for i, (imgs, labels) in enumerate(train_loader):
        # 声明训练，loss等只能在train mode下进行运算
        rnn_model.train()
        # 把训练的数据集合都扔到对应的设备去
        # imgs = imgs.view(-1,1,sequence_dim, input_dim).requires_grad_().to(DEVICE)
        # print("imgs shape", imgs.shape)
        # print("imgs = ", imgs.data)
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        # 防止梯度爆炸，梯度清零
        optimizer.zero_grad()
        # 前向传播
        rnn_model = rnn_model.cuda()  # 这里要从cuda()中取得，不然前面都放在cuda后面放在cpu，会报错，报“不在同一个设备的错误" Input and parameter tensors are not at the same device, found input tensor at cuda:0 and parameter tensor at cpu
        output = rnn_model(imgs)
        # print("RNN output shape", out.shape)
        # print("label shape", labels.shape)
        # 计算损失
        loss = criterion(output, labels)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 计算训练时候的准确度
        train_predict = torch.max(output.data, 1)[1]
        if torch.cuda.is_available():
            train_correct += (train_predict.cuda() == labels.cuda()).sum()
        else:
            train_correct += (train_predict == labels).sum()
        train_total += labels.size(0)
        accuracy = train_correct / train_total * 100.0
        # 只画出最后一次epoch的
        if (epoch + 1) == EPOCHS:
            # 迭代计数器++
            iteration += 1
            train_accuracy_list.append(accuracy)
            train_iteration_list.append(iteration)
            train_loss_list.append(loss)
        # 打印信息
        print("Epoch :%d , Batch : %5d , Loss : %.8f,train_correct:%d,train_total:%d,accuracy:%.6f" % (
            epoch + 1, i + 1, loss.item(), train_correct, train_total, accuracy))
    scheduler.step()
    print("==========================预测开始===========================")
    rnn_model.eval()
    # 验证accuracy
    
    
        
correct = 0.0
total = 0.0

    # 迭代测试集 获取数据 预测
for j, (datas, targets) in enumerate(test_loader):
    datas = datas.to(DEVICE)
    targets = targets.to(DEVICE)
        # datas = datas.view(-1, sequence_dim, input_dim).requires_grad_().to(DEVICE)
        # datas = datas.reshape(datas.size(0), 1, -1)
        # 模型预测
    outputs = rnn_model(datas)
        # 防止梯度爆炸，梯度清零
    optimizer.zero_grad()
        # 获取测试概率最大值的下标
    predicted = torch.max(outputs.data, 1)[1]
        # 统计计算测试集合
    total += targets.size(0)
    if torch.cuda.is_available():
            # print(predicted.cuda() == targets.cuda())
        correct += (predicted.cuda() == targets.cuda()).sum()
            # print("predicted.cuda()=" + repr(predicted.cuda()))
            # print("labels.cuda()=" + repr(targets.cuda()))
    else:
        correct += (predicted == targets).sum()
accuracy = correct / total * 100.0
test_accuracy_list.append(accuracy)
test_loss_list.append(loss.item())
test_iteration_list.append(iteration)
print("TEST--->loop : {}, Loss : {}, correct:{}, total:{}, Accuracy : {}".format(iteration, loss.item(),correct,total, accuracy))

torch.save(rnn_model.state_dict(), f"model/lstm{a}.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#AUC drawing :::：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：


predictions_probabilities_all, true_labels_all = get_predictions_and_labels(rnn_model, test_loader)

num_classes = 3

predictions_probabilities_list = []  # 保存模型预测的概率
true_labels_list = []  # 保存真实标签

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        output = rnn_model(imgs)
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

#绘制混淆矩阵：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：：
# 首先定义一个 分类数*分类数 的空混淆矩阵
conf_matrix = torch.zeros(3, 3)
    # 使用torch.no_grad()可以显著降低测试用例的GPU占用
with torch.no_grad():
    for step, (imgs, targets) in enumerate(test_loader):
            # imgs:     torch.Size([50, 3, 200, 200])   torch.FloatTensor
            # targets:  torch.Size([50, 1]),     torch.LongTensor  多了一维，所以我们要把其去掉
        #targets = targets.squeeze()  # [50,1] ----->  [50]

            # 将变量转为gpu
        targets = targets.cuda()
        imgs = imgs.cuda()
            # print(step,imgs.shape,imgs.type(),targets.shape,targets.type())

        out = rnn_model(imgs)
            # 记录混淆矩阵参数
        conf_matrix = confusion_matrix(out, targets, conf_matrix)
        conf_matrix = conf_matrix.cpu()
conf_matrix = np.array(conf_matrix.cpu())  # 将混淆矩阵从gpu转到cpu再转到np
corrects = conf_matrix.diagonal(offset=0)  # 抽取对角线的每种分类的识别正确个数
per_kinds = conf_matrix.sum(axis=0)  # 抽取每个分类数据总的测试条数

    #print("混淆矩阵总元素个数：{0},测试集总个数:{1}".format(int(np.sum(conf_matrix)), test_num))
print(conf_matrix)

    # 获取每种Emotion的识别准确率
print("测试集中，打分总个数：", per_kinds)
print("每种打分预测正确的个数：", corrects)
print("每种打分的识别准确率为：{0}".format([rate * 100 for rate in corrects / per_kinds]))

    # 绘制混淆矩阵
Emotion = 3 # 这个数值是具体的分类数，大家可以自行修改
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
