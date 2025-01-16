![](https://img.shields.io/badge/cnn-classification%20-orange)
# PK research
Implementation of facial recognition
项目名称
该项目包含多个深度学习模型（如 CNN、LSTM、GCN、ResNet 等），并针对不同口味（astringent, sweet, sour, salt, umami, bitter）的图像或图数据进行处理、训练与测试。本项目还提供了数据增强脚本及各种数据加载方式，方便快速上手。

功能概述
数据加载

提供如 load_data_0, load_data_1, load_data_2, load_data_3, load_data_4, load_data_5 等函数，分别对应不同口味数据的加载（astringent=0, sweet=1, sour=2, salt=3, umami=4, bitter=5）。
根据 choice 参数判断使用普通图像数据或图数据（.pt）进行加载，并拆分为训练集、验证集、测试集。
数据增强

提供了 data_augmentation 脚本示例，可以对图像或图数据进行相应的增广或删除操作（如检测某些文件并删除、亮度调整、旋转、翻转等）。
通过修改相应参数（如文件路径、删除阈值）可轻松适应不同需求。
模型训练

CNN：如在 CNN.py 中使用简单的卷积层 + MaxPool + 全连接层，对图像数据进行分类。
LSTM：以图像展平输入到 LSTM 中，或对时序/序列数据进行分类；训练、测试分别输出准确率、ROC、混淆矩阵等指标。
GCN：对图数据（如 .pt 文件）进行图神经网络训练，同样可输出准确率、ROC、混淆矩阵。
ResNet：提供了基于 Bottleneck 的 ResNet50 结构示例，可加载预训练权重并进行微调，适用于图像分类。
每个模型均提供 train(...) 与 test(...) 函数，通过命令行输入或脚本交互来指定口味与处理方式进行训练。
模型测试

加载训练好的权重，对测试集进行推断，并输出准确率、绘制并显示ROC 曲线 (AUC) 以及混淆矩阵。
提供如 get_predictions_and_labels(...)、confusion_matrix(...) 等工具函数，便于快速评价模型。
可视化

在训练或测试结束时，通过 matplotlib 绘制ROC 和混淆矩阵，帮助评估模型表现。
环境准备
Python 版本：建议 Python 3.7 及以上；
依赖库：
numpy
torch / torchvision / torch_geometric（若使用图神经网络）
matplotlib
scikit-learn
以及项目中可能提到的其他依赖（如 tqdm, cv2 等）。
安装方式：
bash
复制代码
pip install -r requirements.txt
（根据实际需求生成 requirements.txt 后再使用。）
文件说明
数据处理 / DataLoader

data_process_astringent.py / data_process_sweet.py / data_process_sour.py / data_process_salt.py / data_process_umami.py / data_process_bitter.py
不同口味的数据加载脚本，提供 load_data_*(choice) 函数，用于返回 (train, val, test) 或 (train, test)。
data_augmentation.py（示例）
进行数据增广或删改操作，可根据文件名、阈值等规则处理数据。
模型脚本

CNN.py：以 CNN 结构为主的训练、测试脚本，示例性地展示如何对图像数据进行分类、输出准确率/ROC/混淆矩阵。
LSTM.py：以 LSTM 结构为主，示例性地对图像展平后的输入进行序列化处理；同样进行训练、验证、测试并输出关键指标。
GCN.py：以图神经网络 (GCN) 结构为主，对图数据 (.pt) 进行分类；支持精度评估和可视化。
ResNet.py（或在同一脚本中以 ResNet, Bottleneck 类示例）
提供 ResNet 架构示例，支持加载预训练权重并做迁移学习/微调。
每个模型脚本通常包含 train(...)、test(...) 等接口，内部通过命令行输入参数（如口味 index 和数据类型 choice）来选择对应的数据集进行处理。
辅助脚本

utils.py（示例）：可能包含如 load_state_dict、通用工具函数等。
项目入口

各脚本通常以 if __name__ == '__main__': 作为脚本入口，执行 train(...) 与 test(...)。运行脚本时，可通过控制台或脚本交互输入所需参数（口味、图数据或普通数据处理等）。
使用方法
数据准备

根据实际路径与口味，将数据放置在如 D:/code/reach/data/... 等目录下。
若是图数据(.pt)，则放置于对应 face_landmarker 目录；若是普通图像，则放置于 training_datax / testing_datax 目录，并与脚本中约定路径保持一致。
训练某一模型

以 CNN 举例（其他模型类似）：
bash
复制代码
python CNN.py
运行后，脚本会提示：
less
复制代码
0代表astringent
1代表sweet
2代表sour
...
a = input('请问需要输入那种口味的数据:')
0代表普通数据处理
1代表图数据处理
b = input('这决定使用的是图数据的处理或者是普通的数据处理')
输入例如：
css
复制代码
a=1
b=0
（表示选择 sweet, 普通图像数据处理），脚本将自动进行训练、验证，并将最佳模型权重保存在 model/ 目录下。
测试已训练模型

训练结束后，脚本会自动进入测试流程，或通过再次调用 test(...) 函数对特定口味进行测试；
测试过程中会输出准确率、绘制并弹出ROC 曲线 (AUC) 与混淆矩阵。
数据增强 / 删除

如果需要额外的数据处理（如 data_augmentation.py），可直接运行：
bash
复制代码
python data_augmentation.py
按需修改脚本中对应的路径、阈值或增强操作。
注意事项
路径硬编码
项目中部分脚本直接使用硬编码路径（如 D:/code/reach/data/...）。请根据实际数据存放位置自行修改。
GPU / CPU
脚本中通常使用 torch.device("cuda" if torch.cuda.is_available() else "cpu") 自动选择设备；若未安装 GPU，可以保持 CPU 运行，但训练速度会变慢。
依赖版本
建议使用 PyTorch >= 1.6, torchvision >= 0.7, 以及对应版本的 scikit-learn, matplotlib 等。
如有需要，可在项目根目录提供 requirements.txt 以便他人 pip install -r requirements.txt 安装。
贡献与交流
欢迎对项目提出意见或改进建议。
如果在使用过程中遇到问题，可提交 issue 或与作者联系。
