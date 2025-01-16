"""
@Time   : 2024/01/28 11:34
@Author : 李颢
@File   : CNN.py
@Motto  : GO UP
"""

import os
from collections import defaultdict
from random import random

import torch
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


def Myloader(path: str) -> Image.Image:
    """
    读取图像并转为 RGB 模式。
    :param path: 图像文件路径
    :return: PIL 的 Image 对象 (RGB)
    """
    return Image.open(path).convert('RGB')


def check_edge_index(edge_index: torch.Tensor) -> bool:
    """
    检查给定的图结构中 edge_index 是否符合一定条件。
    这里示例为: 如果 edge_index.max() >= 468，则返回 False，表示不符合条件。
    """
    return edge_index.max() < 468


def init_process(path_list: list, lens: int, choice: str):
    """
    根据 choice 判断处理方式:
    - choice='0': 普通图像数据 (将路径和标签存入列表)
    - choice='1': 图数据(.pt 文件)，在读取后直接给 .y 赋值标签
    
    :param path_list: 文件路径列表
    :param lens: 文件数
    :param choice: '0' or '1'
    :return: 处理好的数据列表
    """
    if choice == '0':
        data = []
        for i in range(lens):
            label = find_label(path_list[i])
            data.append([path_list[i], label])
    else:
        data = []
        for i in range(lens):
            label = find_label(path_list[i])
            da = torch.load(path_list[i])
            da.y = label
            data.append(da)
    return data


class MyDataset(Dataset):
    """
    针对图像数据的自定义 Dataset；图数据情况下不使用本类。
    """
    def __init__(self, data, transform, loader=Myloader):
        """
        :param data: [(img_path, label), ...]
        :param transform: torchvision.transforms 组成的变换管线
        :param loader: 用于读取图像的函数
        """
        self.data = data
        self.transform = transform
        self.loader = loader

    def __getitem__(self, item: int):
        img_path, label = self.data[item]
        img = self.loader(img_path)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)


def find_label(file_path: str) -> int:
    """
    根据文件路径解析标签：
    - 当文件名最后一段在 {'1','2','3'} 中时返回 0；
    - 当文件名最后一段在 {'4','5','6'} 中时返回 1；
    - 否则返回 2。
    """
    pure_path = os.path.splitext(file_path)
    f = pure_path[0].split('_')
    label_part = f[-1]

    if label_part in {'1', '2', '3'}:
        return 0
    elif label_part in {'4', '5', '6'}:
        return 1
    else:
        return 2


def load_data_5(choice: str):
    """
    加载 bitter（苦味）数据，并根据 choice 的值来决定是图像数据还是图数据。
    - choice='0': 普通图像数据 (jpg) -> 返回 DataLoader
    - choice='1': 图数据 (pt) -> 返回训练集与测试集列表
    
    :param choice: '0' or '1'
    :return:
       当 choice='1' 时, 返回 (train_data, test_data)；
       当 choice='0' 时, 返回 (Dtr, Val, Dte) (即训练、验证、测试 DataLoader)。
    """
    print('data processing...')

    # ============== 如果是图像数据，定义预处理管线 ==============
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    path_list1 = []
    path_list2 = []

    # ============== 训练数据部分 ==============
    for i in range(58):  # i: 0 ~ 57
        for concentration in range(8):
            if choice == '0':
                file_path_8 = f'D:/code/reach/data/training_datax/{i+1}/bitter/8'
            else:
                file_path_8 = f'D:/code/reach/data/data_face_landmarker/{i+1}/bitter/8'

            if os.path.isdir(file_path_8):
                # 当存在 8 文件夹时
                if choice == '0':
                    image_folder1 = f'D:/code/reach/data/training_datax/{i+1}/bitter/{concentration+1}'
                else:
                    image_folder1 = f'D:/code/reach/data/data_face_landmarker/{i+1}/bitter/{concentration+1}'

                for filename in os.listdir(image_folder1):
                    if choice == '0' and filename.endswith('.jpg'):
                        path_list1.append(os.path.join(image_folder1, filename))
                    elif choice == '1' and filename.endswith('.pt'):
                        path_list1.append(os.path.join(image_folder1, filename))
            else:
                # 当 8 文件夹不存在，但浓度<8时
                if (concentration + 1) < 8:
                    if choice == '0':
                        image_folder1 = f'D:/code/reach/data/training_datax/{i+1}/bitter/{concentration+1}'
                    else:
                        image_folder1 = f'D:/code/reach/data/data_face_landmarker/{i+1}/bitter/{concentration+1}'
                    for filename in os.listdir(image_folder1):
                        if choice == '0' and filename.endswith('.jpg'):
                            path_list1.append(os.path.join(image_folder1, filename))
                        elif choice == '1' and filename.endswith('.pt'):
                            path_list1.append(os.path.join(image_folder1, filename))
                else:
                    break

    data1 = init_process(path_list1, len(path_list1), choice)

    # ============== 测试数据部分 ==============
    for i in range(58, 72):  # i: 58 ~ 71
        for concentration in range(8):
            if choice == '0':
                file_path_8 = f'D:/code/reach/data/testing_datax/{i+1}/bitter/8'
            else:
                file_path_8 = f'D:/code/reach/data/data_face_landmarker/{i+1}/bitter/8'

            if os.path.isdir(file_path_8):
                # 存在 8 文件夹
                if choice == '0':
                    image_folder1 = f'D:/code/reach/data/testing_datax/{i+1}/bitter/{concentration+1}'
                else:
                    image_folder1 = f'D:/code/reach/data/data_face_landmarker/{i+1}/bitter/{concentration+1}'

                for filename in os.listdir(image_folder1):
                    if choice == '0' and filename.endswith('.jpg'):
                        path_list2.append(os.path.join(image_folder1, filename))
                    elif choice == '1' and filename.endswith('.pt'):
                        path_list2.append(os.path.join(image_folder1, filename))

            else:
                # 当 8 文件夹不存在，但浓度<8
                if (concentration + 1) < 8:
                    if choice == '0':
                        image_folder1 = f'D:/code/reach/data/testing_datax/{i+1}/bitter/{concentration+1}'
                    else:
                        image_folder1 = f'D:/code/reach/data/data_face_landmarker/{i+1}/bitter/{concentration+1}'

                    for filename in os.listdir(image_folder1):
                        if choice == '0' and filename.endswith('.jpg'):
                            path_list2.append(os.path.join(image_folder1, filename))
                        elif choice == '1' and filename.endswith('.pt'):
                            path_list2.append(os.path.join(image_folder1, filename))
                else:
                    break

    data2 = init_process(path_list2, len(path_list2), choice)

    # 合并训练 + 测试数据
    data = data1 + data2

    # ============== 如果是图数据 (choice='1')，直接拆分并返回 ==============
    if choice == '1':
        now_data = data1 + data2
        filtered_data = []
        for item in now_data:
            # 如果边索引的最大值 >= 节点数，说明该图不符合要求，过滤掉
            if item.edge_index.max() >= item.num_nodes:
                continue
            filtered_data.append(item)

        # 根据标签分组
        label_indices = defaultdict(list)
        for i, sample in enumerate(filtered_data):
            label = sample.y
            label_indices[label].append(i)

        trainset = []
        testset = []
        test_size = 0.2

        # 对每个标签进行 train / test 划分
        for label, indices in label_indices.items():
            train_indices, test_indices = train_test_split(indices, test_size=test_size)
            trainset.extend(train_indices)
            testset.extend(test_indices)

        train_data = [filtered_data[i] for i in trainset]
        test_data = [filtered_data[i] for i in testset]

        # 统计训练/测试标签数量
        train_label_counts = defaultdict(int)
        test_label_counts = defaultdict(int)

        for sample in train_data:
            train_label_counts[sample.y] += 1
        for sample in test_data:
            test_label_counts[sample.y] += 1

        print("训练集中每个标签的数量：")
        for label, count in train_label_counts.items():
            print(f"标签 {label} 的数量: {count}")

        print("\n测试集中每个标签的数量：")
        for label, count in test_label_counts.items():
            print(f"标签 {label} 的数量: {count}")

        return train_data, test_data

    # ============== 如果是图像数据 (choice='0')，划分训练/验证/测试 ==============
    train_ratio = 0.6
    val_ratio = 0.2  # 剩余 0.2 用于测试

    # 将数据按标签聚合
    label_data = defaultdict(list)
    for img, label in data:
        label_data[label].append((img, label))

    train_data, val_data, test_data = [], [], []

    # 对每个标签分别切分
    for label, data_list in label_data.items():
        num_samples = len(data_list)
        num_train = int(train_ratio * num_samples)
        num_val = int(val_ratio * num_samples)

        train_data.extend(data_list[:num_train])
        val_data.extend(data_list[num_train:num_train + num_val])
        test_data.extend(data_list[num_train + num_val:])

    # 组装 Dataset
    train_dataset = MyDataset(train_data, transform=transform, loader=Myloader)
    val_dataset = MyDataset(val_data, transform=transform, loader=Myloader)
    test_dataset = MyDataset(test_data, transform=transform, loader=Myloader)

    # 统计训练集中每个标签数
    label_counts = {0: 0, 1: 0, 2: 0}
    for _, label in train_dataset:
        label_counts[label] += 1
    for lbl, cnt in label_counts.items():
        print(f"训练标签 {lbl} 的数据数量: {cnt}")

    # 统计测试集中每个标签数
    label_counts = {0: 0, 1: 0, 2: 0}
    for _, label in test_dataset:
        label_counts[label] += 1
    for lbl, cnt in label_counts.items():
        print(f"测试标签 {lbl} 的数据数量: {cnt}")

    # 构建 DataLoader
    Dtr = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    Val = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=0)
    Dte = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=0)

    return Dtr, Val, Dte
