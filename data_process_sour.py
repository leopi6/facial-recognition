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
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


def Myloader(path: str) -> Image.Image:
    """
    读取图像并转换为 RGB 模式。
    :param path: 图像文件路径
    :return: PIL Image 对象 (RGB)
    """
    return Image.open(path).convert('RGB')


def check_edge_index(edge_index: torch.Tensor) -> bool:
    """
    检查给定图数据的 edge_index 是否符合特定条件：
    如果 edge_index.max() >= 468 则返回 False，否则 True。
    """
    return edge_index.max() < 468


def init_process(path_list: list, lens: int, choice: str):
    """
    根据 choice 判断处理方式:
      - choice='0': 将图像路径与解析出的标签组成 [path, label] 的列表
      - choice='1': 加载 .pt 文件作为图数据，为其 .y 属性赋值标签

    :param path_list: 文件路径列表
    :param lens: 文件数量
    :param choice: '0' 或 '1'
    :return: 处理后的数据列表
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
    适用于图像数据的自定义数据集类。若为图数据则不使用。
    """
    def __init__(self, data, transform, loader=Myloader):
        """
        :param data: [(img_path, label), ...]
        :param transform: 图像数据的预处理管线
        :param loader: 加载图像的函数
        """
        self.data = data
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index: int):
        img_path, label = self.data[index]
        img = self.loader(img_path)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)


def find_label(file_path: str) -> int:
    """
    根据文件名解析标签：
    文件末尾若在 {'1','2','3'} -> 标签 0
    若在 {'4','5','6'} -> 标签 1
    否则 -> 标签 2
    """
    pure_path = os.path.splitext(file_path)
    f = pure_path[0].split('_')
    label_str = f[-1]

    if label_str in {'1', '2', '3'}:
        return 0
    elif label_str in {'4', '5', '6'}:
        return 1
    else:
        return 2


def load_data_2(choice: str):
    """
    加载 sour（酸味）数据，根据 choice 选择图像数据或图数据。
      - choice='0': 加载 .jpg 并返回 DataLoader
      - choice='1': 加载 .pt 并返回训练集和测试集列表

    :param choice: '0' or '1'
    :return:
       如果 choice='1' -> (train_data, test_data)
       如果 choice='0' -> (Dtr, Val, Dte)
    """
    print('data processing...')

    # 若为图像数据，需要定义 transforms
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
                file_path_8 = f'D:/code/reach/data/training_datax/{i+1}/sour/8'
            else:
                file_path_8 = f'D:/code/reach/data/data_face_landmarker/{i+1}/sour/8'

            # 如果存在 8 文件夹
            if os.path.isdir(file_path_8):
                if choice == '0':
                    image_folder1 = f'D:/code/reach/data/training_datax/{i+1}/sour/{concentration+1}'
                else:
                    image_folder1 = f'D:/code/reach/data/data_face_landmarker/{i+1}/sour/{concentration+1}'
                for filename in os.listdir(image_folder1):
                    if choice == '0' and filename.endswith('.jpg'):
                        path_list1.append(os.path.join(image_folder1, filename))
                    elif choice == '1' and filename.endswith('.pt'):
                        path_list1.append(os.path.join(image_folder1, filename))
            else:
                # 若 8 文件夹不存在，但浓度 < 8
                if (concentration + 1) < 8:
                    if choice == '0':
                        image_folder1 = f'D:/code/reach/data/training_datax/{i+1}/sour/{concentration+1}'
                    else:
                        image_folder1 = f'D:/code/reach/data/data_face_landmarker/{i+1}/sour/{concentration+1}'
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
                file_path_8 = f'D:/code/reach/data/testing_datax/{i+1}/sour/8'
            else:
                file_path_8 = f'D:/code/reach/data/data_face_landmarker/{i+1}/sour/8'
            if os.path.isdir(file_path_8):
                if choice == '0':
                    image_folder1 = f'D:/code/reach/data/testing_datax/{i+1}/sour/{concentration+1}'
                else:
                    image_folder1 = f'D:/code/reach/data/data_face_landmarker/{i+1}/sour/{concentration+1}'
                for filename in os.listdir(image_folder1):
                    if choice == '0' and filename.endswith('.jpg'):
                        path_list2.append(os.path.join(image_folder1, filename))
                    elif choice == '1' and filename.endswith('.pt'):
                        path_list2.append(os.path.join(image_folder1, filename))
            else:
                # 若 8 文件夹不存在，但浓度 < 8
                if (concentration + 1) < 8:
                    if choice == '0':
                        image_folder1 = f'D:/code/reach/data/testing_datax/{i+1}/sour/{concentration+1}'
                    else:
                        image_folder1 = f'D:/code/reach/data/data_face_landmarker/{i+1}/sour/{concentration+1}'
                    for filename in os.listdir(image_folder1):
                        if choice == '0' and filename.endswith('.jpg'):
                            path_list2.append(os.path.join(image_folder1, filename))
                        elif choice == '1' and filename.endswith('.pt'):
                            path_list2.append(os.path.join(image_folder1, filename))
                else:
                    break

    data2 = init_process(path_list2, len(path_list2), choice)

    # 合并训练与测试数据
    data = data1 + data2

    # ============== 如果是图数据，直接返回 (train_data, test_data) ==============
    if choice == '1':
        now_data = data1 + data2
        filtered_data = []
        for item in now_data:
            # 过滤掉 edge_index.max() >= num_nodes 的数据
            if item.edge_index.max() < item.num_nodes:
                filtered_data.append(item)

        label_indices = defaultdict(list)
        for i, sample in enumerate(filtered_data):
            label = sample.y
            label_indices[label].append(i)

        trainset = []
        testset = []
        test_size = 0.2

        # 为每个标签拆分训练集 / 测试集
        for label, indices in label_indices.items():
            train_indices, test_indices = train_test_split(indices, test_size=test_size)
            trainset.extend(train_indices)
            testset.extend(test_indices)

        train_data = [filtered_data[i] for i in trainset]
        test_data = [filtered_data[i] for i in testset]

        # 打印各标签对应数量
        train_label_counts = defaultdict(int)
        test_label_counts = defaultdict(int)
        for sample in train_data:
            train_label_counts[sample.y] += 1
        for sample in test_data:
            test_label_counts[sample.y] += 1

        print("训练集中每个标签的数量：")
        for lbl, cnt in train_label_counts.items():
            print(f"标签 {lbl} 的数量: {cnt}")

        print("\n测试集中每个标签的数量：")
        for lbl, cnt in test_label_counts.items():
            print(f"标签 {lbl} 的数量: {cnt}")

        return train_data, test_data

    # ============== 如果是图像数据，继续划分训练/验证/测试集 ==============
    train_ratio = 0.6
    val_ratio = 0.2  # 剩下 0.2 用于测试

    # 基于 label 将数据分组
    label_data = defaultdict(list)
    for img, label in data:
        label_data[label].append((img, label))

    train_data, val_data, test_data = [], [], []

    # 依据比例拆分 (train/val/test)
    for label, data_list in label_data.items():
        num_samples = len(data_list)
        num_train = int(train_ratio * num_samples)
        num_val = int(val_ratio * num_samples)

        train_data.extend(data_list[:num_train])
        val_data.extend(data_list[num_train : num_train + num_val])
        test_data.extend(data_list[num_train + num_val :])

    # 构建自定义 Dataset
    train_dataset = MyDataset(train_data, transform=transform, loader=Myloader)
    val_dataset = MyDataset(val_data, transform=transform, loader=Myloader)
    test_dataset = MyDataset(test_data, transform=transform, loader=Myloader)

    # 统计训练集各标签数
    label_counts = {0: 0, 1: 0, 2: 0}
    for _, label in train_dataset:
        label_counts[label] += 1
    for lbl, cnt in label_counts.items():
        print(f"训练标签 {lbl} 的数据数量为: {cnt}")

    # 统计测试集各标签数
    label_counts = {0: 0, 1: 0, 2: 0}
    for _, label in test_dataset:
        label_counts[label] += 1
    for lbl, cnt in label_counts.items():
        print(f"测试标签 {lbl} 的数据数量为: {cnt}")

    # 构建 DataLoader
    Dtr = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    Val = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=0)
    Dte = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=0)

    return Dtr, Val, Dte
