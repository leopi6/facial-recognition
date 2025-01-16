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
    """
    return Image.open(path).convert('RGB')


def check_edge_index(edge_index: torch.Tensor) -> bool:
    """
    检查给定图数据的 edge_index 是否符合条件。
    (示例：当 edge_index.max() >= 468 时返回 False)
    """
    return edge_index.max() < 468


def init_process(path_list: list, lens: int, choice: str):
    """
    根据 choice 处理路径列表：
      - choice='0': 解析图像路径与标签 -> (path, label)
      - choice='1': 解析 .pt 文件并赋值 .y
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
    针对图像数据的自定义 Dataset；图数据不使用本类。
    """
    def __init__(self, data, transform, loader=Myloader):
        """
        :param data: [(path, label), ...]
        :param transform: torchvision.transforms 变换管线
        :param loader: 图像加载函数
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
    根据文件名解析标签:
    {1,2,3} -> 0, {4,5,6} -> 1, 否则 2
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


def load_data_4(choice: str):
    """
    加载 umami（鲜味）数据:
      - choice='0': 返回 (Dtr, Val, Dte)
      - choice='1': 返回 (train_data, test_data)
    """
    print('data processing...')

    # 图像数据的预处理操作
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    path_list1, path_list2 = [], []

    # =========== 训练数据部分 ===========
    for i in range(58):  # 0~57
        for concentration in range(8):
            if choice == '0':
                file_path_8 = f'D:/code/reach/data/training_datax/{i+1}/umami/8'
            else:
                file_path_8 = f'D:/code/reach/data/data_face_landmarker/{i+1}/umami/8'
            if os.path.isdir(file_path_8):
                # 8 文件夹存在
                if choice == '0':
                    image_folder1 = f'D:/code/reach/data/training_datax/{i+1}/umami/{concentration+1}'
                else:
                    image_folder1 = f'D:/code/reach/data/data_face_landmarker/{i+1}/umami/{concentration+1}'
                for filename in os.listdir(image_folder1):
                    if choice == '0' and filename.endswith('.jpg'):
                        path_list1.append(os.path.join(image_folder1, filename))
                    elif choice == '1' and filename.endswith('.pt'):
                        path_list1.append(os.path.join(image_folder1, filename))
            else:
                # 8 文件夹不存在且浓度 < 8
                if (concentration + 1) < 8:
                    if choice == '0':
                        image_folder1 = f'D:/code/reach/data/training_datax/{i+1}/umami/{concentration+1}'
                    else:
                        image_folder1 = f'D:/code/reach/data/data_face_landmarker/{i+1}/umami/{concentration+1}'
                    for filename in os.listdir(image_folder1):
                        if choice == '0' and filename.endswith('.jpg'):
                            path_list1.append(os.path.join(image_folder1, filename))
                        elif choice == '1' and filename.endswith('.pt'):
                            path_list1.append(os.path.join(image_folder1, filename))
                else:
                    break

    data1 = init_process(path_list1, len(path_list1), choice)

    # =========== 测试数据部分 ===========
    for i in range(58, 72):  # 58~71
        for concentration in range(8):
            if choice == '0':
                file_path_8 = f'D:/code/reach/data/testing_datax/{i+1}/umami/8'
            else:
                file_path_8 = f'D:/code/reach/data/data_face_landmarker/{i+1}/umami/8'
            if os.path.isdir(file_path_8):
                if choice == '0':
                    image_folder1 = f'D:/code/reach/data/testing_datax/{i+1}/umami/{concentration+1}'
                else:
                    image_folder1 = f'D:/code/reach/data/data_face_landmarker/{i+1}/umami/{concentration+1}'
                for filename in os.listdir(image_folder1):
                    if choice == '0' and filename.endswith('.jpg'):
                        path_list2.append(os.path.join(image_folder1, filename))
                    elif choice == '1' and filename.endswith('.pt'):
                        path_list2.append(os.path.join(image_folder1, filename))
            else:
                if (concentration + 1) < 8:
                    if choice == '0':
                        image_folder1 = f'D:/code/reach/data/testing_datax/{i+1}/umami/{concentration+1}'
                    else:
                        image_folder1 = f'D:/code/reach/data/data_face_landmarker/{i+1}/umami/{concentration+1}'
                    for filename in os.listdir(image_folder1):
                        if choice == '0' and filename.endswith('.jpg'):
                            path_list2.append(os.path.join(image_folder1, filename))
                        elif choice == '1' and filename.endswith('.pt'):
                            path_list2.append(os.path.join(image_folder1, filename))
                else:
                    break

    data2 = init_process(path_list2, len(path_list2), choice)
    data = data1 + data2

    # =========== 当为图数据 (choice='1') 直接拆分并返回 ===========
    if choice == '1':
        now_data = data1 + data2
        filtered_data = []
        for item in now_data:
            if item.edge_index.max() < item.num_nodes:
                filtered_data.append(item)

        label_indices = defaultdict(list)
        for idx, sample in enumerate(filtered_data):
            label = sample.y
            label_indices[label].append(idx)

        trainset, testset = [], []
        test_size = 0.2
        for label, indices in label_indices.items():
            train_indices, test_indices = train_test_split(indices, test_size=test_size)
            trainset.extend(train_indices)
            testset.extend(test_indices)

        train_data = [filtered_data[i] for i in trainset]
        test_data = [filtered_data[i] for i in testset]

        # 统计各标签数量
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

    # =========== 当为图像数据 (choice='0') -> 拆分训练/验证/测试 ===========
    train_ratio = 0.6
    val_ratio = 0.2  # 剩余 0.2 作为测试
    label_data = defaultdict(list)

    # 按标签分组
    for img, label in data:
        label_data[label].append((img, label))

    train_data, val_data, test_data = [], [], []
    for label, data_list in label_data.items():
        num_samples = len(data_list)
        num_train = int(train_ratio * num_samples)
        num_val = int(val_ratio * num_samples)
        train_data.extend(data_list[:num_train])
        val_data.extend(data_list[num_train:num_train + num_val])
        test_data.extend(data_list[num_train + num_val:])

    train_dataset = MyDataset(train_data, transform=transform, loader=Myloader)
    val_dataset = MyDataset(val_data, transform=transform, loader=Myloader)
    test_dataset = MyDataset(test_data, transform=transform, loader=Myloader)

    # 统计训练集/测试集标签数量
    label_counts = {0: 0, 1: 0, 2: 0}
    for _, label in train_dataset:
        label_counts[label] += 1
    for lbl, cnt in label_counts.items():
        print(f"训练标签 {lbl} 的数据数量为: {cnt}")

    label_counts = {0: 0, 1: 0, 2: 0}
    for _, label in test_dataset:
        label_counts[label] += 1
    for lbl, cnt in label_counts.items():
        print(f"测试标签 {lbl} 的数据数量为: {cnt}")

    # 生成 DataLoader
    Dtr = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    Val = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=0)
    Dte = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=0)
    return Dtr, Val, Dte
