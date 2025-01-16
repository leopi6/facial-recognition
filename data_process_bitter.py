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
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


def Myloader(path: str) -> Image.Image:
    """
    加载图像并转换为 RGB 模式。
    :param path: 图像文件路径
    :return: PIL Image 对象 (RGB)
    """
    return Image.open(path).convert('RGB')


def check_edge_index(edge_index: torch.Tensor) -> bool:
    """
    检查图数据中的边信息是否满足某些约束 (edge_index.max() < 468)。
    :param edge_index: 图数据的边信息
    :return: bool 值，True 表示可用，False 表示不可用
    """
    max_val = edge_index.max()
    return (max_val < 468)


def init_process(path_list: list, length: int, choice: str):
    """
    根据文件路径列表，将其转换为 (path, label) 或 PyTorch 图数据(.pt)。
    :param path_list: 文件路径列表
    :param length: 文件数
    :param choice: '0' 表示图像数据, '1' 表示图数据(.pt)
    :return: list, 元素要么是 [path, label], 要么是含 y 属性的图数据
    """
    if choice == '0':
        # 处理图像数据
        data = []
        for i in range(length):
            label = find_label(path_list[i])
            data.append([path_list[i], label])
    else:
        # 处理图数据
        data = []
        for i in range(length):
            label = find_label(path_list[i])
            da = torch.load(path_list[i])  # 加载 .pt 文件
            da.y = label
            data.append(da)
    return data


class MyDataset(Dataset):
    """
    对图像文件进行自定义 Dataset 封装。
    """
    def __init__(self, data: list, transform: transforms.Compose, loader=Myloader):
        """
        :param data: [(path, label), ...]
        :param transform: torchvision.transforms 的预处理流程
        :param loader: 加载图像函数，默认使用 Myloader
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
    根据文件路径推断其标签。以文件名最后的 '_数字' 作为标签依据。
    :param file_path: 文件路径
    :return: 标签 0 / 1 / 2
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


def load_data_5(choice: str):
    """
    加载 bitter（苦味）相关的数据，分为训练集 + 测试集，
    若 choice='0' 则返回图像 DataLoader(Dtr, Val, Dte)，若 choice='1' 则返回图数据形式(train_data, test_data)。

    :param choice: '0' 表示图像数据(.jpg)，'1' 表示图数据(.pt)
    :return:
        - 如果 choice='1'，返回 train_data, test_data
        - 否则，返回 Dtr, Val, Dte (三个 DataLoader)
    """
    print('data processing...')

    # ======== 图像预处理 (仅当 choice='0' 时使用) ========
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # normalization
    ])

    path_list1 = []
    path_list2 = []

    # ========== 训练集 (i in [0..57]) ==========
    for i in range(58):
        for concentration in range(8):
            # 判断 bitter/8 文件夹是否存在
            if choice == '0':
                file_path_8 = f'D:/code/reach/data/training_datax/{i+1}/bitter/8'
            else:
                file_path_8 = f'D:/code/reach/data/data_face_landmarker/{i+1}/bitter/8'

            if os.path.isdir(file_path_8):
                # 若该路径存在，说明存在 [1..8] 各浓度
                if choice == '0':
                    image_folder1 = f'D:/code/reach/data/training_datax/{i+1}/bitter/{concentration+1}'
                else:
                    image_folder1 = f'D:/code/reach/data/data_face_landmarker/{i+1}/bitter/{concentration+1}'
                for filename in os.listdir(image_folder1):
                    if choice == '0':
                        if filename.endswith('.jpg'):
                            path_list1.append(os.path.join(image_folder1, filename))
                    else:
                        if filename.endswith('.pt'):
                            path_list1.append(os.path.join(image_folder1, filename))
            else:
                # 若文件夹不存在，则说明该浓度不足 8，仅遍历 [1..7]
                if (concentration + 1) < 8:
                    if choice == '0':
                        image_folder1 = f'D:/code/reach/data/training_datax/{i+1}/bitter/{concentration+1}'
                    else:
                        image_folder1 = f'D:/code/reach/data/data_face_landmarker/{i+1}/bitter/{concentration+1}'
                    if os.path.isdir(image_folder1):
                        for filename in os.listdir(image_folder1):
                            if choice == '0':
                                if filename.endswith('.jpg'):
                                    path_list1.append(os.path.join(image_folder1, filename))
                            else:
                                if filename.endswith('.pt'):
                                    path_list1.append(os.path.join(image_folder1, filename))
                else:
                    break

    # ========== 测试集 (i in [58..71]) ==========
    for i in range(58, 72):
        for concentration in range(8):
            if choice == '0':
                file_path_8 = f'D:/code/reach/data/testing_datax/{i+1}/bitter/8'
            else:
                file_path_8 = f'D:/code/reach/data/data_face_landmarker/{i+1}/bitter/8'

            if os.path.isdir(file_path_8):
                if choice == '0':
                    image_folder1 = f'D:/code/reach/data/testing_datax/{i+1}/bitter/{concentration+1}'
                else:
                    image_folder1 = f'D:/code/reach/data/data_face_landmarker/{i+1}/bitter/{concentration+1}'
                if os.path.isdir(image_folder1):
                    for filename in os.listdir(image_folder1):
                        if choice == '0':
                            if filename.endswith('.jpg'):
                                path_list2.append(os.path.join(image_folder1, filename))
                        else:
                            if filename.endswith('.pt'):
                                path_list2.append(os.path.join(image_folder1, filename))
            else:
                if (concentration + 1) < 8:
                    if choice == '0':
                        image_folder1 = f'D:/code/reach/data/testing_datax/{i+1}/bitter/{concentration+1}'
                    else:
                        image_folder1 = f'D:/code/reach/data/data_face_landmarker/{i+1}/bitter/{concentration+1}'
                    if os.path.isdir(image_folder1):
                        for filename in os.listdir(image_folder1):
                            if choice == '0':
                                if filename.endswith('.jpg'):
                                    path_list2.append(os.path.join(image_folder1, filename))
                            else:
                                if filename.endswith('.pt'):
                                    path_list2.append(os.path.join(image_folder1, filename))
                else:
                    break

    # ========== 将路径列表转换为带标签的数据 ==========
    data1 = init_process(path_list1, len(path_list1), choice)
    data2 = init_process(path_list2, len(path_list2), choice)
    data = data1 + data2

    # ========== 如果是图数据，直接返回 train_data, test_data ==========
    if choice == '1':
        now_data = data  # data1 + data2
        filtered_data = []
        for item in now_data:
            # 如果最高边索引 >= 节点总数，则跳过该图
            if item.edge_index.max() >= item.num_nodes:
                continue
            filtered_data.append(item)

        # 按标签分类
        label_indices = defaultdict(list)
        for i, sample in enumerate(filtered_data):
            if isinstance(sample, tuple):
                label = sample[1]
            else:
                label = sample.y
            label_indices[label].append(i)

        # 按标签
