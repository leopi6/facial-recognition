"""
@Time   : 2024/01/28 11:34
@Author : 李颢
@File   : CNN.py
@Motto  : GO UP
"""

import os
from collections import defaultdict

import numpy as np
import torch
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
    检查给定图数据的 edge_index 是否符合某些条件 (例如 max < 468)。
    :param edge_index: 图的边信息
    :return: bool 值，True 表示可用，False 表示不可用
    """
    max_value = edge_index.max()
    return (max_value < 468)


def find_label(file_path: str) -> int:
    """
    根据文件路径来获取对应的图像标签。这里根据文件名末段来解析标签。
    :param file_path: 文件路径
    :return: int 型标签 0 / 1 / 2
    """
    pure_path = os.path.splitext(file_path)
    f = pure_path[0]
    f = f.split('_')
    label = f[-1]  # 提取最后一段作为标签标识

    if label in {'1', '2', '3'}:
        label = 0
    elif label in {'4', '5', '6'}:
        label = 1
    else:
        label = 2
    return int(label)


def init_process(path_list: list, lens: int, choice: str):
    """
    初始化处理数据，将路径列表转换为 (路径, label) 或 PyTorch 图数据。
    :param path_list: 文件路径列表
    :param lens: 文件数
    :param choice: '0' 代表图像数据处理, '1' 代表图数据(.pt)处理
    :return: list, 元素要么是 (path, label) 二元组，要么是带 label 的图数据
    """
    if choice == '0':
        # 处理图像数据
        data = []
        for i in range(lens):
            label = find_label(path_list[i])
            data.append([path_list[i], label])
    else:
        # 处理图数据
        data = []
        for i in range(lens):
            label = find_label(path_list[i])
            da = torch.load(path_list[i])
            da.y = label
            data.append(da)
    return data


class MyDataset(Dataset):
    """
    针对图像数据的自定义 Dataset，实现图像加载与预处理等功能。
    """
    def __init__(self, data: list, transform: transforms.Compose, loader=Myloader):
        """
        :param data: [(img_path, label), ...]
        :param transform: torchvision.transforms 的预处理管线
        :param loader: 加载图像的函数，默认 Myloader
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


def load_data_0(choice: str):
    """
    加载 astringent（涩味）数据，根据是否是图数据 (choice='1') 或图像数据 (choice='0') 
    分别进行处理，并将其划分为训练集、验证集和测试集。
    
    :param choice: '0' 代表普通图像数据, '1' 代表图数据(.pt)
    :return:
        当 choice='1' (图数据)，返回 train_data, test_data
        否则返回 Dtr, Val, Dte (PyTorch DataLoader)
    """
    print('data processing...')

    # 图像数据的预处理（若 choice='0'）
    transform_img = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    data1 = []
    data2 = []
    path_list1 = []
    path_list2 = []

    # ========== 训练数据部分 ==========
    for i in range(10):  # i=0~9
        for concentration in range(8):
            if choice == '0':
                file_path_8 = f'D:/code/reach/data/training_datax/{i+1}/astringent/8'
            else:
                file_path_8 = f
