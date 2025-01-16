"""
@Time ： 2024/01/28 11:34
@Author ：李颢
@File ：CNN.py
@Motto：GO UP

"""
from collections import defaultdict

import torch
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os

def Myloader(path):
    return Image.open(path).convert('RGB')

# get a list of paths and labels.

def check_edge_index(edge_index):
    max = edge_index.max()    
    if  max >= 468:

        return False
    else :
        return True

# 在使用时调用这个函数并提供相应的参数

def init_process(path_list,lens,choice):
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
    def __init__(self, data, transform, loader):
        self.data = data
        self.transform = transform
        self.loader = loader

    def __getitem__(self, item):
        img, label = self.data[item]
        img = self.loader(img)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)


def find_label(file_path):
    """
    Find image tags based on file paths.

    :param str: file path
    :return: image label
    """

    pure_path = os.path.splitext(file_path)
    f = pure_path[0]
    f= f.split('_')
    label = f[-1]
  
    if label in {'1','2','3'}:
        label=0
    elif label in {'4','5','6'}:
        label=1
    else:label=2
    return int(label)




def load_data_0(choice):
    print('data processing...')
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # normalization
    ])
    
    #这里一定要改好！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    data = []
    name_list1=[]
    name_list2=[]
    path_list1 = []
    path_list2 = []
    for i in range(10):
        for concentration in range(8):
            
            ##因为浓度 有部分没有8这个文件夹，所以要检测是否有8这个文件夹，分不同情况处理！！！
            if choice == '0':
                file_path_8 = f'D:/code/reach/data/training_datax/{i+1}/astringent/8'
            else:
                file_path_8 = f'D:/code/reach/data/data_face_landmarker/{i+1}/astringent/8'
    
            if os.path.isdir(file_path_8):
                #训练目录
                if choice == '0':    
                    image_folder1 = f'D:/code/reach/data/training_datax/{i+1}/astringent/{concentration+1}'
                else:
                    image_folder1 = f'D:/code/reach/data/data_face_landmarker/{i+1}/astringent/{concentration+1}'
                for filename in os.listdir(image_folder1):
                    if choice == '0':
                        if filename.endswith('.jpg'):
                            name_list1.append(filename)
                            path = os.path.join(image_folder1,filename)
                            path_list1.append(path)
                    else:
                        if filename.endswith('.pt'):
                            name_list1.append(filename)
                            path = os.path.join(image_folder1,filename)
                            path_list1.append(path)

                
            else:
                if concentration+1 < 8:
                    #训练目录
                    if choice == '0':    
                        image_folder1 = f'D:/code/reach/data/training_datax/{i+1}/astringent/{concentration+1}'
                    else:
                        image_folder1 = f'D:/code/reach/data/data_face_landmarker/{i+1}/astringent/{concentration+1}'
                    for filename in os.listdir(image_folder1):
                        if choice == '0':
                            if filename.endswith('.jpg'):
                                name_list1.append(filename)
                                path = os.path.join(image_folder1,filename)
                                path_list1.append(path)
                        else:
                            if filename.endswith('.pt'):
                                name_list1.append(filename)
                                path = os.path.join(image_folder1,filename)
                                path_list1.append(path)
                else:
                    break
    
    data1 = init_process(path_list1, len(path_list1),choice)

    for i in range(58,59):
        for concentration in range(8):
            if choice == '0':
                file_path_8 = f'D:/code/reach/data/testing_datax/{i+1}/astringent/8'
            else:
                file_path_8 = f'D:/code/reach/data/data_face_landmarker/{i+1}/astringent/8'
            if os.path.isdir(file_path_8):

               #测试目录   
                if choice == '0':    
                    image_folder1 = f'D:/code/reach/data/testing_datax/{i+1}/astringent/{concentration+1}'
                else:
                    image_folder1 = f'D:/code/reach/data/data_face_landmarker/{i+1}/astringent/{concentration+1}'
                for filename in os.listdir(image_folder1):
                    if choice == '0':
                        if filename.endswith('.jpg'):
                            name_list2.append(filename)
                            path = os.path.join(image_folder1,filename)
                            path_list2.append(path)
                    else:
                        if filename.endswith('.pt'):
                            name_list2.append(filename)
                            path = os.path.join(image_folder1,filename)
                            path_list2.append(path)
                        
            else:
                if concentration+1 < 8:
                    #测试目录
                    if choice == '0':    
                        image_folder1 = f'D:/code/reach/data/testing_datax/{i+1}/astringent/{concentration+1}'
                    else:
                        image_folder1 = f'D:/code/reach/data/data_face_landmarker/{i+1}/astringent/{concentration+1}'
                    for filename in os.listdir(image_folder1):
                        if choice == '0':
                            if filename.endswith('.jpg'):
                                name_list2.append(filename)
                                path = os.path.join(image_folder1,filename)
                                path_list2.append(path)
                    else:
                        if filename.endswith('.pt'):
                            name_list2.append(filename)
                            path = os.path.join(image_folder1,filename)
                            path_list2.append(path)
                else:
                    break  
    
    data2 = init_process(path_list2, len(path_list2),choice)
    
               

    data= data1 + data2




#若为图数据则直接返回
    if choice == '1':
        now_data = data1 + data2  # + data3 + data4   # 1400
        data = []
        for item in now_data:
            if item.edge_index.max() >= item.num_nodes:
                continue
            data.append(item)

        '''for item in now_data:
            if isinstance(item, tuple):  # 如果数据是图像数据
                data.append(item)
            elif check_edge_index(item.edge_index):  # 如果数据是图数据
                data.append(item)'''

        label_indices = defaultdict(list)
        for i, sample in enumerate(data):
            if isinstance(sample, tuple):  # 如果数据是图像数据
                label = sample[1]
            else:  # 如果数据是图数据
                label = sample.y
            label_indices[label].append(i)

        trainset = []
        testset = []
        test_size = 0.2

        for label, indices in label_indices.items():
            train_indices, test_indices = train_test_split(indices, test_size=test_size)
            trainset.extend(train_indices)
            testset.extend(test_indices)

        train_data = [data[i] for i in trainset]
        test_data = [data[i] for i in testset]

        train_label_counts = defaultdict(int)
        test_label_counts = defaultdict(int)

        for sample in train_data:
            if isinstance(sample, tuple):
                label = sample[1]
            else:
                label = sample.y
            train_label_counts[label] += 1

        for sample in test_data:
            if isinstance(sample, tuple):
                label = sample[1]
            else:
                label = sample.y
            test_label_counts[label] += 1

        print("训练集中每个标签的数量：")
        for label, count in train_label_counts.items():
            print(f"标签 {label} 的数量: {count}")

        print("\n测试集中每个标签的数量：")
        for label, count in test_label_counts.items():
            print(f"标签 {label} 的数量: {count}")

        return train_data, test_data

        return train_data, test_data

    # 训练集和测试集的划分比例
    train_ratio = 0.6  # 训练集比例
    val_ratio = 0.2  # 验证集比例，测试集比例为 1 - train_ratio - val_ratio

    # 获取每个标签的数据
    label_data = defaultdict(list)
    for img, label in data:
        label_data[label].append((img, label))

    # 初始化训练集、验证集和测试集
    train_data = []
    val_data = []
    test_data = []

    # 将每个标签的数据划分为训练集、验证集和测试集
    for label, data_list in label_data.items():
        num_samples = len(data_list)
        num_train = int(train_ratio * num_samples)
        num_val = int(val_ratio * num_samples)

        train_data.extend(data_list[:num_train])
        val_data.extend(data_list[num_train:num_train + num_val])
        test_data.extend(data_list[num_train + num_val:])

    # 将训练集、验证集和测试集转换为 DataLoader 对象
    train_dataset = MyDataset(train_data, transform=transform, loader=Myloader)
    val_dataset = MyDataset(val_data, transform=transform, loader=Myloader)
    test_dataset = MyDataset(test_data, transform=transform, loader=Myloader)
    label_counts = {0: 0, 1: 0, 2: 0}

    # 统计每个标签的数据数量
    for img, label in train_dataset:
        label_counts[label] += 1

    # 打印结果
    for label, count in label_counts.items():
        print(f"训练标签 {label} 的数据数量为: {count}")

    label_counts = {0: 0, 1: 0, 2: 0}
    for img, label in test_dataset:
        label_counts[label] += 1

    # 打印结果
    for label, count in label_counts.items():
        print(f"测试集标签 {label} 的数据数量为: {count}")

    Dtr = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=0)
    Val = DataLoader(dataset=val_dataset, batch_size=32, shuffle=True, num_workers=0)
    Dte = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True, num_workers=0)

    return Dtr, Val, Dte
