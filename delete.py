import os
from random import random
import cv2
import numpy as np
from matplotlib import pyplot as plt


def data_augmentation(input_folder, output_folder):
    # 创建输出文件夹
    #os.makedirs(output_folder, exist_ok=True)

    # 获取输入文件夹中的所有文件
    #files = os.listdir(input_folder)

    for i in range(72):
        for concentration in range(8):
            file_path_8 = f'D:/code/reach/data/data_face_landmarker/{i + 1}/astringent/8'
            ##因为浓度 有部分没有8这个文件夹，所以要检测是否有8这个文件夹，分不同情况处理！！！

            if os.path.isdir(file_path_8):    # 训练目录
                image_folder1 = f'D:/code/reach/data/data_face_landmarker/{i + 1}/astringent/{concentration + 1}'

    #根据标签来判断进行几种增强，等价于增加几倍的数据**********************************************************************************************
                count = 0
                for filename in os.listdir(image_folder1):
                    if count >= 200:
                        break

                    if filename.startswith("aug") and (filename.endswith("4.pt") or filename.endswith("5.pt") or filename.endswith("6.pt")):
                        img = os.path.join(image_folder1, filename)
                        os.remove(img)
                        count += 1
            else:
                if concentration + 1 < 8:
                    count = 0
                    image_folder1 = f'D:/code/reach/data/data_face_landmarker/{i + 1}/astringent/{concentration + 1}'
                    for filename in os.listdir(image_folder1):
                        if count >= 200:
                            break
                        if filename.startswith("aug") and (filename.endswith("4.pt") or filename.endswith("5.pt") or filename.endswith("6.pt")):

                            img = os.path.join(image_folder1, filename)
                            os.remove(img)
                            count += 1
                else:
                    break



 # 循环处理每个文件


#if filename.startswith("aug") and (filename.endswith("7.jpg") or filename.endswith("8.jpg") or filename.endswith("9.jpg")):
# 输入文件夹和输出文件夹的路径
input_folder = "data"
output_folder = "D:/code/reach/data/augmented_data"
print("\n:::::::strated:::::::")
# 进行数据增强
data_augmentation(input_folder, output_folder)
print("\n:::::::succeed:::::::")