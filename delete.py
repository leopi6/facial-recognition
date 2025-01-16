import os
from random import random
import cv2
import numpy as np
from matplotlib import pyplot as plt


def data_augmentation(input_folder: str, output_folder: str):
    """
    删除数据集中指定数量的以 'aug' 开头、后缀带 '4.pt' 或 '5.pt' 或 '6.pt' 的文件。
    （以减少数据量或进行数据清理）
    
    :param input_folder: 输入文件夹（当前逻辑中未直接使用）
    :param output_folder: 输出文件夹（当前逻辑中未直接使用）
    """
    for i in range(72):  # 共有 72 个目录
        for concentration in range(8):  # 8 个浓度
            # 检测 '8' 文件夹是否存在，以区分处理
            file_path_8 = f'D:/code/reach/data/data_face_landmarker/{i + 1}/astringent/8'
            if os.path.isdir(file_path_8):
                # 训练目录
                image_folder1 = f'D:/code/reach/data/data_face_landmarker/{i + 1}/astringent/{concentration + 1}'
                count = 0
                for filename in os.listdir(image_folder1):
                    # 若处理了 200 个文件，则停止
                    if count >= 200:
                        break

                    # 若文件符合指定命名规则，删除
                    if filename.startswith("aug") and (
                        filename.endswith("4.pt") or
                        filename.endswith("5.pt") or
                        filename.endswith("6.pt")
                    ):
                        file_to_remove = os.path.join(image_folder1, filename)
                        os.remove(file_to_remove)
                        count += 1
            else:
                # 若 8 文件夹不存在，但浓度 < 8
                if (concentration + 1) < 8:
                    image_folder1 = f'D:/code/reach/data/data_face_landmarker/{i + 1}/astringent/{concentration + 1}'
                    count = 0
                    for filename in os.listdir(image_folder1):
                        if count >= 200:
                            break

                        if filename.startswith("aug") and (
                            filename.endswith("4.pt") or
                            filename.endswith("5.pt") or
                            filename.endswith("6.pt")
                        ):
                            file_to_remove = os.path.join(image_folder1, filename)
                            os.remove(file_to_remove)
                            count += 1
                else:
                    # 当浓度 >= 8 且文件夹不存在，跳过
                    break


if __name__ == "__main__":
    # 输入文件夹和输出文件夹的路径（本代码逻辑中并未直接使用这两个参数）
    input_folder = "data"
    output_folder = "D:/code/reach/data/augmented_data"

    print("\n::::::: started :::::::")
    data_augmentation(input_folder, output_folder)
    print("\n::::::: succeed :::::::")
