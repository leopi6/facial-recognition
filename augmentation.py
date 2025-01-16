"""
该脚本用于对指定目录下的图像数据进行多种数据增强操作（如灰度化、亮度调整、旋转等），
并将增强后的图像存储回同一个文件夹中，以便后续训练模型使用。
"""

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


def data_augmentation(input_folder: str, output_folder: str):
    """
    对指定目录下的图像进行数据增强操作。由于函数中已写死特定路径，
    这里的 input_folder 和 output_folder 在当前逻辑中并未直接使用，但仍保留接口。

    :param input_folder: 输入文件夹路径（当前函数中未直接使用）
    :param output_folder: 输出文件夹路径（当前函数中未直接使用）
    """
    # 遍历特定区间（i = 58 ~ 63）
    for i in range(58, 64):
        # 遍历浓度值 range(8)
        for concentration in range(8):
            # 针对“8”文件夹是否存在区分处理
            file_path_8 = f'D:/code/reach/data/testing_datax/{i + 1}/sour/8'
            if os.path.isdir(file_path_8):
                # 训练目录
                image_folder1 = f'D:/code/reach/data/testing_datax/{i + 1}/sour/{concentration + 1}'
                for filename in os.listdir(image_folder1):
                    # 针对后缀名为 7.jpg / 8.jpg / 9.jpg 的图片进行增强
                    if filename.endswith("7.jpg") or filename.endswith("8.jpg") or filename.endswith("9.jpg"):
                        img_path = os.path.join(image_folder1, filename)
                        img = cv2.imread(img_path)

                        # ============= 1. 灰度图 (直方图均衡化) =============
                        img_eq = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                        output_file = os.path.join(image_folder1, "aug1" + filename)
                        cv2.imwrite(output_file, img_eq)

                        # ============= 2. 亮度调整 =============
                        brightness_factor = 1.5  # 亮度调整因子
                        adjusted_img = cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)
                        output_file = os.path.join(image_folder1, "aug3" + filename)
                        cv2.imwrite(output_file, adjusted_img)

                        # ============= 3. 旋转(示例中被注释) =============
                        """
                        rows, cols, _ = img.shape
                        angle = np.random.randint(-90, 90)
                        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
                        rotated_img = cv2.warpAffine(img, M, (cols, rows))
                        output_file = os.path.join(image_folder1, "aug4" + filename)
                        cv2.imwrite(output_file, rotated_img)
                        """

                        # ============= 4. 翻转(示例中被注释) =============
                        """
                        if np.random.choice([True, False]):
                            flipped_img = cv2.flip(img, 1)  # 水平翻转
                        else:
                            flipped_img = img
                        if np.random.choice([True, False]):
                            flipped_img = cv2.flip(flipped_img, 0)  # 垂直翻转
                        output_file = os.path.join(image_folder1, "aug2" + filename)
                        cv2.imwrite(output_file, flipped_img)
                        """
            else:
                # 如果 8 文件夹不存在，但 still < 8
                if (concentration + 1) < 8:
                    image_folder1 = f'D:/code/reach/data/testing_datax/{i + 1}/sour/{concentration + 1}'
                    for filename in os.listdir(image_folder1):
                        if filename.endswith("7.jpg") or filename.endswith("8.jpg") or filename.endswith("9.jpg"):
                            img_path = os.path.join(image_folder1, filename)
                            img = cv2.imread(img_path)

                            # ============= 1. 灰度图 (直方图均衡化) =============
                            img_eq = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                            output_file = os.path.join(image_folder1, "aug1" + filename)
                            cv2.imwrite(output_file, img_eq)

                            # ============= 2. 旋转 =============
                            rows, cols, _ = img.shape
                            angle = np.random.randint(-90, 90)
                            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
                            rotated_img = cv2.warpAffine(img, M, (cols, rows))
                            output_file = os.path.join(image_folder1, "aug4" + filename)
                            cv2.imwrite(output_file, rotated_img)

                            # ============= 3. 翻转(示例中被注释) =============
                            """
                            if np.random.choice([True, False]):
                                flipped_img = cv2.flip(img, 1)
                            else:
                                flipped_img = img
                            if np.random.choice([True, False]):
                                flipped_img = cv2.flip(flipped_img, 0)
                            output_file = os.path.join(image_folder1, "aug2" + filename)
                            cv2.imwrite(output_file, flipped_img)
                            
                            # ============= 4. 亮度调整 =============
                            brightness_factor = 1.5
                            adjusted_img = cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)
                            output_file = os.path.join(image_folder1, "aug3" + filename)
                            cv2.imwrite(output_file, adjusted_img)
                            """


# ============= 主流程 =============
if __name__ == "__main__":
    input_folder = "data"
    output_folder = "D:/code/reach/data/augmented_data"

    print("\n::::::: started :::::::")
    data_augmentation(input_folder, output_folder)
    print("\n::::::: succeed :::::::")
