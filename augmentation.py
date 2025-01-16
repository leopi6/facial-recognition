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

    for i in range(58,64):
        for concentration in range(8):

            ##因为浓度 有部分没有8这个文件夹，所以要检测是否有8这个文件夹，分不同情况处理！！！
            file_path_8 = f'D:/code/reach/data/testing_datax/{i + 1}/sour/8'
            if os.path.isdir(file_path_8):
                # 训练目录
                image_folder1 = f'D:/code/reach/data/testing_datax/{i + 1}/sour/{concentration + 1}'
                for filename in os.listdir(image_folder1):
                    if filename.endswith("7.jpg") or filename.endswith("8.jpg") or filename.endswith("9.jpg"):
                        img = cv2.imread(os.path.join(image_folder1, filename ))
#灰度图**********************************************************************************************
                        # 在这里添加你的数据增强方法，这里使用直方图均衡化作为示例
                        img_eq = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                        # 保存增强后的图片
                        output_file = os.path.join(image_folder1, "aug1"+filename)
                        cv2.imwrite(output_file, img_eq)
# 亮度**********************************************************************************************
                        brightness_factor = 1.5  # 亮度调整因子，可以根据需求调整
                        adjusted_img = cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)

                        # 保存增强后的图片
                        output_file = os.path.join(image_folder1, "aug3" + filename)  # 在文件名前加上 "brightness_adjusted_"
                        cv2.imwrite(output_file, adjusted_img)
#旋转**********************************************************************************************
                    ''' rows, cols, _ = img.shape
                        # 随机选择旋转角度（这里以90度为例）
                        angle = np.random.randint(-90, 90)
                        # 计算旋转矩阵
                        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
                        # 进行旋转
                        rotated_img = cv2.warpAffine(img, M, (cols, rows))
                        # 保存增强后的图片
                    #改到这！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
                        output_file = os.path.join(image_folder1, "aug4"+filename)  # 在文件名前加上 "augmented_"
                        cv2.imwrite(output_file, rotated_img)'''
#翻转**********************************************************************************************
                '''if np.random.choice([True, False]):
                        flipped_img = cv2.flip(img, 1)  # 参数1表示水平翻转
                    else:
                        flipped_img = img
                    # 随机垂直翻转
                    if np.random.choice([True, False]):
                        flipped_img = cv2.flip(flipped_img, 0)  # 参数0表示垂直翻转
                    # 保存增强后的图片
                    output_file = os.path.join(image_folder1, "aug2"+filename)
                    cv2.imwrite(output_file, flipped_img)'''

#根据标签来判断进行几种增强，等价于增加几倍的数据**********************************************************************************************

                '''if filename.endswith("7.jpg") or filename.endswith("8.jpg") or filename.endswith("9.jpg"):
                       k = random.random()
                        if k==1:
                            img = cv2.imread(os.path.join(image_folder1, filename))
                            img_eq = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                        # 保存增强后的图片
                            output_file = os.path.join(image_folder1, "aug4" + filename)
                            cv2.imwrite(output_file, img_eq)
                        else:
                            brightness_factor = 1.5  # 亮度调整因子，可以根据需求调整
                            adjusted_img = cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)

                            # 保存增强后的图片
                            output_file = os.path.join(image_folder1,"aug4" + filename)  # 在文件名前加上 "brightness_adjusted_"
                            cv2.imwrite(output_file, adjusted_img)
                        for j in range(5):

                            img = cv2.imread(os.path.join(image_folder1, filename))
                        # 灰度图**********************************************************************************************
                        # 在这里添加你的数据增强方法，这里使用直方图均衡化作为示例
                            img_eq = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                        # 保存增强后的图片
                            output_file = os.path.join(image_folder1, f"aug{3+j}" + filename)
                            cv2.imwrite(output_file, img_eq)
                        # 旋转**********************************************************************************************
                            rows, cols, _ = img.shape
                            # 随机选择旋转角度（这里以90度为例）
                            angle = np.random.randint(-90, 90)
                            # 计算旋转矩阵
                            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
                            # 进行旋转
                            rotated_img = cv2.warpAffine(img, M, (cols, rows))
                            # 保存增强后的图片
                            # 改到这！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
                            output_file = os.path.join(image_folder1,f"aug{5+j}" + filename)  # 在文件名前加上 "augmented_"
                            cv2.imwrite(output_file, rotated_img)
                            # 翻转**********************************************************************************************
                            if np.random.choice([True, False]):
                                flipped_img = cv2.flip(img, 1)  # 参数1表示水平翻转
                            else:
                                flipped_img = img
                            # 随机垂直翻转
                            if np.random.choice([True, False]):
                                flipped_img = cv2.flip(flipped_img, 0)  # 参数0表示垂直翻转
                            # 保存增强后的图片
                            output_file = os.path.join(image_folder1, f"aug{4+j}" + filename)
                            cv2.imwrite(output_file, flipped_img)
                            # 亮度**********************************************************************************************

                            brightness_factor = 1.5  # 亮度调整因子，可以根据需求调整
                            adjusted_img = cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)

                            # 保存增强后的图片
                            output_file = os.path.join(image_folder1, f"aug{5+j}" + filename)  # 在文件名前加上 "brightness_adjusted_"
                            cv2.imwrite(output_file, adjusted_img)'''
            else:
                if concentration + 1 < 8:
                    image_folder1 = f'D:/code/reach/data/testing_datax/{i + 1}/sour/{concentration + 1}'
                    for filename in os.listdir(image_folder1):
                        if filename.endswith("7.jpg") or filename.endswith("8.jpg") or filename.endswith("9.jpg"):
                            img = cv2.imread(os.path.join(image_folder1, filename))
                            # 灰度图**********************************************************************************************
                            # 在这里添加你的数据增强方法，这里使用直方图均衡化作为示例
                            img_eq = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                            # 保存增强后的图片
                            output_file = os.path.join(image_folder1, "aug1" + filename)
                            cv2.imwrite(output_file, img_eq)
                            # 旋转**********************************************************************************************
                            rows, cols, _ = img.shape
                            # 随机选择旋转角度（这里以90度为例）
                            angle = np.random.randint(-90, 90)
                            # 计算旋转矩阵
                            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
                            # 进行旋转
                            rotated_img = cv2.warpAffine(img, M, (cols, rows))
                            # 保存增强后的图片
                            # 改到这！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
                            output_file = os.path.join(image_folder1, "aug4" + filename)  # 在文件名前加上 "augmented_"
                            cv2.imwrite(output_file, rotated_img)
# 翻转**********************************************************************************************
                    ''' if np.random.choice([True, False]):
                            flipped_img = cv2.flip(img, 1)  # 参数1表示水平翻转
                        else:
                            flipped_img = img
                        # 随机垂直翻转
                        if np.random.choice([True, False]):
                            flipped_img = cv2.flip(flipped_img, 0)  # 参数0表示垂直翻转
                        # 保存增强后的图片
                        output_file = os.path.join(image_folder1, "aug2" + filename)
                        cv2.imwrite(output_file, flipped_img)
                        # 亮度**********************************************************************************************

                        brightness_factor = 1.5  # 亮度调整因子，可以根据需求调整
                        adjusted_img = cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)

                        # 保存增强后的图片
                        output_file = os.path.join(image_folder1, "aug3" + filename)  # 在文件名前加上 "brightness_adjusted_"
                        cv2.imwrite(output_file, adjusted_img)'''


 # 循环处理每个文件



# 输入文件夹和输出文件夹的路径
input_folder = "data"
output_folder = "D:/code/reach/data/augmented_data"
print("\n:::::::strated:::::::")
# 进行数据增强
data_augmentation(input_folder, output_folder)
print("\n:::::::succeed:::::::")