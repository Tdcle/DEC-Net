# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
import glob
import os

# 参数设置
height = 256  # 图像高度
width = 256  # 图像宽度
channels = 3  # 图像通道数

dataset_name = 'ETIS'
# 自动获取数据集数量
train_list = glob.glob(f"../dataset/xirou/data/{dataset_name}/train/images/*.png")
test_list = glob.glob(f"../dataset/xirou/data/{dataset_name}/val/images/*.png")
train_number = len(train_list)
test_number = len(test_list)

print(f"发现训练图像: {train_number} 张, 测试图像: {test_number} 张")

# 初始化数据容器 (保持原始数据类型)
Data_train = np.zeros((train_number, height, width, channels), dtype=np.float32)
Label_train = np.zeros((train_number, height, width), dtype=np.uint8)
Data_test = np.zeros((test_number, height, width, channels), dtype=np.float32)
Label_test = np.zeros((test_number, height, width), dtype=np.uint8)


def legacy_imresize(img, size, interp):
    """替代 scipy.misc.imresize 的功能"""
    pil_img = Image.fromarray(img.astype(np.uint8))
    return np.array(pil_img.resize((size[1], size[0]), {
        'nearest': Image.NEAREST,
        'bilinear': Image.BILINEAR,
        'bicubic': Image.BICUBIC
    }[interp]))


def process_dataset(img_list, data_container, label_container, dataset_type):
    print(f'\n正在处理 {dataset_type} 数据集:')
    for idx, img_path in enumerate(img_list):
        # 处理原始图像
        pil_img = Image.open(img_path)
        # 如果是灰度图，转换为RGB
        if pil_img.mode == 'L':
            pil_img = pil_img.convert('RGB')
        img = np.array(pil_img)
        img = legacy_imresize(img, [height, width, channels], 'bilinear')
        data_container[idx] = img.astype(np.float32)

        # 生成掩码路径
        base_name = os.path.basename(img_path)  # 获取文件名（带扩展名）
        name, ext = os.path.splitext(base_name)  # 分割文件名和扩展名

        # 新掩码文件名 = 原文件名 + 原扩展名
        mask_filename = f"{name}.png"
        mask_dir = os.path.dirname(img_path).replace("images", "masks")
        mask_path = os.path.join(mask_dir, mask_filename)

        # 处理掩码图像
        img2 = np.array(Image.open(mask_path))

        if img2.max() == 1:
            img2 = img2 * 255

        # 处理可能的多通道情况
        if img2.ndim == 3:
            img2 = img2[..., 0]  # 取第一个通道
        img2 = legacy_imresize(img2, [height, width], 'bilinear')
        # img2 = (img2 > 128).astype(np.uint8)
        label_container[idx] = img2.astype(np.float32)

        # 进度显示
        if (idx + 1) % 100 == 0 or (idx + 1) == len(img_list):
            print(f'已处理 {idx + 1}/{len(img_list)}')


# 处理训练集
process_dataset(train_list, Data_train, Label_train, "训练集")
# 处理测试集
process_dataset(test_list, Data_test, Label_test, "测试集")

# 保存数据集 (保持原始格式)
print("\n保存文件中...")
np.save(f'../dataset/xirou/data/{dataset_name}/data_train', Data_train)
np.save(f'../dataset/xirou/data/{dataset_name}/data_test', Data_test)
np.save(f'../dataset/xirou/data/{dataset_name}/mask_train', Label_train)
np.save(f'../dataset/xirou/data/{dataset_name}/mask_test', Label_test)

print("数据集制作完成！")