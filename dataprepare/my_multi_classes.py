# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
import glob
import os

# 参数设置
height = 256
width = 256
channels = 3  # 图像通道数
num_classes = 2  # 两个前景类别（不包括背景）

# 定义颜色到通道的映射
COLOR_MAP = {
    (128, 128, 128): 0,  # 类别1 → 通道0 (值为1)
    (0, 0, 0): 1,  # 类别2 → 通道1 (值为1)
}


def rgb_to_multichannel_mask(mask_array, color_map, num_classes):
    """将RGB掩码转换为多通道二值掩码"""
    # 初始化多通道掩码 (H, W, C)，默认所有像素为背景（所有通道为0）
    mask = np.zeros((mask_array.shape[0], mask_array.shape[1], num_classes), dtype=np.float32)

    for rgb_value, channel_idx in color_map.items():
        # 找到匹配当前颜色的像素位置
        matches = np.all(mask_array == np.array(rgb_value).reshape(1, 1, 3), axis=-1)
        # 在对应通道标记为1（其他通道保持0）
        mask[matches, channel_idx] = 1

    return mask


# 自动获取数据集数量
train_list = glob.glob("../dataset/ORIGA/images/*.png")
test_list = glob.glob("../dataset/ORIGA/images/*.png")
train_number = len(train_list)
test_number = len(test_list)

print(f"发现训练图像: {train_number} 张, 测试图像: {test_number} 张")

# 初始化数据容器
Data_train = np.zeros((train_number, height, width, channels), dtype=np.float32)
Label_train = np.zeros((train_number, height, width, num_classes), dtype=np.float32)  # 两个前景通道
Data_test = np.zeros((test_number, height, width, channels), dtype=np.float32)
Label_test = np.zeros((test_number, height, width, num_classes), dtype=np.float32)


def legacy_imresize(img, size, interp):
    """调整图像尺寸（保持原有实现）"""
    pil_img = Image.fromarray(img)
    return np.array(pil_img.resize((size[1], size[0]), {
        'nearest': Image.NEAREST,
        'bilinear': Image.BILINEAR
    }[interp]))


def process_dataset(img_list, data_container, label_container, dataset_type):
    print(f'\n正在处理 {dataset_type} 数据集:')
    for idx, img_path in enumerate(img_list):
        # 处理CT图像
        img = Image.open(img_path).convert('RGB')
        img = legacy_imresize(np.array(img), [height, width], 'bilinear')
        data_container[idx] = img.astype(np.float32)

        # 生成掩码路径
        base_name = os.path.basename(img_path)
        mask_name = os.path.splitext(base_name)[0] + ".png"
        mask_dir = os.path.dirname(img_path).replace("images", "masks")
        mask_path = os.path.join(mask_dir, mask_name)

        # 处理多分类掩码
        mask = Image.open(mask_path).convert('RGB')
        mask_array = np.array(mask)

        # 生成多通道二值掩码
        multichannel_mask = rgb_to_multichannel_mask(mask_array, COLOR_MAP, num_classes)

        # 调整尺寸（每个通道独立处理）
        resized_mask = np.zeros((height, width, num_classes), dtype=np.float32)
        for c in range(num_classes):
            channel = legacy_imresize(multichannel_mask[..., c], [height, width], 'nearest')
            resized_mask[..., c] = channel

        label_container[idx] = resized_mask

        # 进度显示
        if (idx + 1) % 10 == 0 or (idx + 1) == len(img_list):
            print(f'已处理 {idx + 1}/{len(img_list)}')


# 处理数据集
process_dataset(train_list, Data_train, Label_train, "训练集")
process_dataset(test_list, Data_test, Label_test, "测试集")

# 保存数据集
print("\n保存文件中...")
np.save('../dataset/REFUGE2/data_train.npy', Data_train)
np.save('../dataset/REFUGE2/data_test.npy', Data_test)
np.save('../dataset/REFUGE2/mask_train.npy', Label_train)
np.save('../dataset/REFUGE2/mask_test.npy', Label_test)

print("CT多分类数据集制作完成！")