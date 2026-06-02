import torch
import numpy as np
import random
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from scipy import ndimage


class GeneralMedicalLoader(Dataset):
    def __init__(self, path_Data, mode='binary', train=True, Test=False, flag=0, img_size=256,
                 normalization_stats=None):
        """
        Args:
            path_Data (str): 数据路径
            mode (str): 'binary' (单类) 或 'multiclass' (多类)
            img_size (int): 目标图像大小 (例如 224, 256, 512)
            normalization_stats (tuple): (mean, std) 用于归一化，必须来自训练集
        """
        super(GeneralMedicalLoader, self).__init__()
        self.train = train
        self.img_size = img_size
        self.mode = mode

        # 1. 加载数据
        if train:
            self.data = np.load(path_Data + 'data_train.npy')
            self.mask = np.load(path_Data + 'mask_train.npy')
        else:
            if Test:
                if flag == 1:
                    self.data = np.load(path_Data + 'data1_test.npy')
                    self.mask = np.load(path_Data + 'mask1_test.npy')
                elif flag == 2:
                    self.data = np.load(path_Data + 'data2_test.npy')
                    self.mask = np.load(path_Data + 'mask2_test.npy')
                else:
                    self.data = np.load(path_Data + 'data_test.npy')
                    self.mask = np.load(path_Data + 'mask_test.npy')
            else:
                self.data = np.load(path_Data + 'data_val.npy')
                self.mask = np.load(path_Data + 'mask_val.npy')

        # 2. 预处理 Mask (根据模式)
        if self.mode == 'binary':
            # 单类：强制二值化 (0, 1)
            self.mask = np.where(self.mask >= 128, 1, 0).astype(np.uint8)
            # 确保维度是 (N, H, W, 1)
            if len(self.mask.shape) == 3:
                self.mask = np.expand_dims(self.mask, axis=3)

        elif self.mode == 'multiclass':
            # 多类：保持原样
            if len(self.mask.shape) == 3:
                self.mask = np.expand_dims(self.mask, axis=3)

        # 3. 调整大小 (Resize)
        # 逻辑修改：只有当原始数据的 H 或 W 与目标 img_size 不一致时才 Resize
        # self.data shape 通常是 (N, H, W, C)
        current_h, current_w = self.data.shape[1], self.data.shape[2]
        if current_h != self.img_size or current_w != self.img_size:
            print(f"Resizing data from ({current_h}, {current_w}) to ({self.img_size}, {self.img_size})...")
            self.data = self._resize_data(self.data)
            self.mask = self._resize_mask(self.mask)

        # 4. 设置归一化参数
        if normalization_stats is None:
            # 默认值
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
        else:
            self.mean = normalization_stats[0]
            self.std = normalization_stats[1]

        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)

    def _resize_data(self, data):
        """Image 使用双线性插值 (Bilinear)"""
        resized_data = []
        for i in range(len(data)):
            img = cv2.resize(data[i], (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            resized_data.append(img)
        return np.array(resized_data)

    def _resize_mask(self, mask):
        """Mask 必须使用最近邻插值 (Nearest Neighbor) 以保持类别整数值"""
        resized_mask = []
        for i in range(len(mask)):
            msk = cv2.resize(mask[i], (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            if len(msk.shape) == 2:
                msk = np.expand_dims(msk, axis=-1)
            resized_mask.append(msk)
        return np.array(resized_mask)

    def __getitem__(self, indx):
        img = self.data[indx]  # (H, W, C)
        seg = self.mask[indx]  # (H, W, C)

        # 1. 增强 (仅训练集)
        if self.train:
            img, seg = self.apply_augmentation(img, seg)

        # 2. 转换 Image
        img = torch.from_numpy(img.copy()).float().permute(2, 0, 1) / 255.0

        # 3. 转换 Mask
        if self.mode == 'binary':
            seg = torch.from_numpy(seg.copy()).float().permute(2, 0, 1)  # (1, H, W)
        else:
            seg = torch.from_numpy(seg.copy()).float().permute(2, 0, 1)  # (C, H, W)

        # 4. 归一化 Image
        img = self.normalize(img)

        return img, seg

    def apply_augmentation(self, image, label):
        if random.random() > 0.5:
            k = np.random.randint(1, 4)
            image = np.rot90(image, k)
            label = np.rot90(label, k)

        if random.random() > 0.5:
            axis = np.random.randint(0, 2)
            image = np.flip(image, axis=axis)
            label = np.flip(label, axis=axis)

        if random.random() > 0.5:
            angle = np.random.randint(-20, 20)
            image = ndimage.rotate(image, angle, order=1, reshape=False)
            label = ndimage.rotate(label, angle, order=0, reshape=False)

        return image, label

    def __len__(self):
        return len(self.data)


# ==========================================
# 如何调用 (已添加 img_size 参数)
# ==========================================

def get_datasets(path_Data, mode='binary', need_val="False", img_size=256):
    """
    Args:
        path_Data: 数据路径
        mode: 'binary' 或 'multiclass'
        img_size: 目标图片大小 (默认 256)
        need_val: 是否需要验证集 "True"/"False"
    """
    # 1. 计算训练集统计量
    print(f"Calculating stats from train data (Original Size)...")
    temp_train = np.load(path_Data + 'data_train.npy')

    # 注意：计算均值方差通常在原始尺寸做即可，不需要 Resize 后再算，结果是一样的
    temp_train = temp_train.astype(np.float32) / 255.0
    mean = np.mean(temp_train, axis=(0, 1, 2))
    std = np.std(temp_train, axis=(0, 1, 2))
    del temp_train

    stats = (mean, std)
    print(f"Stats computed. Mean: {mean}, Std: {std}")
    print(f"Loading datasets with target img_size: {img_size}")

    # 2. 创建 Dataset (传入 img_size)
    train_dataset = GeneralMedicalLoader(path_Data, mode=mode, train=True, img_size=img_size, normalization_stats=stats)
    test_dataset = GeneralMedicalLoader(path_Data, mode=mode, train=False, Test=True, img_size=img_size,
                                        normalization_stats=stats)

    if need_val == "True":
        val_dataset = GeneralMedicalLoader(path_Data, mode=mode, train=False, Test=False, img_size=img_size,
                                           normalization_stats=stats)
        return train_dataset, val_dataset, test_dataset
    else:
        return train_dataset, test_dataset

def get_test_datasets(path_Data, task_id, mode='binary', img_size=256):
    """
    Args:
        path_Data: 数据路径
        mode: 'binary' 或 'multiclass'
        img_size: 目标图片大小 (默认 256)
        need_val: 是否需要验证集 "True"/"False"
    """
    # 1. 计算训练集统计量
    print(f"Load stats...")

    # 注意：计算均值方差通常在原始尺寸做即可，不需要 Resize 后再算，结果是一样的
    # mean_list = [[0.17066666,0.17066666,0.17066666],
    #         [0.22857143,0.14945954,0.11630471],
    #         [0.08891977,0.08891977,0.08891977]]
    # std_list = [[0.4131182,0.3257771,0.31193987],
    #        [0.26245958,0.16993749,0.12710536],
    #        [0.21180855,0.21180855,0.21180855]]
    mean_list = [[0.39766145, 0.26233402, 0.17052175],
            [0.4357091, 0.28105298, 0.18593493],
            [0.6084567, 0.43629387, 0.37632418]]
    std_list = [[0.29468757, 0.19973984, 0.13595806],
           [0.3082962, 0.21928723, 0.15732406],
           [0.25860268, 0.23404145, 0.2182634]]

    mean = mean_list[task_id]
    std = std_list[task_id]
    stats = (mean, std)
    print(f"Mean: {mean}, Std: {std}")
    print(f"Loading datasets with target img_size: {img_size}")

    # 2. 创建 Dataset (传入 img_size)
    test_dataset = GeneralMedicalLoader(path_Data, mode=mode, train=False, Test=True, img_size=img_size,
                                        normalization_stats=stats)
    return test_dataset