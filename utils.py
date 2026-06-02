import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
import numpy as np
import os
import math
import random
import logging
import logging.handlers
from matplotlib import pyplot as plt


def set_seed(seed):
    # 1. 固定Python随机数
    random.seed(seed)
    # 2. 固定NumPy随机数
    np.random.seed(seed)
    # 3. 固定PyTorch CPU随机数
    torch.manual_seed(seed)
    # 4. 固定PyTorch GPU随机数
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 5. 强制CUDA算子确定性
    os.environ['PYTORCH_CUDNN_DETERMINISTIC'] = '1'
    os.environ['PYTORCH_CUDNN_BENCHMARK'] = '0'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 6. 禁用TF32（NVIDIA GPU的低精度优化）
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # 7. 强制所有PyTorch算子确定性（忽略不支持的警告）
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:
        print(f"Warning: {e}")


def get_logger(name, log_dir):
    '''
    Args:
        name(str): name of logger
        log_dir(str): path of log
    '''

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    info_name = os.path.join(log_dir, '{}.info.log'.format(name))
    info_handler = logging.handlers.TimedRotatingFileHandler(info_name,
                                                             when='D',
                                                             encoding='utf-8')
    info_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    info_handler.setFormatter(formatter)

    logger.addHandler(info_handler)

    return logger


def log_config_info(config, logger):
    config_dict = config.__dict__
    log_info = f'#----------Config info----------#'
    logger.info(log_info)
    for k, v in config_dict.items():
        if k[0] == '_':
            continue
        else:
            log_info = f'{k}: {v},'
            logger.info(log_info)



def get_optimizer(config, model):
    assert config.opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'], 'Unsupported optimizer!'

    if config.opt == 'Adadelta':
        return torch.optim.Adadelta(
            model.parameters(),
            lr = config.lr,
            rho = config.rho,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Adagrad':
        return torch.optim.Adagrad(
            model.parameters(),
            lr = config.lr,
            lr_decay = config.lr_decay,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Adam':
        return torch.optim.Adam(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay,
            amsgrad = config.amsgrad
        )
    elif config.opt == 'AdamW':
        return torch.optim.AdamW(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay,
            amsgrad = config.amsgrad
        )
    elif config.opt == 'Adamax':
        return torch.optim.Adamax(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'ASGD':
        return torch.optim.ASGD(
            model.parameters(),
            lr = config.lr,
            lambd = config.lambd,
            alpha  = config.alpha,
            t0 = config.t0,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'RMSprop':
        return torch.optim.RMSprop(
            model.parameters(),
            lr = config.lr,
            momentum = config.momentum,
            alpha = config.alpha,
            eps = config.eps,
            centered = config.centered,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Rprop':
        return torch.optim.Rprop(
            model.parameters(),
            lr = config.lr,
            etas = config.etas,
            step_sizes = config.step_sizes,
        )
    elif config.opt == 'SGD':
        return torch.optim.SGD(
            model.parameters(),
            lr = config.lr,
            momentum = config.momentum,
            weight_decay = config.weight_decay,
            dampening = config.dampening,
            nesterov = config.nesterov
        )
    else: # default opt is SGD
        return torch.optim.SGD(
            model.parameters(),
            lr = 0.01,
            momentum = 0.9,
            weight_decay = 0.05,
        )



def get_scheduler(config, optimizer):
    assert config.sch in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau',
                        'CosineAnnealingWarmRestarts', 'WP_MultiStepLR', 'WP_CosineLR'], 'Unsupported scheduler!'
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=5)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max - 5, eta_min=1e-5, last_epoch = config.last_epoch)
    if config.sch == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size = config.step_size,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones = config.milestones,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'CosineAnnealingLR':
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer,
        #     T_max = config.T_max,
        #     eta_min = config.eta_min,
        #     last_epoch = config.last_epoch
        # )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[5]  # 第 5 个 epoch 时从 warmup 切换到 cosine
        )
    elif config.sch == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode = config.mode,
            factor = config.factor,
            patience = config.patience,
            threshold = config.threshold,
            threshold_mode = config.threshold_mode,
            cooldown = config.cooldown,
            min_lr = config.min_lr,
            eps = config.eps
        )
    elif config.sch == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0 = config.T_0,
            T_mult = config.T_mult,
            eta_min = config.eta_min,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'WP_MultiStepLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else config.gamma**len(
                [m for m in config.milestones if m <= epoch])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    elif config.sch == 'WP_CosineLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else 0.5 * (
                math.cos((epoch - config.warm_up_epochs) / (config.epochs - config.warm_up_epochs) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    return scheduler


import matplotlib

matplotlib.use('Agg')  # 非交互模式
import matplotlib.pyplot as plt


def save_imgs(img, msk, msk_pred, batch_idx, save_dir, dataset_name, threshold=0.5, test_data_name=None):
    """
    批量保存可视化结果 (适配numpy输入)
    参数：
        img: numpy数组 [B, H, W, C]
        msk: numpy数组 [B, H, W]
        msk_pred: numpy数组 [B, H, W]
        batch_idx: 当前批次索引
        save_dir: 保存路径
    """
    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 遍历批次中的每个样本
    for sample_idx in range(img.shape[0]):
        # --- 处理当前样本 ---
        # 图像处理
        curr_img = img[sample_idx]
        if curr_img.max() > 1.0:
            curr_img = curr_img / 255.0  # 归一化到 [0,1]

        # 标签处理
        curr_gt = msk[sample_idx]
        curr_pred = msk_pred[sample_idx]

        # 根据数据集类型处理
        if dataset_name == 'retinal':
            curr_gt = np.squeeze(curr_gt)
            curr_pred = np.squeeze(curr_pred)
        else:
            curr_gt = (curr_gt > 0.5).astype(np.uint8)
            curr_pred = (curr_pred > threshold).astype(np.uint8)

        # --- 创建可视化布局 ---
        fig = plt.figure(figsize=(18, 6))

        # 显示原图
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(curr_img)
        ax1.set_title('Input Image')
        ax1.axis('off')

        # 显示真实标签
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.imshow(curr_gt, cmap='gray')
        ax2.set_title('Ground Truth')
        ax2.axis('off')

        # 显示预测结果
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.imshow(curr_pred, cmap='gray')
        ax3.set_title(f'Prediction (Threshold={threshold})')
        ax3.axis('off')

        # --- 保存文件 ---
        filename = f"batch{batch_idx:03d}_sample{sample_idx:02d}"
        if test_data_name:
            filename = f"{test_data_name}_{filename}"

        plt.savefig(
            os.path.join(save_dir, f"{filename}.png"),
            bbox_inches='tight',
            dpi=150
        )
        plt.close(fig)

def save_imgs_multi(img, msk, msk_pred, batch_idx, save_dir, dataset_name, threshold=0.5,
                    test_data_name=None):
    """
    视盘/视杯分割专用可视化函数
    参数说明：
        img: 原始图像 [B, C, H, W]
        msk: 真实标签 [B, 1, H, W] (0=背景, 1=视盘, 2=视杯)
        msk_pred: 模型预测 [B, 2, H, W] (通道0=视盘概率, 通道1=视杯概率)
    """
    # 定义视盘/视杯颜色映射
    COLOR_MAP = {
        (0, 0, 0): 0,  # 背景 - 黑色
        (128, 128, 128): 1,  # 视盘 - 灰色
        (255, 255, 255): 2  # 视杯 - 白色
    }

    # 生成颜色数组
    color_list = sorted(COLOR_MAP.items(), key=lambda x: x[1])
    color_array = np.array([k for k, v in color_list], dtype=np.uint8)

    os.makedirs(save_dir, exist_ok=True)

    # --- 原始图像处理 ---
    curr_img = img
    if curr_img.max() > 1.0:
        curr_img = (curr_img / 255.0).astype(np.float32)
    # 转换为HWC格式用于显示
    if curr_img.shape[0] == 3:  # CHW → HWC
        curr_img = np.transpose(curr_img, (1, 2, 0))
    # --- 真实标签处理 ---
    # curr_gt = msk[sample_idx, 0]  # 单通道标签 [H, W]
    curr_gt = msk  # 单通道标签 [H, W]
    gt_rgb = np.zeros((*curr_gt.shape, 3), dtype=np.uint8)
    # 根据标签值着色
    gt_rgb[curr_gt == 0] = color_array[0]  # 背景
    gt_rgb[curr_gt == 1] = color_array[1]  # 视盘
    gt_rgb[curr_gt == 2] = color_array[2]  # 视杯
    # --- 预测结果处理 ---
    # 获取视盘和视杯预测
    disc_pred = msk_pred[0]  # 视盘概率 [H, W]
    cup_pred = msk_pred[1]  # 视杯概率 [H, W]
    # 创建RGB预测图
    pred_rgb = np.zeros((*disc_pred.shape, 3), dtype=np.uint8)
    # 应用阈值生成二值掩码
    disc_mask = disc_pred > threshold
    cup_mask = cup_pred > threshold
    # 着色规则：视杯覆盖视盘
    pred_rgb[disc_mask] = color_array[1]  # 视盘区域
    pred_rgb[cup_mask] = color_array[2]  # 视杯区域（覆盖视盘）
    # --- 可视化布局 ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # 输入图像
    axes[0].imshow(curr_img, cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    # 真实标签
    axes[1].imshow(gt_rgb)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    # 预测结果
    axes[2].imshow(pred_rgb)
    axes[2].set_title(f'Prediction (Threshold={threshold})')
    axes[2].axis('off')
    # --- 保存文件 ---
    filename = f"batch{batch_idx:03d}"
    if test_data_name:
        filename = f"{test_data_name}_{filename}"
    plt.savefig(
        os.path.join(save_dir, f"{filename}.png"),
        bbox_inches='tight',
        dpi=150  # 提高分辨率
    )
    plt.close(fig)


class CrossEntropyLoss(nn.Module):
    """多分类交叉熵损失"""

    def __init__(self, weight=None, ignore_index=-100):
        super(CrossEntropyLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) - logits，不需要softmax
            target: (B, H, W) - 类别索引，范围[0, C-1]，long类型
        """
        return self.ce_loss(pred, target)


class DiceLoss(nn.Module):
    """多分类Dice损失"""

    def __init__(self, smooth=1.0, ignore_background=False):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_background = ignore_background

    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) - softmax概率或logits
            target: (B, H, W) - 类别索引，long类型
        """
        pred = pred.float()
        # 如果pred是logits，先转为概率
        if pred.dim() == 4:
            pred = F.softmax(pred, dim=1)

        pred = torch.clamp(pred, 1e-7, 1 - 1e-7)

        num_classes = pred.size(1)
        batch_size = pred.size(0)

        # 将target转为one-hot编码: (B, H, W) -> (B, C, H, W)
        target_one_hot = F.one_hot(target, num_classes=num_classes)  # (B, H, W, C)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # Flatten
        pred = pred.view(batch_size, num_classes, -1)  # (B, C, H*W)
        target_one_hot = target_one_hot.view(batch_size, num_classes, -1)  # (B, C, H*W)

        # 计算每个类别的Dice
        intersection = (pred * target_one_hot).sum(2)  # (B, C)
        union = pred.sum(2) + target_one_hot.sum(2)  # (B, C)

        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)  # (B, C)

        # 是否忽略背景类（索引0）
        if self.ignore_background:
            dice_score = dice_score[:, 1:]  # 去掉背景类

        # 平均所有类别和batch
        dice_loss = 1 - dice_score.mean()

        return dice_loss


class BceDiceLoss(nn.Module):
    """交叉熵 + Dice 组合损失"""

    def __init__(self, wce=1.0, wdice=1.0, smooth=1.0,
                 ignore_background=True, class_weights=None):
        super(BceDiceLoss, self).__init__()
        self.ce = CrossEntropyLoss(weight=class_weights)
        self.dice = DiceLoss(smooth=smooth, ignore_background=ignore_background)
        self.wce = wce
        self.wdice = wdice

    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) - logits
            target: (B, 1, H, W) 或 (B, H, W) - 类别索引，值为0(背景), 1(前景1), 2(前景2)...
        """
        # 步骤1: 处理target维度 (B, 1, H, W) -> (B, H, W)
        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)  # 去掉通道维度

        # 步骤2: 确保target是long类型（类别索引必须是整数）
        target = target.long()

        # 步骤3: 验证target的值范围（可选，debug时使用）
        num_classes = pred.size(1)
        assert target.min() >= 0 and target.max() < num_classes, \
            f"Target values must be in [0, {num_classes - 1}], but got [{target.min()}, {target.max()}]"

        # 步骤4: 计算损失（内部会自动转换为one-hot）
        ce_loss = self.ce(pred, target)  # CrossEntropy内部会处理
        dice_loss = self.dice(pred, target)  # Dice内部会转为one-hot

        loss = self.wce * ce_loss + self.wdice * dice_loss
        # loss = self.wdice * dice_loss
        return loss

# 新增MAE损失函数
class MAELoss(nn.Module):
    """
    MAE专用损失函数，仅计算掩码区域的像素级重建损失
    """

    def __init__(self, norm_pix=False):
        super().__init__()
        self.norm_pix = norm_pix

    def forward(self, pred, target, mask):
        """
        pred: 重建图像 [B, C, H, W]
        target: 原始图像 [B, C, H, W]
        mask: 掩码图 [B, H, W] (0=可见, 1=掩码)
        """
        # 扩展掩码维度以匹配图像通道数
        mask = mask.unsqueeze(1)  # [B, 1, H, W]

        # 像素级归一化（可选）
        if self.norm_pix:
            mean = target.mean(dim=(1, 2, 3), keepdim=True)
            var = target.var(dim=(1, 2, 3), keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        # 计算MSE损失，仅对掩码区域
        loss = (pred - target) ** 2
        loss = loss * mask  # 只保留掩码区域

        # 归一化损失值（除以掩码区域像素总数）
        return loss.sum() / mask.sum()


from thop import profile		 ## 导入thop模块
def cal_params_flops(model, size, logger):
    input = torch.randn(1, 3, size, size).cuda()
    flops, params = profile(model, inputs=(input,))
    print('flops',flops/1e9)			## 打印计算量
    print('params',params/1e6)			## 打印参数量

    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.3fM" % (total/1e6))
    logger.info(f'flops: {flops/1e9}, params: {params/1e6}, Total params: : {total/1e6:.4f}')

