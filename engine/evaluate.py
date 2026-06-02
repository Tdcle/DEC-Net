import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

def get_dice_threshold(output, mask, threshold=0.5):
    """
    :param output: output shape per image, float, (0,1)
    :param mask: mask shape per image, float, (0,1)
    :param threshold: the threshold to binarize output and feature (0,1)
    :return: dice of threshold t
    """
    smooth = 1e-6

    zero = torch.zeros_like(output)
    one = torch.ones_like(output)
    output = torch.where(output > threshold, one, zero)
    mask = torch.where(mask > threshold, one, zero)
    intersection = (output * mask).sum()
    dice = (2. * intersection + smooth) / (output.sum() + mask.sum() + smooth)

    return dice

def get_hard_dice(outputs, masks, return_list=False, threshold=0.5):
    """
    :param outputs: 模型输出 [B, 2, H, W] (双通道概率图)
    :param masks: 标签 [B, 1, H, W] (单通道整数掩码)
    """
    batch_size = outputs.size(0)
    # 分别存储视盘/视杯的Dice
    disc_dice_list = []
    cup_dice_list = []

    for i in range(batch_size):
        # 步骤1：生成真值二值掩码
        disc_mask = (masks[i, 0] > 0).float()  # 视盘真值 [H, W]
        cup_mask = (masks[i, 0] == 2).float()  # 视杯真值 [H, W]

        # 步骤2：按通道计算Dice
        # 视盘通道 (通道0)
        disc_output = outputs[i, 0]  # [H, W]
        disc_dice = get_dice_threshold(disc_output, disc_mask, threshold)
        disc_dice_list.append(disc_dice)

        # 视杯通道 (通道1)
        cup_output = outputs[i, 1]  # [H, W]
        cup_dice = get_dice_threshold(cup_output, cup_mask, threshold)
        cup_dice_list.append(cup_dice)

    # 步骤3：返回结果
    mean_disc_dice = np.mean(disc_dice_list)
    mean_cup_dice = np.mean(cup_dice_list)

    if return_list:
        return (mean_disc_dice, mean_cup_dice), (disc_dice_list, cup_dice_list)
    else:
        return mean_disc_dice, mean_cup_dice

# def evaluate_oneClass(model, criterion, config, loader, dataset_name, task_id):
#     preds = []
#     gts = []
#     loss_list = []
#     aux_loss_list = []
#
#     with torch.no_grad():
#         for data in tqdm(loader, desc=f'Validating {dataset_name}'):
#             img, msk = data
#             img = img.cuda().float()
#             msk = msk.cuda().float()
#
#             # 前向传播
#             out = model(img, task_id=task_id)
#             loss = criterion(out, msk)
#             loss_list.append(loss.item())
#
#
#             # 处理多输出模型
#             if isinstance(out, tuple):
#                 out = out[0]
#
#             # 转换为CPU numpy
#             batch_pred = out.squeeze(1).cpu().numpy()
#             batch_gt = msk.squeeze(1).cpu().numpy()
#
#             # # 逐样本计算HD95
#             # for pred, gt in zip(batch_pred, batch_gt):
#             #     y_pred = (pred >= config.threshold).astype(np.uint8)
#             #     y_true = (gt >= 0.5).astype(np.uint8)
#             #
#             #     try:
#             #         hd_val = hd95(y_pred, y_true)
#             #     except:
#             #         hd_val = np.nan  # 处理空标签等异常
#             #     hd95_list.append(hd_val)
#
#             # 收集数据
#             preds.append(batch_pred.reshape(-1))
#             gts.append(batch_gt.reshape(-1))
#
#     # 合并结果
#     y_pred_all = np.concatenate(preds)
#     y_true_all = np.concatenate(gts)
#
#     # 二值化
#     y_pred_bin = (y_pred_all >= config.threshold).astype(np.uint8)
#     y_true_bin = (y_true_all >= 0.5).astype(np.uint8)
#
#     # 计算指标
#     tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()
#     # acc = (tp + tn) / (tp + tn + fp + fn)
#     # sens = tp / (tp + fn) if (tp + fn) > 0 else 0
#     # spec = tn / (tn + fp) if (tn + fp) > 0 else 0
#     dsc = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
#     miou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
#     # mean_hd = np.nanmean(hd95_list)
#
#     return {
#         'loss': np.mean(loss_list),
#         # 'acc': acc,
#         # 'sens': sens,
#         # 'spec': spec,
#         'dsc': dsc,
#         # 'miou': miou,
#     }


def evaluate_oneClass(model, criterion, config, loader, dataset_name, task_id):
    """
    验证函数 - 每个前景类单独计算指标并求平均
    """
    # 获取该任务的类别数
    num_classes = model.num_classes_with_bg[task_id]  # 包括背景

    preds = []
    gts = []
    loss_list = []

    model.eval()
    with torch.no_grad():
        for data in tqdm(loader, desc=f'Validating {dataset_name}'):
            img, msk = data
            img = img.cuda().float()
            msk = msk.cuda().float()

            # 1. 获取logits
            logits = model(img, task_id=task_id)  # (B, C, H, W)

            # 2. 计算loss
            msk_for_loss = msk.squeeze(1).long() if msk.dim() == 4 else msk.long()
            loss = criterion(logits, msk_for_loss)
            loss_list.append(loss.item())

            # 3. 使用argmax获取预测类别（不再使用阈值）
            pred_class = torch.argmax(logits, dim=1)  # (B, H, W)
            batch_pred = pred_class.cpu().numpy()

            # 4. 处理ground truth
            batch_gt = msk.squeeze(1).cpu().numpy() if msk.dim() == 4 else msk.cpu().numpy()

            # 收集完整的预测和标签（保留类别信息）
            preds.append(batch_pred.reshape(-1))
            gts.append(batch_gt.reshape(-1))

    # 合并所有数据
    y_pred = np.concatenate(preds).astype(np.uint8)
    y_true = np.concatenate(gts).astype(np.uint8)

    # 对每个前景类分别计算指标
    metrics = {'loss': np.mean(loss_list)}

    class_dsc_list = []
    class_iou_list = []

    # 遍历所有前景类（跳过背景类0）
    for class_id in range(1, num_classes):
        # One-vs-rest：当前类 vs 其他所有类
        pred_binary = (y_pred == class_id).astype(np.uint8)
        true_binary = (y_true == class_id).astype(np.uint8)

        # 只在该类别存在时计算（避免除零）
        if true_binary.sum() > 0:
            cm = confusion_matrix(true_binary, pred_binary, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()

            dsc = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

            class_dsc_list.append(dsc)
            class_iou_list.append(iou)

            # 保存每个类的指标
            metrics[f'class{class_id}_dsc'] = dsc
            metrics[f'class{class_id}_iou'] = iou

    # 计算所有前景类的平均指标
    metrics['dsc'] = np.mean(class_dsc_list) if class_dsc_list else 0
    metrics['iou'] = np.mean(class_iou_list) if class_iou_list else 0

    return metrics

def evaluate_refuge2(model, criterion, config, loader, dataset_name, task_id):
    loss_list = []
    # hd95_list = []
    disc_dice_list = []
    cup_dice_list = []
    aux_loss_list = []

    with torch.no_grad():
        for data in tqdm(loader, desc=f'Validating {dataset_name}'):
            img, msk = data
            img = img.cuda(non_blocking=True).float()  # [B, C, H, W]
            msk = msk.cuda(non_blocking=True).float()  # [B, C, H, W]

            # 视盘真值：所有非背景区域（标签1和2）
            disc_target = (msk > 0).float()  # [B, 1, H, W]
            # 视杯真值：仅标签2区域
            cup_target = (msk == 2).float()  # [B, 1, H, W]

            # 前向推理
            out = model(img, task_id=task_id)  # [B, C, H, W]
            # 分别计算视盘和视杯损失
            disc_loss = criterion(out[:, 0:1], disc_target)  # 通道0 → 视盘
            cup_loss = criterion(out[:, 1:2], cup_target)  # 通道1 → 视杯
            loss = disc_loss + cup_loss
            loss_list.append(loss.item())

            # 转换为numpy数组
            batch_pred = out.cpu().numpy()  # [B, C, H, W]
            batch_gt = msk.cpu().numpy()  # [B, C, H, W]

            for b in range(batch_pred.shape[0]):
                # 获取当前样本的双通道输出和标签
                batch_output = torch.from_numpy(batch_pred[b])  # [2, H, W]
                batch_mask = torch.from_numpy(batch_gt[b])  # [1, H, W]

                # 计算单样本视盘/视杯Dice
                disc_dice, cup_dice = get_hard_dice(
                    batch_output.unsqueeze(0),
                    batch_mask.unsqueeze(0),
                    threshold=config.threshold
                )
                disc_dice_list.append(disc_dice)
                cup_dice_list.append(cup_dice)

                # # === HD95计算 ===
                # try:
                #     # 视盘HD95（通道0）
                #     disc_pred = (batch_pred[b, 0] > config.threshold).astype(np.float32)
                #     disc_gt = (batch_gt[b, 0] > 0).astype(np.float32)  # 标签>0即视盘
                #     disc_hd = hd95(disc_pred, disc_gt) if np.any(disc_gt) else 0
                #
                #     # 视杯HD95（通道1）
                #     cup_pred = (batch_pred[b, 1] > config.threshold).astype(np.float32)
                #     cup_gt = (batch_gt[b, 0] == 2).astype(np.float32)  # 标签=2即视杯
                #     cup_hd = hd95(cup_pred, cup_gt) if np.any(cup_gt) else 0
                #
                #     hd95_list.append((disc_hd + cup_hd) / 2)  # 样本平均HD95
                # except Exception as e:
                #     logger.warning(f"HD95计算异常: {str(e)}")
                #     hd95_list.append(np.nan)

    mean_loss = np.mean(loss_list)
    mean_disc_dice = np.mean(disc_dice_list)
    mean_cup_dice = np.mean(cup_dice_list)
    average_dice = (mean_disc_dice + mean_cup_dice) / 2
    # mean_hd95 = np.nanmean(hd95_list)

    return {
        'loss': mean_loss,
        'dsc': average_dice,
        # 'miou': miou,
    }