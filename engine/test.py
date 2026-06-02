import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from utils import save_imgs, save_imgs_multi
import os
import time
from medpy.metric.binary import hd95

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


def test_oneClass(model, criterion, config, loader, dataset_name, task_id):
    """
    测试函数 - 每个前景类单独计算指标并求平均
    包括HD95计算
    """
    # 获取该任务的类别数
    num_classes = model.num_classes_with_bg[task_id]  # 包括背景

    preds = []
    gts = []
    loss_list = []

    # 为每个前景类准备HD95列表
    hd95_per_class = {class_id: [] for class_id in range(1, num_classes)}

    # --- 新增：时间统计变量 ---
    total_inference_time = 0.0  # 纯推理时间累积
    total_samples = 0  # 处理样本总数
    start_run_time = time.time()  # 整个测试过程开始时间
    # -----------------------
    # ================= 准备保存图片的目录 =================
    save_vis = getattr(config, 'save_visualization', False)
    if save_vis:
        base_save_dir = os.path.join(config.outputs, dataset_name)
        img_save_dir = os.path.join(base_save_dir, 'images')
        pred_save_dir = os.path.join(base_save_dir, 'preds')
        gt_save_dir = os.path.join(base_save_dir, 'ground_truth')
        overlay_save_dir = os.path.join(base_save_dir, 'overlay')  # 新增：叠加图目录

        os.makedirs(img_save_dir, exist_ok=True)
        os.makedirs(pred_save_dir, exist_ok=True)
        os.makedirs(gt_save_dir, exist_ok=True)
        os.makedirs(overlay_save_dir, exist_ok=True)  # 创建叠加图目录

        global_sample_idx = 0
        # ======================================================
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(loader, desc=f'Testing {dataset_name}')):
            img, msk = data
            img = img.cuda().float()
            msk = msk.cuda().float()

            batch_size = img.size(0)

            # --- 新增：纯推理时间计时 ---
            torch.cuda.synchronize()  # 等待数据加载完成
            t_start = time.time()
            # 1. 获取logits
            logits = model(img, task_id=task_id)  # (B, C, H, W)
            torch.cuda.synchronize()  # 等待模型推理完成
            t_end = time.time()
            total_inference_time += (t_end - t_start)
            total_samples += batch_size
            # -------------------------

            logits = logits[:, :num_classes, :, :]
            # 2. 计算loss
            msk_for_loss = msk.squeeze(1).long() if msk.dim() == 4 else msk.long()
            loss = criterion(logits, msk_for_loss)
            loss_list.append(loss.item())

            # 3. 使用argmax获取预测类别
            if isinstance(logits, tuple):
                logits = logits[0]

            pred_class = torch.argmax(logits, dim=1)  # (B, H, W)
            batch_pred = pred_class.cpu().numpy()

            # 4. 处理ground truth
            batch_gt = msk.squeeze(1).cpu().numpy() if msk.dim() == 4 else msk.cpu().numpy()

            # 5. 逐样本计算HD95（对每个前景类）
            for sample_idx in range(batch_pred.shape[0]):
                pred = batch_pred[sample_idx]  # (H, W)
                gt = batch_gt[sample_idx]  # (H, W)

                # 对每个前景类计算HD95
                for class_id in range(1, num_classes):
                    # 二值化：当前类 vs 其他
                    y_pred_class = (pred == class_id).astype(np.uint8)
                    y_true_class = (gt == class_id).astype(np.uint8)

                    # 只在该类别存在时计算HD95
                    if y_true_class.sum() > 0:
                        try:
                            hd_val = hd95(y_pred_class, y_true_class)
                            hd95_per_class[class_id].append(hd_val)
                        except:
                            hd95_per_class[class_id].append(np.nan)


            # 6. 收集完整数据用于计算其他指标
            preds.append(batch_pred.reshape(-1))
            gts.append(batch_gt.reshape(-1))

            # 7. 可选：保存可视化
            # if config.save_visualization:
            #     save_imgs(...)

    # --- 新增：计算时间指标 ---
    end_run_time = time.time()
    total_duration = end_run_time - start_run_time  # 整个测试流程耗时（秒）
    # 防止除以零
    if total_samples > 0:
        avg_inference_time = total_inference_time / total_samples  # 秒/张
        fps = 1.0 / avg_inference_time
        time_per_img_ms = avg_inference_time * 1000  # 毫秒/张
    else:
        fps = 0
        time_per_img_ms = 0
    # -----------------------

    # 合并所有数据
    y_pred = np.concatenate(preds).astype(np.uint8)
    y_true = np.concatenate(gts).astype(np.uint8)

    # 对每个前景类计算指标
    metrics = {
        'loss': np.mean(loss_list),
        'fps': fps,  # 新增
        'time_ms': time_per_img_ms,  # 新增
        'total_duration': total_duration  # 新增
    }

    class_metrics = {
        'acc': [], 'sens': [], 'spec': [], 'dsc': [], 'iou': [], 'hd95': []
    }

    # 遍历所有前景类
    for class_id in range(1, num_classes):
        # One-vs-rest
        pred_binary = (y_pred == class_id).astype(np.uint8)
        true_binary = (y_true == class_id).astype(np.uint8)

        # 只在该类别存在时计算
        if true_binary.sum() > 0:
            cm = confusion_matrix(true_binary, pred_binary, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()

            acc = (tn + tp) / (tn + fp + fn + tp) if (tn + fp + fn + tp) > 0 else 0
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            dsc = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
            hd95_mean = np.nanmean(hd95_per_class[class_id]) if hd95_per_class[class_id] else np.nan

            # 保存该类的指标
            metrics[f'class{class_id}_acc'] = acc
            metrics[f'class{class_id}_sens'] = sens
            metrics[f'class{class_id}_spec'] = spec
            metrics[f'class{class_id}_dsc'] = dsc
            metrics[f'class{class_id}_iou'] = iou
            metrics[f'class{class_id}_hd95'] = hd95_mean

            # 添加到列表用于计算平均
            class_metrics['acc'].append(acc)
            class_metrics['sens'].append(sens)
            class_metrics['spec'].append(spec)
            class_metrics['dsc'].append(dsc)
            class_metrics['iou'].append(iou)
            if not np.isnan(hd95_mean):
                class_metrics['hd95'].append(hd95_mean)

    # 计算所有前景类的平均指标
    metrics['acc'] = np.mean(class_metrics['acc']) if class_metrics['acc'] else 0
    metrics['sens'] = np.mean(class_metrics['sens']) if class_metrics['sens'] else 0
    metrics['spec'] = np.mean(class_metrics['spec']) if class_metrics['spec'] else 0
    metrics['dsc'] = np.mean(class_metrics['dsc']) if class_metrics['dsc'] else 0
    metrics['iou'] = np.mean(class_metrics['iou']) if class_metrics['iou'] else 0
    metrics['hd95'] = np.nanmean(class_metrics['hd95']) if class_metrics['hd95'] else np.nan

    return metrics

