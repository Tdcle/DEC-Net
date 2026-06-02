import numpy as np
import torch
from tqdm import tqdm
from ALRA import ALRA
from engine.evaluate import evaluate_oneClass
from engine.test import test_oneClass


def train_one_epoch(
        train_loader1,
        train_loader2,
        train_loader3,
        model,
        criterion,
        optimizer,
        scheduler,
        epoch,
        logger,
        config,
        alra,  # 必须传入外部初始化的 alra 实例
        scaler=None
):
    model.train()
    total_samples = 0

    # 1. 初始化累计变量
    # 用于计算当前 Epoch 的累计平均 Loss (Training Loss)
    # 形状 [3], 分别对应 Task1, Task2, Task3
    epoch_task_losses_sum = torch.zeros(3).cuda()
    n_batches = 0

    if alra is None:
        raise ValueError("Error: AKRA instance is None.")

    class ZippedLoader:
        def __init__(self, loader1, loader2, loader3):
            self.loader1, self.loader2, self.loader3 = loader1, loader2, loader3
            self.length = min(len(loader1), len(loader2), len(loader3))

        def __iter__(self):
            return zip(self.loader1, self.loader2, self.loader3)

        def __len__(self):
            return self.length

    combined_loader = ZippedLoader(train_loader1, train_loader2, train_loader3)
    ds_weights = [0.2, 0.2, 0.2, 0.2]

    # 记录 epoch 开始时的旧权重，用于最后对比
    with torch.no_grad():
        old_weights = torch.nn.functional.softmax(alra.w, dim=-1).clone()

    pbar = tqdm(
        combined_loader,
        desc=f"Epoch {epoch} [Train]",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar:10}{r_bar}"
    )

    for batch_idx, (data1, data2, data3) in enumerate(pbar):
        # --- 数据移动 ---
        images1, targets1 = data1[0].cuda(non_blocking=True).float(), data1[1].cuda(non_blocking=True).float()
        images2, targets2 = data2[0].cuda(non_blocking=True).float(), data2[1].cuda(non_blocking=True).float()
        images3, targets3 = data3[0].cuda(non_blocking=True).float(), data3[1].cuda(non_blocking=True).float()

        batch_size = images1.size(0) + images2.size(0) + images3.size(0)

        # --- 前向计算 ---
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            # Task 1
            outputs1, output1_ds = model(images1, task_id=0, epoch=epoch)
            loss1 = criterion(outputs1, targets1)
            for ds_out, weight in zip(output1_ds, ds_weights):
                loss1 += weight * criterion(ds_out, targets1)

            # Task 2
            outputs2, output2_ds = model(images2, task_id=1, epoch=epoch)
            loss2 = criterion(outputs2, targets2)
            for ds_out, weight in zip(output2_ds, ds_weights):
                loss2 += weight * criterion(ds_out, targets2)

            # Task 3
            outputs3, output3_ds = model(images3, task_id=2, epoch=epoch)
            loss3 = criterion(outputs3, targets3)
            for ds_out, weight in zip(output3_ds, ds_weights):
                loss3 += weight * criterion(ds_out, targets3)

            # 堆叠当前 Batch 的 Loss
            current_losses = torch.stack([loss1, loss2, loss3])

        # --- 2. 累计 Loss (用于显示平滑平均值 和 最终ALRA更新) ---
        epoch_task_losses_sum += current_losses.detach()
        n_batches += 1

        # 计算当前的累计平均值 (这就是 ALRA 在 epoch 结束时看到的数值)
        current_running_avg = epoch_task_losses_sum / n_batches

        # --- 反向传播 ---
        optimizer.zero_grad()
        weighted_loss = alra.backward(current_losses, epoch=epoch, scaler=scaler if config.amp else None)

        if config.amp:
            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_samples += batch_size

        # --- 更新进度条 ---
        # 显示的是：[当前平均 Loss] 和 [当前固定权重]
        pbar.set_postfix({
            'AvgL1': f"{current_running_avg[0]:.3f}",
            'AvgL2': f"{current_running_avg[1]:.3f}",
            'AvgL3': f"{current_running_avg[2]:.3f}",
            'W': f"[{old_weights[0]:.2f},{old_weights[1]:.2f},{old_weights[2]:.2f}]",
        })

    # === Epoch 结束 ===

    # 计算最终的 Epoch 平均 Loss
    final_epoch_avg_losses = epoch_task_losses_sum / n_batches

    # 获取上一个 Epoch 的 Loss (如果有的话)，用于打印调试信息
    prev_loss_log = "None"
    delta_log = "None"
    if alra.prev_epoch_loss is not None:
        prev_loss_vec = alra.prev_epoch_loss
        # 模拟计算下降比例 (Delta)
        delta_vec = (prev_loss_vec + 1e-8).log() - (final_epoch_avg_losses + 1e-8).log()

        prev_loss_log = f"[{prev_loss_vec[0]:.3f}, {prev_loss_vec[1]:.3f}, {prev_loss_vec[2]:.3f}]"
        # Delta > 0 代表下降了，值越大下降越快
        delta_log = f"[{delta_vec[0]:.3f}, {delta_vec[1]:.3f}, {delta_vec[2]:.3f}]"

    # --- 更新 ALRA 权重 ---
    alra.update_epoch(final_epoch_avg_losses, epoch=epoch, policy='balance')
    # alra.update_epoch(final_epoch_avg_losses, policy='accelerate')

    # 获取更新后的新权重
    with torch.no_grad():
        new_weights = torch.nn.functional.softmax(alra.w, dim=-1)

    current_lr = optimizer.param_groups[0]['lr']

    # --- 详细日志 ---
    logger.info(f"\n{'=' * 20} Epoch {epoch} alra Diagnosis {'=' * 20}")
    logger.info(f"1. Last Epoch Loss : {prev_loss_log}")
    logger.info(
        f"2. This Epoch Loss : [{final_epoch_avg_losses[0]:.3f}, {final_epoch_avg_losses[1]:.3f}, {final_epoch_avg_losses[2]:.3f}]")
    logger.info(f"3. Decrease Rate   : {delta_log} (Higher means faster drop)")
    logger.info(f"4. Weights Change  : "
                f"[{old_weights[0]:.2f}, {old_weights[1]:.2f}, {old_weights[2]:.2f}] -> "
                f"[{new_weights[0]:.2f}, {new_weights[1]:.2f}, {new_weights[2]:.2f}]")
    logger.info(f"{'=' * 60}\n")

    scheduler.step()



from medpy.metric.binary import hd95
def val_one_epoch(
        test_loader1,
        test_loader2,
        test_loader3,
        model,
        criterion,
        epoch,
        logger,
        config
):
    model.eval()
    # 分别评估两个数据集
    metrics1 = evaluate_oneClass(model, criterion, config, test_loader1, config.datasets_name[0], task_id=0)
    metrics2 = evaluate_oneClass(model, criterion, config, test_loader2, config.datasets_name[1], task_id=1)
    metrics3 = evaluate_oneClass(model, criterion, config, test_loader3, config.datasets_name[2], task_id=2)

    # 计算综合指标（加权平均）
    combined_metrics = {
        key: (metrics1[key] + metrics2[key] + metrics3[key]) / 3
        for key in metrics1.keys()
    }

    # 1. 获取数据集名称
    d1_name, d2_name, d3_name = config.datasets_name[0], config.datasets_name[1], config.datasets_name[2]

    # 2. 动态构建日志模板
    # 我们不再使用一个大的format，而是分步构建
    log_info = (
        f"Epoch {epoch} || "
        f"{d1_name}: Loss {metrics1['loss']:.4f} DSC {metrics1['dsc']:.4f} | "
        f"{d2_name}: Loss {metrics2['loss']:.4f} DSC {metrics2['dsc']:.4f} | "
        f"{d3_name}: Loss {metrics3['loss']:.4f} DSC {metrics3['dsc']:.4f} | "
        f"Combined: Loss {combined_metrics['loss']:.4f} DSC {combined_metrics['dsc']:.4f}\n"
    )


    print(log_info)
    logger.info(log_info)

    return combined_metrics['dsc']  # 返回综合dsc用于模型选择


def test_one_epoch(
        test_loader1,
        test_loader2,
        test_loader3,
        model,
        criterion,
        logger,
        config
):
    dataset_names = (config.datasets_name[0], config.datasets_name[1], config.datasets_name[2])
    model.eval()

    # 分别评估数据集
    results = {}
    results[dataset_names[0]] = test_oneClass(
        model, criterion, config, test_loader1,
        dataset_name=dataset_names[0], task_id=0
    )

    results[dataset_names[1]] = test_oneClass(
        model, criterion, config, test_loader2,
        dataset_name=dataset_names[1], task_id=1
    )

    results[dataset_names[2]] = test_oneClass(
        model, criterion, config, test_loader3,
        dataset_name=dataset_names[2], task_id=2
    )

    # 计算综合指标
    metric_keys = ['loss', 'acc', 'sens', 'spec', 'dsc', 'iou', 'hd95', 'time_ms', 'fps']
    combined = {}
    for metric in metric_keys:
        values = [results[d][metric] for d in dataset_names if metric in results[d]]
        if metric == 'hd95':
            combined[metric] = np.nanmean(values) if values else np.nan
        else:
            combined[metric] = np.mean(values) if values else 0

    # 构建紧凑的表格式日志
    log_info = "\n" + "=" * 100 + "\n"
    log_info += "                                    TEST RESULTS\n"
    log_info += "=" * 100 + "\n"

    # 表头
    log_info += f"{'Dataset':<15} {'Loss':<8} {'DSC':<8} {'IoU':<8} {'HD95':<8} {'Acc':<8} {'Sens':<8} {'Spec':<8} {'Time(ms)':<10} {'FPS':<8}\n"
    log_info += "-" * 100 + "\n"

    # 每个数据集的结果
    for idx, name in enumerate(dataset_names):
        num_classes = model.num_classes_with_bg[idx]
        num_foreground = num_classes - 1

        # 获取时间指标
        time_ms = results[name]['time_ms']
        fps = results[name]['fps']

        # 主要指标行
        log_info += (
            f"{name:<15} "
            f"{results[name]['loss']:<8.4f} "
            f"{results[name]['dsc']:<8.4f} "
            f"{results[name]['iou']:<8.4f} "
            f"{results[name]['hd95']:<8.2f} "
            f"{results[name]['acc']:<8.4f} "
            f"{results[name]['sens']:<8.4f} "
            f"{results[name]['spec']:<8.4f} "
            f"{time_ms:<10.2f} "  # 单张图片毫秒数
            f"{fps:<8.2f}\n"  # FPS
        )

        # 如果有多个前景类，显示每个类的DSC和IoU
        if num_foreground > 1:
            for class_id in range(1, num_classes):
                if f'class{class_id}_dsc' in results[name]:
                    log_info += (
                        f"  └─ Class {class_id}   "
                        f"{'─' * 8} "
                        f"{results[name][f'class{class_id}_dsc']:<8.4f} "
                        f"{results[name][f'class{class_id}_iou']:<8.4f} "
                        f"{results[name][f'class{class_id}_hd95']:<8.2f}\n"
                    )

    # 综合指标
    log_info += "=" * 100 + "\n"
    log_info += (
        f"{'COMBINED':<15} "
        f"{combined['loss']:<8.4f} "
        f"{combined['dsc']:<8.4f} "
        f"{combined['iou']:<8.4f} "
        f"{combined['hd95']:<8.2f} "
        f"{combined['acc']:<8.4f} "
        f"{combined['sens']:<8.4f} "
        f"{combined['spec']:<8.4f} "
        f"{combined['time_ms']:<10.2f} "
        f"{combined['fps']:<8.2f}\n"
    )
    log_info += "=" * 100 + "\n"

    print(log_info)
    logger.info(log_info)

    return combined['dsc']
