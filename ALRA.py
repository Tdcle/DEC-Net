import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union

# (Adaptive Loss Rate Alignment)
class ALRA:
    def __init__(
            self,
            n_tasks: int,
            device: torch.device,
            gamma: float = 0.01,
            w_lr: float = 0.05,  # [建议] 稍微调小一点 LR，配合梯度裁剪，提升平滑度
            max_norm: float = 1.0,  # 这是给骨干网络用的
            warmup_epochs: int = 10,
            loss_smooth: float = 0.8,
            floor_value: float = 0.2,  # [修改] 含义变为：每个任务的绝对最小权重值
            w_grad_clip: float = 0.1  # [新增] w 的梯度裁剪阈值，防止权重跳变
    ):
        self.n_tasks = n_tasks
        self.device = device

        # 检查保底设置是否合法 (例如 3个任务，每个保底0.4，总和1.2 > 1.0 就会报错)
        if floor_value * n_tasks > 1.0:
            raise ValueError(f"Floor value {floor_value} too high for {n_tasks} tasks. Max allowed: {1.0 / n_tasks}")

        self.w = torch.tensor([0.0] * n_tasks, device=device, requires_grad=True)

        self.w_opt = torch.optim.Adam(
            [self.w],
            lr=w_lr,
            weight_decay=gamma,
            betas=(0.9, 0.999)  # [微调] beta1=0.9 增加动量，让权重变化更平滑
        )

        self.max_norm = max_norm
        self.prev_epoch_loss = None
        self.warmup_epochs = warmup_epochs
        self.loss_smooth = loss_smooth
        self.floor_value = floor_value
        self.w_grad_clip = w_grad_clip

        # [新增] 权重的影子变量，用于输出平滑后的权重 (类似 EMA Model)
        self.smooth_w_cache = None

    def get_weights(self, epoch):
        """
        获取当前使用的实际归一化权重。
        逻辑：Hard Floor + Dynamic Residual
        """
        # 1. 预热期：完全均匀
        if epoch < self.warmup_epochs:
            return torch.ones(self.n_tasks, device=self.device) / self.n_tasks

        # 2. 计算 ALRA 的动态部分 (Softmax)
        # 使用 detach() 的 smooth_w_cache 如果你想让权重本身也极其平滑 (可选)
        # 这里我们还是用实时的 w，但在 Update 时控制了它的跳变速度
        z = F.softmax(self.w, dim=-1)

        # 3. [硬保底逻辑]
        # Total Floor = 0.2 * 3 = 0.6
        # Residual (Dynamic) = 1.0 - 0.6 = 0.4
        # Final = 0.2 + 0.4 * Softmax(w)
        total_floor = self.floor_value * self.n_tasks
        residual = 1.0 - total_floor

        final_w = self.floor_value + residual * z

        return final_w

    def get_weighted_loss(self, losses, epoch):
        # 获取硬保底后的权重
        z = self.get_weights(epoch)

        D = losses + 1e-8
        # 使用你之前的 "Gradient Normalization" 风格公式
        c = (z / D).sum().detach()
        loss = (D.log() * z / c).sum()

        # [建议] 恢复量级：乘以 n_tasks
        # 这样 loss 的数值大概就在 sum(losses) 的量级，不用大幅调整模型 LR
        loss = loss * self.n_tasks

        return loss

    def update_epoch(self, curr_epoch_loss_raw, epoch, policy='balance'):  # 既然要平滑，建议用 balance
        """
        curr_epoch_loss_raw: Tensor shape [3]
        """
        # --- 1. Loss 平滑 (Momentum EMA) ---
        if self.prev_epoch_loss is None:
            self.prev_epoch_loss = curr_epoch_loss_raw.detach()
            smoothed_loss = self.prev_epoch_loss
        else:
            smoothed_loss = self.loss_smooth * self.prev_epoch_loss + \
                            (1 - self.loss_smooth) * curr_epoch_loss_raw.detach()

        if epoch < self.warmup_epochs:
            self.prev_epoch_loss = smoothed_loss
            return

        # --- 2. 计算 Delta ---
        # Delta > 0 代表进步了
        delta = (self.prev_epoch_loss + 1e-8).log() - (smoothed_loss + 1e-8).log()

        # --- 3. 计算梯度 ---
        with torch.enable_grad():
            # 这里很重要：我们只对 "动态部分" 求导
            # 因为保底部分是常数，没有梯度
            pure_softmax_w = F.softmax(self.w, -1)

            d = torch.autograd.grad(pure_softmax_w,
                                    self.w,
                                    grad_outputs=delta.detach())[0]

        self.w_opt.zero_grad()

        # 设置梯度方向
        if policy == 'balance':
            self.w.grad = d
        elif policy == 'accelerate':
            self.w.grad = -d

            # [关键新增] 梯度裁剪：防止 w 单次更新幅度过大
        torch.nn.utils.clip_grad_norm_([self.w], self.w_grad_clip)

        self.w_opt.step()

        # --- 4. 防止数值漂移 ---
        with torch.no_grad():
            self.w.data = self.w.data - self.w.data.mean()
            # 这里的 clamp 可以稍微小一点，因为我们已经有硬保底了
            # 限制 logits 范围，防止 softmax 变得极度尖锐
            self.w.data.clamp_(-2.0, 2.0)

            # 更新历史记录
        self.prev_epoch_loss = smoothed_loss

    def backward(self, losses, epoch, shared_parameters=None, scaler=None):
        loss = self.get_weighted_loss(losses=losses, epoch=epoch)

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if self.max_norm > 0 and shared_parameters is not None and scaler is None:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
        return loss