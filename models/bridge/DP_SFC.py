import torch
import torch.nn as nn
import torch.nn.functional as F


# ================= 基础组件 (Norm 默认 GroupNorm(1)) =================

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class Basic(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, relu=True, bn=True, bias=False):
        super(Basic, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, bias=bias)
        self.bn = nn.GroupNorm(1, out_planes) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


# ================= 实例感知任务提示生成器 (Instance-Aware TPG) =================

class TaskPromptGenerator(nn.Module):
    def __init__(self, num_tasks, embed_dim, in_channels):
        super().__init__()
        self.task_embed = nn.Embedding(num_tasks, embed_dim)

        # 1. 任务对齐层：将 embed_dim 映射到图片的通道数 in_channels
        self.task_proj = nn.Sequential(
            nn.Linear(embed_dim, in_channels),
            nn.ReLU()
        )

        # 2. 动态通道提示生成支路 (Instance-Aware Channel Prompt)
        # 接收 [B, in_channels, H, W] -> 输出 [B, in_channels]
        self.c_prompter = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 压缩空间维度 [B, C, 1, 1]
            nn.Flatten(),  # 展平为 [B, C]
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Linear(in_channels // 2, in_channels),
            nn.Sigmoid()  # 激发通道权重
        )

        # 3. 早期动态空间提示生成支路 (Early Spatial Prompt)
        # 接收 [B, in_channels, H, W] -> 输出 [B, 1, H, W]
        self.s_prompter = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.GroupNorm(1, in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1)
        )

    def forward(self, x, task_id):
        """
        x: 原始图像特征 [B, C, H, W]
        """
        B = x.shape[0]
        if isinstance(task_id, int):
            ids = torch.tensor([task_id] * B, device=x.device)
        else:
            ids = task_id

        # 纯任务 Embedding [B, embed_dim]
        embed = self.task_embed(ids)

        # --- 任务与特征的融合 ---
        # 映射任务向量以匹配通道数: [B, C]
        t_feat = self.task_proj(embed)

        # 广播机制相加: [B, C, H, W] + [B, C, 1, 1]
        x_task_fused = x + t_feat.unsqueeze(-1).unsqueeze(-1)

        # --- 分两路生成提示 ---
        # 结合图片内容的动态通道提示 (基础值 0.5 + 调制)
        c_prompt = self.c_prompter(x_task_fused) + 0.5
        # 结合图片内容的早期空间提示
        s_prompt = self.s_prompter(x_task_fused)

        return c_prompt, s_prompt, embed


class TaskGuidedCrossScaleContext(nn.Module):
    def __init__(self, n_feat, embed_dim):
        super(TaskGuidedCrossScaleContext, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(n_feat * 3 + embed_dim, n_feat // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(n_feat // 4, n_feat * 3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x1, x2, x3, task_embed):
        b, c, _, _ = x1.size()
        y1 = self.avg_pool(x1).view(b, c)
        y2 = self.avg_pool(x2).view(b, c)
        y3 = self.avg_pool(x3).view(b, c)

        y = torch.cat([y1, y2, y3, task_embed], dim=1)
        y = self.fc(y).view(b, c * 3, 1, 1)
        scale1, scale2, scale3 = torch.split(y, c, dim=1)
        return x1 * scale1, x2 * scale2, x3 * scale3


# ================= 主模块: TP-MSF =================

class DomainPrompted_SkipFeatureCalibration(nn.Module):
    def __init__(self, n_feat, num_tasks=3, embed_dim=32):
        super(DomainPrompted_SkipFeatureCalibration, self).__init__()

        # 1. 动态提示生成器 (接收特征和task_id)
        self.prompter = TaskPromptGenerator(num_tasks, embed_dim, n_feat)

        # 2. 头部处理
        self.head_conv = nn.Conv2d(n_feat, n_feat, 3, 1, 1)
        self.head_norm = nn.ModuleList([nn.GroupNorm(1, n_feat) for _ in range(num_tasks)])
        self.act = nn.ReLU()

        # 3. 降维卷积
        inter_dim = max(8, n_feat // 4)
        self.reduce_conv = nn.Conv2d(n_feat, inter_dim, 1)

        # 4. 多尺度特征提取 (DW-Conv)
        self.multi_scale_conv = nn.ModuleList([
            nn.Conv2d(inter_dim, inter_dim, 3, padding=1, groups=inter_dim),
            nn.Conv2d(inter_dim, inter_dim, 5, padding=2, groups=inter_dim),
            nn.Conv2d(inter_dim, inter_dim, 7, padding=3, groups=inter_dim)
        ])

        # 5. 跨尺度上下文融合
        self.cross_scale_context = TaskGuidedCrossScaleContext(inter_dim, embed_dim)

        # 6. 融合卷积
        total_channels = 3 * inter_dim
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(total_channels, n_feat, 1),
            nn.GroupNorm(1, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, n_feat, 3, padding=1, groups=n_feat),
            nn.Conv2d(n_feat, n_feat, 1)
        )

    def forward(self, x, task_id):
        res = x
        B = x.shape[0]

        tid_int = task_id if isinstance(task_id, int) else task_id[0].item()

        # 得到通道提示 (c_prompt) 和 早期空间图 (early_s_prompt)
        c_prompt, early_s_prompt, t_embed = self.prompter(res, task_id)

        # Head 处理
        x = self.head_conv(x)
        x = self.head_norm[tid_int](x)
        x = self.act(x)

        # Pre-Modulation (使用通道提示乘以所有通道)
        x_modulated = x * c_prompt.unsqueeze(-1).unsqueeze(-1)

        # --- 特征提取阶段 ---
        x_reduced = self.reduce_conv(x_modulated)
        ms_feats = [conv(x_reduced) for conv in self.multi_scale_conv]
        ms_feats[0], ms_feats[1], ms_feats[2] = self.cross_scale_context(
            ms_feats[0], ms_feats[1], ms_feats[2], t_embed
        )
        feats = torch.cat(ms_feats, dim=1)
        x_fused = self.fusion_conv(feats)  # [B, n_feat, H, W]

        out = x_fused + early_s_prompt

        # 残差连接并返回
        return out + res