import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# AWRE (Adaptive Wavelet Refinement Expert) —— 自适应小波提炼专家

# =========================
# 构造 Daubechies-2 (Db2) 小波核
# =========================
def build_wavelet_kernels(device=None, dtype=torch.float32):
    """
    构造 Db2 (Daubechies-2) 小波核
    分解核 (Analysis) 和 重构核 (Synthesis)
    形状: (4, 1, 4, 4)
    """
    # Db2 的四个基础系数 (Low-pass Decomposition)
    c0 = (1 + math.sqrt(3)) / (4 * math.sqrt(2))
    c1 = (3 + math.sqrt(3)) / (4 * math.sqrt(2))
    c2 = (3 - math.sqrt(3)) / (4 * math.sqrt(2))
    c3 = (1 - math.sqrt(3)) / (4 * math.sqrt(2))

    # 构造一维滤波器 (按 PyTorch 卷积权重顺序排列)
    # 分解低通 h0 (Low-pass Analysis)
    h0 = torch.tensor([c3, c2, c1, c0], dtype=dtype, device=device)
    # 分解高通 h1 (High-pass Analysis): alternating flip
    h1 = torch.tensor([-c0, c1, -c2, c3], dtype=dtype, device=device)

    # 重构低通 g0 (Low-pass Synthesis): h0 的逆序
    g0 = torch.tensor([c0, c1, c2, c3], dtype=dtype, device=device)
    # 重构高通 g1 (High-pass Synthesis): h1 的逆序
    g1 = torch.tensor([c3, -c2, c1, -c0], dtype=dtype, device=device)

    # --- 1. 构建分解 (Analysis) 2D 核 ---
    # 外积生成 4x4
    LL = torch.ger(h0, h0)
    LH = torch.ger(h0, h1)
    HL = torch.ger(h1, h0)
    HH = torch.ger(h1, h1)
    # stack -> (4, 1, 4, 4)
    filt_ana = torch.stack([LL, LH, HL, HH], dim=0).unsqueeze(1)

    # --- 2. 构建重构 (Synthesis) 2D 核 ---
    LL_r = torch.ger(g0, g0)
    LH_r = torch.ger(g0, g1)
    HL_r = torch.ger(g1, g0)
    HH_r = torch.ger(g1, g1)
    filt_syn = torch.stack([LL_r, LH_r, HL_r, HH_r], dim=0).unsqueeze(1)

    return filt_ana, filt_syn


class AdaptiveWaveletRefinementExpert(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # 1. 软阈值系数 (针对不同域的噪声水平学习不同的阈值)
        self.theta = nn.Parameter(torch.zeros(3, channels, 1, 1))

        # 2. 频率增强适配器
        # 用来学习该领域特定的高频纹理特征
        self.high_freq_adapter = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False),
            nn.Sigmoid()  # 生成 0~1 的门控权重
        )

        # 3. 获取 Db2 核
        filt_ana, filt_syn = build_wavelet_kernels()
        # 注册为 buffer
        self.register_buffer("w_ana", filt_ana)  # 分解核
        self.register_buffer("w_syn", filt_syn)  # 重构核

    def dwt(self, x):
        """
        Db2 DWT: Input (B, C, H, W) -> Output (LH, HL, HH, LL) size (H/2, W/2)
        """
        B, C, H, W = x.shape

        # 1. 处理奇数尺寸 (如果有)
        # 即使是反射填充，也最好先保证输入是偶数，防止下采样丢失
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # 2. 反射填充 (Reflection Padding)
        x_pad = F.pad(x, (1, 1, 1, 1), mode='reflect')

        # 3. 卷积
        weight = self.w_ana.repeat(C, 1, 1, 1)
        y = F.conv2d(x_pad, weight=weight, stride=2, padding=0, groups=C)

        # 4. 拆分
        y = y.view(B, C, 4, y.shape[2], y.shape[3])
        LL = y[:, :, 0]
        LH = y[:, :, 1]
        HL = y[:, :, 2]
        HH = y[:, :, 3]
        return LH, HL, HH, LL

    def idwt(self, LH, HL, HH, LL):
        """
        Db2 IDWT: 逆变换
        """
        B, C, h, w = LL.shape

        # 1. 堆叠回去
        y = torch.stack([LL, LH, HL, HH], dim=2).view(B, 4 * C, h, w)

        # 2. 转置卷积 (Transposed Conv)
        weight = self.w_syn.repeat(C, 1, 1, 1)

        x_rec = F.conv_transpose2d(y, weight=weight, stride=2, padding=1, groups=C)

        return x_rec

    # ---------- 高频软阈值 ----------
    @staticmethod
    def soft_threshold(x, thr):
        return torch.sign(x) * F.relu(torch.abs(x) - thr)

    def forward(self, x):
        # x: [B, C, H, W]
        # 1. DWT 分解
        LH, HL, HH, LL = self.dwt(x)

        # 2. 软阈值去噪 (Soft Thresholding)
        # 计算阈值
        m_LH = LH.abs().mean(dim=(2, 3), keepdim=True)
        m_HL = HL.abs().mean(dim=(2, 3), keepdim=True)
        m_HH = HH.abs().mean(dim=(2, 3), keepdim=True)

        # Sigmoid 约束 theta，防止阈值变为负数或过大
        t = torch.sigmoid(self.theta)
        LH_hat = self.soft_threshold(LH, t[0].unsqueeze(0) * m_LH)
        HL_hat = self.soft_threshold(HL, t[1].unsqueeze(0) * m_HL)
        HH_hat = self.soft_threshold(HH, t[2].unsqueeze(0) * m_HH)

        # 3. 领域高频增强
        H_cat = torch.cat([LH_hat, HL_hat, HH_hat], dim=1)  # [B, 3C, H/2, W/2]

        # 计算高频门控
        scale = self.high_freq_adapter(H_cat)  # [B, 3C, H/2, W/2]
        scale_LH, scale_HL, scale_HH = torch.chunk(scale, 3, dim=1)

        # 重新加权高频
        LH_refined = LH_hat * scale_LH
        HL_refined = HL_hat * scale_HL
        HH_refined = HH_hat * scale_HH

        # 4. IDWT 重建
        X_re = self.idwt(LH_refined, HL_refined, HH_refined, LL)


        # 如果输入因为奇数尺寸被pad过，输出可能会比x大1个像素，需要crop一下
        if X_re.shape != x.shape:
            X_re = X_re[:, :, :x.shape[2], :x.shape[3]]

        # 5. 输出
        return X_re + x  # 残差连接

