import torch
from torch import nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_
import math

from .DAMoE import DAME
from .bridge import DP_SFC

class LayerNorm(nn.Module):

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class DEC_Net(nn.Module):

    def __init__(self, num_classes=[1, 2, 1], input_channels=3, c_list=[8, 16, 24, 32, 48, 64], bridge=True, deep_supervision=True):
        super().__init__()

        self.bridge = bridge
        self.deep_supervision = deep_supervision
        self.num_tasks = len(num_classes)
        self.num_classes_with_bg = [nc + 1 for nc in num_classes]

        # ========== Encoder ==========
        self.encoder1 = nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1)

        depths = (1, 1, 1, 1)
        dp_rates = [x.tolist() for x in torch.linspace(0, 0.3, sum(depths)).split(depths)]

        self.encoder3 = DAME.DAMELayer(in_chs=c_list[1], out_chs=c_list[2], depth=depths[0],
                                 drop_path_rates=dp_rates[0], seq_len=64 * 64, num_tasks=self.num_tasks)

        self.encoder4 = DAME.DAMELayer(in_chs=c_list[2], out_chs=c_list[3], depth=depths[1],
                                 drop_path_rates=dp_rates[1], seq_len=32 * 32, num_tasks=self.num_tasks)

        self.encoder5 = DAME.DAMELayer(in_chs=c_list[3], out_chs=c_list[4], depth=depths[2],
                                 drop_path_rates=dp_rates[2], seq_len=16 * 16, num_tasks=self.num_tasks)

        # ========== Decoder ==========
        self.decoder1 = DAME.DAMELayer(in_chs=c_list[4], out_chs=c_list[3], depth=depths[2],
                                 drop_path_rates=dp_rates[2], seq_len=16 * 16, num_tasks=self.num_tasks)

        self.decoder2 = DAME.DAMELayer(in_chs=c_list[3], out_chs=c_list[2], depth=depths[1],
                                 drop_path_rates=dp_rates[1], seq_len=32 * 32, num_tasks=self.num_tasks)

        self.decoder3 = DAME.DAMELayer(in_chs=c_list[2], out_chs=c_list[1], depth=depths[0],
                                 drop_path_rates=dp_rates[0], seq_len=64 * 64, num_tasks=self.num_tasks)

        self.ebn1 = nn.ModuleList([ nn.GroupNorm(1, c_list[0]) for _ in range (len(num_classes)) ])
        self.ebn2 = nn.ModuleList([ nn.GroupNorm(1, c_list[1]) for _ in range (len(num_classes)) ])
        self.ebn3 = nn.ModuleList([ nn.GroupNorm(1, c_list[2]) for _ in range (len(num_classes)) ])
        self.ebn4 = nn.ModuleList([ nn.GroupNorm(1, c_list[3]) for _ in range (len(num_classes)) ])
        self.ebn5 = nn.ModuleList([ nn.GroupNorm(1, c_list[4]) for _ in range (len(num_classes)) ])

        self.dbn1 = nn.ModuleList([ nn.GroupNorm(1, c_list[3]) for _ in range (len(num_classes)) ])
        self.dbn2 = nn.ModuleList([ nn.GroupNorm(1, c_list[2]) for _ in range (len(num_classes)) ])
        self.dbn3 = nn.ModuleList([ nn.GroupNorm(1, c_list[1]) for _ in range (len(num_classes)) ])
        self.dbn4 = nn.ModuleList([ nn.GroupNorm(1, c_list[0]) for _ in range (len(num_classes)) ])

        self.decoder4 = nn.ModuleList([ nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1) for _ in range (len(num_classes)) ])


        self.final = nn.ModuleList([ nn.Conv2d(c_list[0], self.num_classes_with_bg[i], kernel_size=1) for i in range (len(num_classes)) ])

        if bridge:
            self.bg1 = DP_SFC.DomainPrompted_SkipFeatureCalibration(n_feat=c_list[0], embed_dim=64)
            self.bg2 = DP_SFC.DomainPrompted_SkipFeatureCalibration(n_feat=c_list[1], embed_dim=64)
            self.bg3 = DP_SFC.DomainPrompted_SkipFeatureCalibration(n_feat=c_list[2], embed_dim=64)
            self.bg4 = DP_SFC.DomainPrompted_SkipFeatureCalibration(n_feat=c_list[3], embed_dim=64)


        # ========== Deep Supervision Heads ==========
        if self.deep_supervision:
            self.ds_out4 = nn.ModuleList(
                [nn.Conv2d(c_list[3], self.num_classes_with_bg[i], kernel_size=1) for i in range(len(num_classes))])
            self.ds_out3 = nn.ModuleList(
                [nn.Conv2d(c_list[2], self.num_classes_with_bg[i], kernel_size=1) for i in range(len(num_classes))])
            self.ds_out2 = nn.ModuleList(
                [nn.Conv2d(c_list[1], self.num_classes_with_bg[i], kernel_size=1) for i in range(len(num_classes))])
            self.ds_out1 = nn.ModuleList(
                [nn.Conv2d(c_list[0], self.num_classes_with_bg[i], kernel_size=1) for i in range(len(num_classes))])

            print('Deep Supervision enabled')

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, task_id=0, epoch=1):
        # --- Stage 1 ---
        # Encoder1 保持原分辨率 (H, W)
        x1 = self.encoder1(x)
        x1 = self.ebn1[task_id](x1)  # GroupNorm
        t1 = x1  # 【保存】跳跃连接 (H, W)
        out = F.max_pool2d(x1, 2, 2)  # 下采样 -> (H/2, W/2)
        out = F.gelu(out)

        # --- Stage 2 ---
        x2 = self.encoder2(out)
        x2 = self.ebn2[task_id](x2)
        t2 = x2  # 【保存】 (H/2, W/2)
        out = F.max_pool2d(x2, 2, 2)  # -> (H/4, W/4)
        out = F.gelu(out)

        # --- Stage 3 ---
        x3 = self.encoder3(out, task_id, epoch)
        x3 = self.ebn3[task_id](x3)
        t3 = x3  # 【保存】 (H/4, W/4)
        out = F.max_pool2d(x3, 2, 2)  # -> (H/8, W/8)

        # --- Stage 4 ---
        x4 = self.encoder4(out, task_id, epoch)
        x4 = self.ebn4[task_id](x4)
        t4 = x4  # 【保存】 (H/8, W/8)
        out = F.max_pool2d(x4, 2, 2)  # -> (H/16, W/16)

        # --- Stage 5 (Bottleneck) ---
        # 此时 out 为 (H/32, W/32)
        out = self.encoder5(out, task_id, epoch)
        t5 = out

        # ===================== Bridge =====================
        if self.bridge:
            t1 = self.bg1(t1,task_id)
            t2 = self.bg2(t2,task_id)
            t3 = self.bg3(t3,task_id)
            t4 = self.bg4(t4,task_id)

        # ===================== Decoder =====================

        # --- Decoder Level 1 (对应 Encoder Stage 4) ---
        out = torch.add(out, t5)
        dec4 = self.decoder1(out, task_id, epoch)  # In: H/16, Out: H/16
        dec4 = self.dbn1[task_id](dec4)
        # 上采样到 H/8 以匹配 t4
        dec4_up = F.interpolate(dec4, size=t4.shape[2:], mode='bilinear', align_corners=True)
        out4 = torch.add(dec4_up, t4)

        # --- Decoder Level 2 (对应 Encoder Stage 3) ---
        dec3 = self.decoder2(out4, task_id, epoch)  # In: H/8, Out: H/8
        dec3 = self.dbn2[task_id](dec3)
        # 上采样到 H/4 以匹配 t3
        dec3_up = F.interpolate(dec3, size=t3.shape[2:], mode='bilinear', align_corners=True)
        out3 = torch.add(dec3_up, t3)

        # --- Decoder Level 3 (对应 Encoder Stage 2) ---
        dec2 = self.decoder3(out3, task_id, epoch)  # In: H/4, Out: H/4
        dec2 = self.dbn3[task_id](dec2)
        # 上采样到 H/2 以匹配 t2
        dec2_up = F.interpolate(dec2, size=t2.shape[2:], mode='bilinear', align_corners=True)
        out2 = torch.add(dec2_up, t2)

        # --- Decoder Level 4 (对应 Encoder Stage 1) ---
        dec1 = self.decoder4[task_id](out2)  # In: H/2, Out: H/2
        dec1 = self.dbn4[task_id](dec1)
        dec1 = F.gelu(dec1)
        # 上采样到 H/1 以匹配 t1
        dec1_up = F.interpolate(dec1, size=t1.shape[2:], mode='bilinear', align_corners=True)
        out1 = torch.add(dec1_up, t1)

        # --- Final Output ---
        # out1 已经是 (B, C, H, W) 了，直接过 Final Conv
        out0 = self.final[task_id](out1)
        # 再次强制对齐到输入 x 的尺寸 (处理 padding 可能导致的边缘像素差异)
        out0 = F.interpolate(out0, size=x.shape[2:], mode='bilinear', align_corners=True)

        # ===================== Deep Supervision =====================
        if self.deep_supervision and self.training:
            ds4 = F.interpolate(self.ds_out4[task_id](out4), size=x.shape[2:], mode='bilinear', align_corners=True)
            ds3 = F.interpolate(self.ds_out3[task_id](out3), size=x.shape[2:], mode='bilinear', align_corners=True)
            ds2 = F.interpolate(self.ds_out2[task_id](out2), size=x.shape[2:], mode='bilinear', align_corners=True)
            ds1 = F.interpolate(self.ds_out1[task_id](out1), size=x.shape[2:], mode='bilinear', align_corners=True)
            return out0, [ds4, ds3, ds2, ds1]

        return out0
