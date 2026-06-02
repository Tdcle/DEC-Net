from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from functools import partial
import torch
import torch.nn as nn
import math
from timm.models.layers import trunc_normal_, DropPath
import torch.nn.functional as F
from torch.nn.init import constant_
from einops import repeat

from .MoE.SFDE import SpatialFrequencyDomainExperts


from .ops_dcnv3.functions import DCNv3Function

from .utils import selective_scan_state_flop_jit, selective_scan_fn


class to_channels_first(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class to_channels_last(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)


def my_build_norm_layer(dim,
                     norm_layer,
                     in_format='channels_last',
                     out_format='channels_last',
                     eps=1e-6):
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(
            f'build_norm_layer does not support {norm_layer}')
    return nn.Sequential(*layers)


def build_act_layer(act_layer):
    if act_layer == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_layer == 'GELU':
        return nn.GELU()

    raise NotImplementedError(f'build_act_layer does not support {act_layer}')

class FFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=4, bias=True):
        super(FFN, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2, hidden_features * 2,
            kernel_size=3, stride=1, padding=1,
            groups=hidden_features * 2, bias=bias
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class ResDWC(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2, groups=dim)
        a = torch.zeros(kernel_size ** 2)
        a[4] = 1.
        self.conv_constant = nn.Parameter(a.reshape(1, 1, kernel_size, kernel_size))
        self.conv_constant.requires_grad = False
    def forward(self, x):
        return F.conv2d(x, self.conv.weight + self.conv_constant, self.conv.bias, stride=1,
                        padding=self.kernel_size // 2, groups=self.dim)  # equal to x + conv(x)


class CenterFeatureScaleModule(nn.Module):
    def forward(self,
                query,
                center_feature_scale_proj_weight,
                center_feature_scale_proj_bias):
        center_feature_scale = F.linear(query,
                                        weight=center_feature_scale_proj_weight,
                                        bias=center_feature_scale_proj_bias).sigmoid()
        return center_feature_scale


class Dynamic_Adaptive_Scan(nn.Module):
    def __init__(
            self,
            channels=64,
            kernel_size=1,
            dw_kernel_size=3,
            stride=1,
            pad=0,
            dilation=1,
            group=1,
            offset_scale=1.0,
            act_layer='GELU',
            norm_layer='LN',
            center_feature_scale=False,
            remove_center=False,
    ):
        super().__init__()
        if channels ==1:
            group = 1
        if channels % group != 0:
            raise ValueError(
                f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size
        # you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation

        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.center_feature_scale = center_feature_scale
        self.remove_center = int(remove_center)

        if self.remove_center and self.kernel_size % 2 == 0:
            raise ValueError('remove_center is only compatible with odd kernel size.')

        self.dw_conv = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=dw_kernel_size,
                stride=1,
                padding=(dw_kernel_size - 1) // 2,
                groups=channels),
            my_build_norm_layer(
                channels,
                norm_layer,
                'channels_first',
                'channels_last'),
            build_act_layer(act_layer))
        self.offset = nn.Linear(
            channels,
            group * (kernel_size * kernel_size - remove_center) * 2)
        self._reset_parameters()

        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(
                torch.zeros((group, channels), dtype=torch.float))
            self.center_feature_scale_proj_bias = nn.Parameter(
                torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group, ))
            self.center_feature_scale_module = CenterFeatureScaleModule()

    def _reset_parameters(self):
        constant_(self.offset.weight.data, 0.)
        constant_(self.offset.bias.data, 0.)


    def forward(self, input, x):
        N, _, H, W = input.shape
        x_proj = x
        x1 = input
        x1 = self.dw_conv(x1)
        offset = self.offset(x1)
        mask = torch.ones(N, H, W, self.group, device=x.device, dtype=x.dtype)
        with torch.cuda.amp.autocast(enabled=False):
            # 确保所有输入都是float32
            if x.dtype != torch.float32:
                x = x.float()
            if offset.dtype != torch.float32:
                offset = offset.float()
            if mask.dtype != torch.float32:
                mask = mask.float()

            # 调用DCNv3Function
            x = DCNv3Function.apply(
                x, offset, mask,
                self.kernel_size, self.kernel_size,
                self.stride, self.stride,
                self.pad, self.pad,
                self.dilation, self.dilation,
                self.group, self.group_channels,
                self.offset_scale,
                256,
                self.remove_center
            )

            # 转换回原始数据类型
            if x_proj.dtype != torch.float32:
                x = x.to(x_proj.dtype)

        if self.center_feature_scale:
            center_feature_scale = self.center_feature_scale_module(
                x1, self.center_feature_scale_proj_weight, self.center_feature_scale_proj_bias)
            # N, H, W, groups -> N, H, W, groups, 1 -> N, H, W, groups, _d_per_group -> N, H, W, channels
            center_feature_scale = center_feature_scale[..., None].repeat(
                1, 1, 1, 1, self.channels // self.group).flatten(-2)
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale

        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class DASSM(nn.Module):
    def __init__(
        self,
        d_model,
        head_dim=16,
        d_state=1,
        d_conv=3,
        expand=1,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj =nn.Conv2d(self.d_model, self.d_inner, 1,bias=bias, **factory_kwargs)

        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner, (self.dt_rank + self.d_state*2), bias=False, **factory_kwargs)
        self.x_proj_weight = nn.Parameter(self.x_proj.weight)
        del self.x_proj

        self.dt_projs = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
        self.dt_projs_weight = nn.Parameter(self.dt_projs.weight)
        self.dt_projs_bias = nn.Parameter(self.dt_projs.bias)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, dt_init)
        self.Ds = self.D_init(self.d_inner, dt_init)

        self.selective_scan = selective_scan_fn

        self.out_norm = LayerNorm(self.d_inner)
        self.out_proj = nn.Conv2d(self.d_inner, self.d_model, 1,bias=bias, **factory_kwargs)

        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        num_group=d_model//head_dim
        self.da_scan = Dynamic_Adaptive_Scan(channels=self.d_inner,group=num_group)
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, bias=True,**factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=bias, **factory_kwargs)

        if bias:
            # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
            dt = torch.exp(
                torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))

            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)
            # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
            dt_proj.bias._no_reinit = True

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        elif dt_init == "simple":
            with torch.no_grad():
                dt_proj.weight.copy_(0.1 * torch.randn((d_inner, dt_rank)))
                dt_proj.bias.copy_(0.1 * torch.randn((d_inner)))
                dt_proj.bias._no_reinit = True
        elif dt_init == "zero":
            with torch.no_grad():
                dt_proj.weight.copy_(0.1 * torch.rand((d_inner, dt_rank)))
                dt_proj.bias.copy_(0.1 * torch.rand((d_inner)))
                dt_proj.bias._no_reinit = True
        else:
            raise NotImplementedError

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, init, device=None):
        if init=="random" or "constant":
            # S4D real initialization
            A = repeat(
                torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=d_inner,
            ).contiguous()
            A_log = torch.log(A)
            A_log = nn.Parameter(A_log)
            A_log._no_weight_decay = True
        elif init=="simple":
            A_log = nn.Parameter(torch.randn((d_inner, d_state)))
        elif init=="zero":
            A_log = nn.Parameter(torch.zeros((d_inner, d_state)))
        else:
            raise NotImplementedError
        return A_log

    @staticmethod
    def D_init(d_inner, init="random", device=None):
        if init=="random" or "constant":
            # D "skip" parameter
            D = torch.ones(d_inner, device=device)
            D = nn.Parameter(D)
            D._no_weight_decay = True
        elif init == "simple" or "zero":
            D = nn.Parameter(torch.ones(d_inner))
        else:
            raise NotImplementedError
        return D

    def ssm(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W

        xs = x.view(B, -1, L)

        x_dbl = torch.matmul(self.x_proj_weight.view(1, -1, C), xs)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=1)
        dts = torch.matmul(self.dt_projs_weight.view(1, C, -1), dts)

        As = -torch.exp(self.A_logs)
        Ds = self.Ds
        dts = dts.contiguous()
        dt_projs_bias = self.dt_projs_bias

        h = self.selective_scan(
            xs, dts,
            As, Bs, None,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        )

        h=h.reshape(B,C,H*W)

        y = h * Cs
        y = y + xs * Ds.view(-1, 1)

        return y

    def forward(self, x: torch.Tensor):
        B, C,H, W = x.shape
        input=x
        x = self.in_proj(x)
        x = self.act(self.conv2d(x))


        x=self.da_scan(input,x.permute(0, 2,3,1).contiguous())
        y = self.ssm(x)
        y=y.reshape(B, C,H, W)

        y = self.out_norm(y)
        y = self.out_proj(y)
        if self.dropout is not None:
            y = self.dropout(y)
        return y

class Block_image(nn.Module):
    def __init__(
            self,
            dim,
            token_mixer=nn.Identity,
            head_dim=24,
            drop_path=0.,
            index=None,
            layerscale=False,
            seq_len=0,
            num_tasks=1,
    ):
        super().__init__()
        if isinstance(token_mixer, list):
            if index % 2 == 0:
                self.token_mixer = token_mixer[0](dim, head_dim=head_dim)
            elif index % 2 == 1:
                self.token_mixer = token_mixer[1](dim, head_dim=head_dim)
        else:
            self.token_mixer = token_mixer(dim, head_dim=head_dim)

        self.norm1 = nn.GroupNorm(1, dim)
        self.norm2 = nn.GroupNorm(1, dim)

        self.MoE = SpatialFrequencyDomainExperts(dim, 4, num_tasks=num_tasks)

        self.seq_len = seq_len
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        layer_scale_init_value = 1e-6
        self.layerscale = layerscale
        if layerscale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.pos_embed = ResDWC(dim, 3)

    def forward(self, x, task_id=1, epoch=1):
        x = x + self.pos_embed(x)
        if self.layerscale == False:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.MoE(self.norm2(x), task_id=task_id, epoch=epoch))
        else:
            x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.MoE(self.norm2(x), task_id=task_id, epoch=epoch))
        return x



class DAMELayer(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            depth=2,
            drop_path_rates=None,
            token_mixer=DASSM,
            head_dim=8,
            layerscale=True,
            seq_len = 0,
            num_tasks=1):
        super().__init__()
        self.grad_checkpointing = False
        self.proj = nn.Conv2d(
            in_channels=in_chs,  # 输入通道数
            out_channels=out_chs,  # 输出通道数
            kernel_size=1  # 1x1 卷积核
        )
        drop_path_rates = drop_path_rates or [0.] * depth
        stage_blocks = []

        for i in range(depth):
            stage_blocks.append(Block_image(
                dim=out_chs,
                drop_path=drop_path_rates[i],
                token_mixer=token_mixer,
                head_dim=head_dim,
                index=i,
                layerscale=layerscale,
                seq_len=seq_len,
                num_tasks=num_tasks,
            ))
            in_chs = out_chs
        self.blocks = nn.Sequential(*stage_blocks)


    def forward(self, x, task_id=1, epoch=1):
        # x = self.downsample(x)
        x = self.proj(x)
        for block in self.blocks:
            x = block(x, task_id, epoch=epoch)
        # x = self.blocks(x, task_id,epoch)
        return x



class LayerNorm(nn.Module):

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        if self.training:
            x = x.permute(0, 2, 3, 1).contiguous()
            x = self.norm(x)
            x = x.permute(0, 3, 1, 2).contiguous()
        else:
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
            x = x.permute(0, 3, 1, 2)
        return x



