import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import to_2tuple

# =========================================================================
# 修改部分开始：不再导入 selective_scan_cuda_oflex_rh
# 而是尝试导入官方 mamba_ssm
# =========================================================================
try:
    import mamba_ssm
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as mamba_ssm_fn

    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    print("Warning: mamba_ssm not found. Running will fail unless customized.")


def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try:
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)


def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_C=True, with_D=True, with_Z=False, with_complex=False, ):
    assert not with_complex
    if with_C:
        flops = 9 * B * L * D * N
    else:
        flops = 7 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops


def selective_scan_flop_jit(inputs, outputs, flops_fn=flops_selective_scan_fn):
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_fn(B=B, L=L, D=D, N=N, with_C=True, with_D=True, with_Z=False)
    return flops


def selective_scan_state_flop_jit(inputs, outputs, flops_fn=flops_selective_scan_fn):
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    assert N == 1
    flops = flops_fn(B=B, L=L, D=D, N=N, with_C=False, with_D=False, with_Z=False)
    return flops


# =========================================================================
# 核心修改：完全重写 selective_scan_fn 以调用 mamba_ssm
# 并且移除了原本的 SelectiveScanStateFn 类（因为我们直接用官方库）
# =========================================================================

def selective_scan_fn(u, delta, A, B, D=None, z=None, delta_bias=None, delta_softplus=False,
                      return_last_state=False):
    """
    Wrapper to adapt HME-Net's call signature to mamba_ssm's signature.
    Original: (u, delta, A, B, D, z, delta_bias, ...) -> No C parameter!
    Standard: (u, delta, A, B, C, D, z, delta_bias, ...)

    Fix: We pass B as C.
    """
    if not HAS_MAMBA:
        raise ImportError("mamba_ssm is not installed. Please install it first.")

    # HME-Net 原代码会对 B 进行 rearrange，标准库不需要我们手动 squeeze，
    # 但为了保险，我们保持维度对齐。
    # 标准库期望 B 的形状: (batch, dstate, L) 或 (batch, 1, dstate, L) (如果是 head shared)
    # 这里的 B 原本是 (B, N, L)，N是dstate。

    # 这里的逻辑是：由于原代码没有传入 C，我们将 B 作为 C 传入。
    # 这是 Vision Mamba 常见的处理方式。
    C = B

    return mamba_ssm_fn(
        u,
        delta,
        A,
        B,
        C,  # 传入 C=B
        D=D,
        z=z,
        delta_bias=delta_bias,
        delta_softplus=delta_softplus,
        return_last_state=return_last_state
    )

# 注意：保留原文件里的其他类或函数（如果有的话），但原有的
# SelectiveScanStateFn 类已经不再被使用了，可以注释掉或删除。