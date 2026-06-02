import torch
import torch.nn as nn
from . import AWRE
import torch.nn.functional as F



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


class SpatialFrequencyDomainExperts(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, num_tasks, share_fusion=False):
        super().__init__()
        self.num_tasks = num_tasks
        self.share_fusion = share_fusion

        # 共享专家
        self.shared_expert = FFN(dim=dim, ffn_expansion_factor = ffn_expansion_factor, bias=True)

        # 任务专属专家
        self.task_experts = nn.ModuleList([
            AWRE.AdaptiveWaveletRefinementExpert(channels=dim) for _ in range(num_tasks)
        ])

    def forward(self, x, task_id, epoch=1):
        # 共享专家
        shared_out = self.shared_expert(x)

        # 任务专家
        task_out = self.task_experts[task_id](x)

        fused_out = shared_out + task_out

        output = fused_out

        return output