import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.norm_dist import norm_dist, bound_inf_dist
from .basic_modules import MeanShift


def apply_if_not_none(paras, func):
    return [None if x is None else func(x) for x in paras]


class NormDistBase(nn.Module):
    def __init__(self, in_features, out_features, p=float('inf'), groups=1, bias=True, std=1.0, mean_shift=True):
        super(NormDistBase, self).__init__()
        assert (in_features % groups == 0)
        assert (out_features % groups == 0)
        self.weight = nn.Parameter(torch.randn(out_features, in_features // groups) * std)
        self.groups = groups
        self.p = p
        self.mean_shift = MeanShift(out_channels=out_features, affine=False) if mean_shift else None
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        if not hasattr(NormDistBase, 'tag'):
            NormDistBase.tag = 0
        NormDistBase.tag += 1
        self.tag = NormDistBase.tag

    # x, lower and upper should be 3d tensors with shape (B, C, H*W)
    def forward(self, x=None, lower=None, upper=None):
        if x is not None:
            x = norm_dist(x, self.weight, self.p, self.groups, tag=self.tag)
        if lower is not None and upper is not None:
            assert math.isinf(self.p)
            lower, upper = bound_inf_dist(lower, upper, self.weight, self.groups, tag=self.tag)
        if self.mean_shift is not None:
            x, lower, upper = self.mean_shift(x, lower, upper)
        if self.bias is not None:
            x, lower, upper = apply_if_not_none((x, lower, upper), lambda z: z + self.bias.view(1, -1, 1))
        return x, lower, upper


class NormDist(NormDistBase):
    def __init__(self, in_features, out_features, groups=1, bias=True, identity_val=None, **kwargs):
        super(NormDist, self).__init__(in_features, out_features, groups=groups, bias=bias, **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        if identity_val is not None and in_features <= out_features:
            for i in range(out_features):
                weight = self.weight.data
                weight[i, i % (in_features // groups)] = -identity_val

    def forward(self, x=None, lower=None, upper=None):
        x, lower, upper = apply_if_not_none((x, lower, upper), lambda z: z.unsqueeze(-1))
        x, lower, upper = super(NormDist, self).forward(x, lower, upper)
        x, lower, upper = apply_if_not_none((x, lower, upper), lambda z: z.squeeze(-1))
        return x, lower, upper

    def extra_repr(self):
        s = 'in_features={}, out_features={}, bias={}'
        if self.groups != 1:
            s += ', groups={groups}'
        return s.format(self.in_features, self.out_features, self.bias is not None, groups=self.groups)


class NormDistConv(NormDistBase):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, identity_val=None, **kwargs):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        in_features = in_channels * kernel_size[0] * kernel_size[1]
        assert (in_channels % groups == 0)
        super(NormDistConv, self).__init__(in_features, out_channels, groups=groups, bias=bias, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        if identity_val is not None and in_channels <= out_channels:
            for i in range(out_channels):
                weight = self.weight.data.view(out_channels, -1, *kernel_size)
                weight[i, i % (in_channels // groups), kernel_size[0] // 2, kernel_size[1] // 2] = -identity_val

    def forward(self, x=None, lower=None, upper=None):
        unfold_paras = self.kernel_size, self.dilation, self.padding, self.stride
        h, w = 0, 0
        if x is not None:
            h, w = x.size(2), x.size(3)
            x = F.unfold(x, *unfold_paras)
        if lower is not None and upper is not None:
            h, w = lower.size(2), lower.size(3)
            lower = F.unfold(lower, *unfold_paras)
            upper = F.unfold(upper, *unfold_paras)
        x, lower, upper = super(NormDistConv, self).forward(x, lower, upper)
        h, w = [(s + 2 * self.padding - k) // self.stride + 1 for s, k in zip((h, w), self.kernel_size)]
        x, lower, upper = apply_if_not_none((x, lower, upper), lambda z: z.view(z.size(0), -1, h, w))
        return x, lower, upper

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != 0:
            s += ', padding={padding}'
        if self.dilation != 1:
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

