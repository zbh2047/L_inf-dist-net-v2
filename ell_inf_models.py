import math
import torch
import torch.nn as nn
from core.modules import NormDistConv, NormDist, NormDistBase
from core.modules import BoundTanh, BoundLinear, BoundFinalLinear, BoundSequential
from core.modules import apply_if_not_none


def check_inf_and_eval(model):
    for m in model.modules():
        if isinstance(m, NormDistBase) and (not math.isinf(m.p) or m.training):
            return False
    return True


class MLPModel(nn.Module):
    def __init__(self, depth, width, input_dim, num_classes=10,
                 std=1.0, identity_val=None, scalar=False):
        super(MLPModel, self).__init__()
        dist_kwargs = {'std': std, 'identity_val': identity_val}
        pixels = input_dim[0] * input_dim[1] * input_dim[2]
        fc_dist = []
        fc_dist.append(NormDist(pixels, width, bias=False, mean_shift=True, **dist_kwargs))
        for i in range(depth - 2):
            fc_dist.append(NormDist(width, width, bias=False, mean_shift=True, **dist_kwargs))
        fc_dist.append(NormDist(width, num_classes, bias=True, mean_shift=False, **dist_kwargs))
        self.fc_dist = BoundSequential(*fc_dist)
        self.scalar = nn.Parameter(torch.ones(1)) if scalar else 1

    def forward(self, x=None, targets=None, eps=0, up=None, down=None):
        if up is not None and down is not None and check_inf_and_eval(self):  # certification
            paras = (x, torch.maximum(x - eps, down), torch.minimum(x + eps, up))
        else:
            paras = (x, None, None)
        paras = apply_if_not_none(paras, lambda z: z.view(z.size(0), -1))
        paras = self.fc_dist(paras)
        x = paras[0]
        if targets is None:
            return -x * self.scalar
        else:
            lower = x - eps if paras[1] is None else paras[1]
            upper = x + eps if paras[2] is None else paras[2]
            x, lower, upper = -x, -upper, -lower
            margin = upper - torch.gather(lower, 1, targets.view(-1, 1))
            margin = margin.scatter(1, targets.view(-1, 1), 0)
            return x * self.scalar, margin / (2 * eps)


class HybridModel(nn.Module):
    def __init__(self, depth, width, input_dim, hidden=512, num_classes=10, std=1.0, identity_val=None):
        super(HybridModel, self).__init__()
        dist_kwargs = {'std': std, 'identity_val': identity_val}
        pixels = input_dim[0] * input_dim[1] * input_dim[2]
        fc_dist = []
        fc_linear = []
        fc_dist.append(NormDist(pixels, width, bias=False, mean_shift=True, **dist_kwargs))
        for i in range(depth - 3):
            fc_dist.append(NormDist(width, width, bias=False, mean_shift=True, **dist_kwargs))
        fc_linear.append(BoundLinear(width, hidden, bias=True))
        fc_linear.append(BoundTanh())
        self.fc_dist = BoundSequential(*fc_dist)
        self.fc_linear = BoundSequential(*fc_linear)
        self.fc_final = BoundFinalLinear(hidden, num_classes, bias=True)

    def forward(self, x=None, targets=None, eps=0, up=None, down=None):
        if up is not None and down is not None and check_inf_and_eval(self):  # certification
            paras = (x, torch.maximum(x - eps, down), torch.minimum(x + eps, up))
        else:
            paras = (x, None, None)
        paras = apply_if_not_none(paras, lambda z: z.view(z.size(0), -1))
        paras = self.fc_dist(paras)
        if targets is not None and (paras[1] is None or paras[2] is None):
            paras = paras[0], paras[0] - eps, paras[0] + eps
        paras = self.fc_linear(paras)
        return self.fc_final(*paras, targets=targets)

