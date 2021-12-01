import torch
import math
import norm_dist_cuda as _C


class NormDistF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, group, p, need_grad, tag):
        output = torch.empty(x.size(0), weight.size(0), x.size(2), device=x.device)
        assert weight.size(1) * group == x.size(1)
        ctx.group = group
        ctx.p = p
        ctx.tag = tag
        if math.isinf(p):
            if not need_grad:
                _C.inf_dist_forward_nograd(x, weight, output, group)
            else:
                pos = torch.empty_like(output, dtype=torch.int)
                _C.inf_dist_forward(x, weight, output, pos, group)
                ctx.save_for_backward(x, weight, pos)
        elif p > 0:
            _C.norm_dist_forward(x, weight, output, group, p)
            ctx.save_for_backward(x, weight, output)
        else:
            raise NotImplementedError
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_weight = None
        grad_output = grad_output.contiguous()
        if math.isinf(ctx.p):
            x, weight, pos = ctx.saved_tensors
            grad_input = torch.empty_like(x)
            if ctx.needs_input_grad[1]:
                grad_weight = torch.empty_like(weight)
                _C.inf_dist_backward_input_weight(grad_output, pos, grad_input, grad_weight, ctx.group)
            elif ctx.needs_input_grad[0]:
                _C.inf_dist_backward_input(grad_output, pos, grad_input, ctx.group)
        else:
            x, weight, output = ctx.saved_tensors
            grad_input = torch.empty_like(x)
            if ctx.needs_input_grad[1]:
                grad_weight = torch.empty_like(weight)
                _C.norm_dist_backward_input_weight(grad_output, x, weight, output, grad_input, grad_weight,
                                                   ctx.group, ctx.p)
            elif ctx.needs_input_grad[0]:
                _C.norm_dist_backward_input(grad_output, x, weight, output, grad_input, ctx.group, ctx.p)
        if not ctx.needs_input_grad[0]:
            grad_input = None
        return grad_input, grad_weight, None, None, None, None, None


class BoundInfDistF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_lower, x_upper, weight, group, need_grad, tag):
        assert x_lower.size() == x_upper.size()
        assert weight.size(1) * group == x_lower.size(1)
        y_lower = torch.empty(x_lower.size(0), weight.size(0), x_lower.size(2), device=x_lower.device)
        y_upper = torch.empty_like(y_lower)
        ctx.group = group
        ctx.tag = tag
        if not need_grad:
            _C.bound_inf_dist_forward_nograd(x_lower, x_upper, weight, y_lower, y_upper, group)
        else:
            pos_lower = torch.empty_like(y_lower, dtype=torch.int)
            pos_upper = torch.empty_like(pos_lower)
            _C.bound_inf_dist_forward(x_lower, x_upper, weight, y_lower, y_upper, pos_lower, pos_upper, group)
            ctx.save_for_backward(x_lower, x_upper, weight, pos_lower, pos_upper)
        return y_lower, y_upper

    @staticmethod
    def backward(ctx, grad_y_lower, grad_y_upper):
        grad_weight = None
        grad_y_lower = grad_y_lower.contiguous()
        grad_y_upper = grad_y_upper.contiguous()
        x_lower, x_upper, weight, pos_lower, pos_upper = ctx.saved_tensors
        grad_x_lower = torch.zeros_like(x_lower)
        grad_x_upper = torch.zeros_like(x_upper)
        if ctx.needs_input_grad[2]:
            grad_weight = torch.zeros_like(weight)
            _C.bound_inf_dist_backward_input_weight(grad_y_lower, grad_y_upper, pos_lower, pos_upper,
                                                    grad_x_lower, grad_x_upper, grad_weight, ctx.group)
        elif ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            _C.bound_inf_dist_backward_input(grad_y_lower, grad_y_upper, pos_lower, pos_upper,
                                             grad_x_lower, grad_x_upper, ctx.group)
        if not ctx.needs_input_grad[0]:
            grad_x_lower = None
        if not ctx.needs_input_grad[1]:
            grad_x_upper = None
        return grad_x_lower, grad_x_upper, grad_weight, None, None, None, None, None


def norm_dist(input, weight, p, groups=1, use_custom_cuda_func=True, tag=None):
    if use_custom_cuda_func:
        need_grad = torch.is_grad_enabled() and (input.requires_grad or weight.requires_grad)
        y = NormDistF.apply(input, weight, groups, p, need_grad, tag)
    else:
        y = input.view(input.size(0), groups, 1, -1, input.size(2)) - weight.view(groups, -1, weight.size(-1), 1)
        with torch.no_grad():
            normalize = torch.norm(y, dim=3, p=float('inf'), keepdim=True)
        y = torch.norm(y / normalize, dim=3, p=p, keepdim=True) * normalize
        y = y.view(y.size(0), -1, y.size(-1))
    return y


def bound_inf_dist(x_lower, x_upper, weight, groups=1, use_custom_cuda_func=True, tag=None):
    if use_custom_cuda_func:
        need_grad = torch.is_grad_enabled() and (x_lower.requires_grad or x_upper.requires_grad or weight.requires_grad)
        y_lower, y_upper = BoundInfDistF.apply(x_lower, x_upper, weight, groups, need_grad, tag)
    else:
        w = weight.view(groups, -1, weight.size(-1), 1)
        x1 = w - x_lower.view(x_lower.size(0), groups, 1, -1, x_lower.size(2))
        x2 = x_upper.view(x_upper.size(0), groups, 1, -1, x_upper.size(2)) - w
        z1 = torch.maximum(x1, x2).clamp(min=1e-10)
        z2 = torch.minimum(x1, x2).clamp(max=-1e-10)
        y_upper = torch.norm(z1, dim=3, p=float('inf'), keepdim=True)
        y_lower = torch.norm(-z2, dim=3, p=float('inf'), keepdim=True)
        y_upper = y_upper.view(x_lower.size(0), -1, x_lower.size(-1))
        y_lower = y_lower.view(x_lower.size(0), -1, x_lower.size(-1))
    return y_lower, y_upper
