import numpy as np
import torch
from torch import optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import math
import random
import warnings
import copy
import torch
import torch.nn as nn
from functools import partial
from math import sqrt
from torch.nn  import Parameter
import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as weight_norm
from torch.nn import Parameter
import math
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as weight_norm
from torch.nn import Parameter
import math
import torch.nn.functional as F


class TPReLU(nn.Module):

    def __init__(self, num_parameters=1, init=0.25):
        self.num_parameters = num_parameters
        super(TPReLU, self).__init__()
        self.weight = Parameter(torch.Tensor(num_parameters).fill_(init))
        self.bias = Parameter(torch.zeros(num_parameters))

    def forward(self, input):
        bias_resize = self.bias.view(1, self.num_parameters, *((1,) * (input.dim() - 2))).expand_as(input)
        return F.prelu(input - bias_resize, self.weight.clamp(0, 1)) + bias_resize

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.num_parameters) + ')'
class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


    # average Batch Normalization's statistics (Should we stick with EMA for BacthNorm?)

def to_canonical(data, rev=False):
    n_dim = data.shape[1]
    p = torch.zeros_like(data)
    if rev:
        p[:, :, 0] = torch.arctanh(data[:, :, 2]/ torch.sqrt(data[:, :, 0] ** 2 + data[:, :, 1] ** 2 + data[:, :, 2] ** 2))
        p[:, :, 1] = torch.atan2(data[:, :, 1], data[:, :, 0])
        p[:, :, 2] = torch.sqrt(data[:, :, 0] ** 2 + data[:, :, 1] ** 2)
        return p
    else:

        p[:, :, 0] = data[:, :, 2] * torch.cos(data[:, :, 1])
        p[:, :, 1] = data[:, :, 2] * torch.sin(data[:, :, 1])
        p[:, :, 2] = data[:, :, 2] * torch.sinh(data[:, :, 0])
        E=p[:,:,0]**2+p[:,:,1]**2+p[:,:,2]**2
        return torch.cat((p,E.reshape(data.shape[0],data.shape[1],1)),dim=-1)





def mmd(x, y, device, kernel="rbf"):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)

    dxx = rx.t() + rx - 2.0 * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2.0 * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2.0 * zz  # Used for C in (1)

    XX, YY, XY = (
        torch.zeros(xx.shape).to(device),
        torch.zeros(xx.shape).to(device),
        torch.zeros(xx.shape).to(device),
    )

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx) ** -1
            YY += a**2 * (a**2 + dyy) ** -1
            XY += a**2 * (a**2 + dxy) ** -1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

    return torch.mean(XX + YY - 2.0 * XY)

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class Scheduler(_LRScheduler):
    def __init__(self,
                 optimizer: Optimizer,
                 dim_embed: int,
                 warmup_steps: int,
                 last_epoch: int=-1,
                 verbose: bool=False) -> None:

        self.dim_embed = dim_embed
        self.warmup_steps = warmup_steps
        self.num_param_groups = len(optimizer.param_groups)

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> float:
        lr = calc_lr(self._step_count, self.dim_embed, self.warmup_steps)
        return [lr] * self.num_param_groups


def calc_lr(step, dim_embed, warmup_steps):
        return dim_embed**(-0.5) * min(step**(-0.5), step * warmup_steps**(-1.5))

class Rational(torch.nn.Module):
    """Rational Activation function.
    It follows:
    `f(x) = P(x) / Q(x),
    where the coefficients of P and Q are initialized to the best rational
    approximation of degree (3,2) to the ReLU function
    # Reference
        - [Rational neural networks](https://arxiv.org/abs/2004.01902)
    """

    def __init__(self):
        super().__init__()
        self.coeffs = torch.nn.Parameter(torch.Tensor(4, 2))
        self.reset_parameters()

    def reset_parameters(self):
        self.coeffs.data = torch.Tensor(
            [[1.1915, 0.0], [1.5957, 2.383], [0.5, 0.0], [0.0218, 1.0]]
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.coeffs.data[0, 1].zero_()
        exp = torch.tensor([3.0, 2.0, 1.0, 0.0], device=input.device, dtype=input.dtype)
        X = torch.pow(input.unsqueeze(-1), exp)
        PQ = X @ self.coeffs
        output = torch.div(PQ[..., 0], PQ[..., 1])
        return output


class WeightNormalizedLinear(nn.Module):

    def __init__(self, in_features, out_features, scale=False, bias=False, init_factor=1, init_scale=1):
        super(WeightNormalizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        if scale:
            self.scale = Parameter(torch.Tensor(out_features).fill_(init_scale))
        else:
            self.register_parameter('scale', None)

        self.reset_parameters(init_factor)

    def reset_parameters(self, factor):
        stdv = 1. * factor / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def weight_norm(self):
        return self.weight.pow(2).sum(1).sqrt().add(1e-8)

    def norm_scale_bias(self, input):
        output = input.div(self.weight_norm().unsqueeze(0))
        if self.scale is not None:
            output = output.mul(self.scale.unsqueeze(0))
        if self.bias is not None:
            output = output.add(self.bias.unsqueeze(0))
        return output

    def forward(self, input):
        return self.norm_scale_bias(F.linear(input, self.weight))

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'

def masked_layer_norm(x, mask, eps = 1e-5):
    """
    x of shape: [batch_size (N), num_objects (L), features(C)]
    mask of shape: [batch_size (N), num_objects (L)]
    """
    mask = mask.float().unsqueeze(-1)  # (N,L,1)
    mean = (torch.sum(x * mask, 1) / torch.sum(mask, 1))   # (N,C)
    mean = mean.detach()
    var_term = ((x - mean.unsqueeze(1).expand_as(x)) * mask)**2  # (N,L,C)
    var = (torch.sum(var_term, 1) / torch.sum(mask, 1))  #(N,C)
    var = var.detach()
    mean_reshaped = mean.unsqueeze(1).expand_as(x)  # (N, L, C)
    var_reshaped = var.unsqueeze(1).expand_as(x)    # (N, L, C)
    ins_norm = (x - mean_reshaped) / torch.sqrt(var_reshaped + eps)   # (N, L, C)
    return ins_norm
