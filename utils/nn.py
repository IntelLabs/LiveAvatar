from __future__ import annotations
import numpy as np
import torch


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> torch.nn.Conv2d:
    """3x3 convolution with padding"""
    return torch.nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> torch.nn.Conv2d:
    """1x1 convolution"""
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def set_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


def to_numpy(ft: torch.Tensor | None) -> np.ndarray | None:
    if ft is None:
        return None
    if isinstance(ft, np.ndarray):
        return ft
    try:
        return ft.detach().cpu().numpy()
    except AttributeError:
        pass
    return np.array(ft)


def unsqueeze(x):
    if isinstance(x, np.ndarray):
        return x[np.newaxis, ...]
    else:
        return x.unsqueeze(dim=0)


def atleast4d(x):
    if x is None:
        return x
    if len(x.shape) == 3:
        return unsqueeze(x)
    return x


def atleast3d(x):
    if x is None:
        return x
    if len(x.shape) == 2:
        return unsqueeze(x)
    return x
