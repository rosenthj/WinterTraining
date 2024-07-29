import torch
import torch.nn as nn


class SqCReLU(nn.Module):
    def __init__(self, clip_min=0, clip_max=1):
        super(SqCReLU, self).__init__()
        self.clip_min = clip_min
        self.clip_max = clip_max

    def forward(self, x):
        return torch.pow(torch.clip(x, self.clip_min, self.clip_max), 2.0)


class SqCReLUvSF(nn.Module):
    def __init__(self, clip_min=0, clip_max=1):
        super(SqCReLUvSF, self).__init__()
        self.clip_min = clip_min
        self.clip_max = clip_max

    def forward(self, x):
        return torch.clip(torch.pow(x, 2.0), self.clip_min, self.clip_max)


class MishSqrt(nn.Module):
    def __init__(self):
        super(MishSqrt, self).__init__()

    def forward(self, x):
        return torch.sqrt(nn.functional.mish(x)+0.30885) - 0.30885
