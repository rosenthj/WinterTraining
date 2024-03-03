import torch
import torch.nn as nn


class SqCReLU(nn.Module):
    def __init__(self, clip_min=0, clip_max=1):
        super(SqCReLU, self).__init__()
        self.clip_min = clip_min
        self.clip_max = clip_max

    def forward(self, x):
        x = torch.clip(x, self.clip_min, self.clip_max)
        return x * x


class MishSqrt(nn.Module):
    def __init__(self):
        super(MishSqrt, self).__init__()

    def forward(self, x):
        return torch.sqrt(nn.funtional.mish(x)+0.30885) - 0.30885
