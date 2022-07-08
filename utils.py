import torch
import torch.nn as nn
import cv2
from torchvision import transforms
from torchvision import models
from torch.utils.checkpoint import checkpoint, checkpoint_sequential


class Reshape(nn.Module):
    def __init__(self, shape=None):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        if self.shape is None:
            x = x.view(x.size(0), -1)
        else:
            x = x.view(x.size(0), *self.shape)
        return x

def num_params(model):
    return sum(x.numel() for x in model.parameters())


class FakeFn(nn.Module):
    def __init__(self, fn=lambda x: x):
        super(FakeFn, self).__init__()
        self.fn = fn

    def forward(self, *x):
        return self.fn(*x)


class DummyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.ones(1, dtype=torch.float32))

    def forward(self,x):
        return x + self.dummy - self.dummy