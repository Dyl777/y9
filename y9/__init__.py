"""
y9 - PyTorch-like interface for C backend with autograd
"""
from .pytorch_interface import (
    Tensor,
    Module,
    Linear,
    Conv2d,
    MaxPool2d,
    AvgPool2d,
    LayerNorm,
    MultiheadAttention,
    FeedForward,
    Sequential,
    relu,
    mse_loss,
    matmul,
    SGD,
    Optimizer,
)

__version__ = "0.1.0"
