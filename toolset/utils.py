"""
各种辅助性的实用工具
"""

import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import random
import math
from typing import Any, BinaryIO, List, Optional, Tuple, Union
from types import FunctionType

def reset_seed(number:int) -> None:
    """
    设置random和torch的随机种子。

    Args:
        number: 种子数
    """
    random.seed(number),torch.manual_seed(number)
    torch.cuda.manual_seed_all(number)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return

def tensor_to_image(tensor:torch.Tensor) -> np.ndarray:
    """
    将tensor转换为ndarray并修改维度顺序，用来可视化。不修改源数据。

    Args:
        tensor: 形态为(3, H, W)，值的范围为[0, 1] 的torch tensor

    Returns:
        ndarr: uint8类型ndarray, shape (H, W, 3)
    """

    # 转换到[0, 255], add_(0.5)用来四舍五入
    x = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)

    return x.to("cpu", torch.uint8).numpy()
