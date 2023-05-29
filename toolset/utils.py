"""
辅助性的工具
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from typing import Any, BinaryIO, List, Optional, Tuple, Union
from types import FunctionType
import gc
from imageio.v2 import imread
from PIL import Image
from torchvision.transforms import ToTensor


def clear(x: Any = None) -> None:
    """
    清除指定变量、内存和GPU内存中的无效变量、临时变量。
    """
    x = None
    gc.collect(), torch.cuda.empty_cache()
    # 不会影响当前有效（非None）的变量。
    # gc.collect() 会清理掉Python环境中所有未被引用的对象，而 torch.cuda.empty_cache() 会清理掉PyTorch在CUDA中的缓存


def reset_seed(number: int) -> None:
    """
    设置random和torch的随机种子。

    Args:
        number: 种子数
    """
    random.seed(number), torch.manual_seed(number)
    torch.cuda.manual_seed_all(number)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
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



def visual_conv(Conv):
    """演示卷积操作对图像处理的影响。

    Args:
    Conv (module): 一个执行卷积操作的模块。

    使用两个图像（小猫和小狗）进行处理，演示卷积滤波器的使用。
    """
    
    # 导入小猫和小狗的图片
    kitten_url = 'https://web.eecs.umich.edu/~justincj/teaching/eecs498/assets/a3/kitten.jpg'
    puppy_url = 'https://web.eecs.umich.edu/~justincj/teaching/eecs498/assets/a3/puppy.jpg'
    kitten = imread(kitten_url)
    puppy = imread(puppy_url)
    
    # 小猫图片较宽，裁剪成正方形
    d = kitten.shape[1] - kitten.shape[0]
    kitten_cropped = kitten[:, d // 2:-d // 2, :]

    # 设置图像的大小
    img_size = 200
    # 将原始图片的大小调整为 img_size，并将其转换为张量
    resized_puppy = ToTensor()(Image.fromarray(puppy).resize((img_size, img_size)))
    resized_kitten = ToTensor()(Image.fromarray(
        kitten_cropped).resize((img_size, img_size)))
    x = torch.stack([resized_puppy, resized_kitten])

    # 创建两个3x3的卷积滤波器
    w = torch.zeros(2, 3, 3, 3, dtype=x.dtype)

    # 第一个滤波器将图像转换为灰度图，设定红色，绿色和蓝色通道的权重
    w[0, 0, :, :] = torch.tensor([[0, 0, 0], [0, 0.3, 0], [0, 0, 0]])
    w[0, 1, :, :] = torch.tensor([[0, 0, 0], [0, 0.6, 0], [0, 0, 0]])
    w[0, 2, :, :] = torch.tensor([[0, 0, 0], [0, 0.1, 0], [0, 0, 0]])

    # 第二个滤波器在蓝色通道上检测水平边缘
    w[1, 2, :, :] = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # 偏置向量，对于灰度滤波器我们不需要偏置，但是对于边缘检测滤波器我们需要加上128以保证输出非负
    b = torch.tensor([0, 128], dtype=x.dtype)

    # 执行卷积操作，stride设置为1，pad设置为1
    out, _ = Conv.forward(x, w, b, {'stride': 1, 'pad': 1})

    def imshow_no_ax(img, normalize=True):
        """用于显示图像的小工具函数，将图像显示为uint8类型并移除坐
        Args:
        img (tensor): 待显示的图像。
        normalize (bool): 是否需要正则化图像。
        """
        if normalize:
            img_max, img_min = img.max(), img.min()
            img = 255.0 * (img - img_min) / (img_max - img_min)
        plt.imshow(img)
        plt.gca().axis('off')

    # 显示原始图像和卷积操作的结果
    plt.subplot(2, 3, 1)
    imshow_no_ax(puppy, normalize=False)
    plt.title('Original image')
    plt.subplot(2, 3, 2)
    imshow_no_ax(out[0, 0])
    plt.title('Grayscale')
    plt.subplot(2, 3, 3)
    imshow_no_ax(out[0, 1])
    plt.title('Edges')
    plt.subplot(2, 3, 4)
    imshow_no_ax(kitten_cropped, normalize=False)
    plt.subplot(2, 3, 5)
    imshow_no_ax(out[1, 0])
    plt.subplot(2, 3, 6)
    imshow_no_ax(out[1, 1])
    plt.show()
    clear()


