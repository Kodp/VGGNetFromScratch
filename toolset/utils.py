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
    kitten_url = 'https://web.eecs.umich.edu/~justincj/teaching/eecs498/assets/a3/kitten.jpg'
    puppy_url = 'https://web.eecs.umich.edu/~justincj/teaching/eecs498/assets/a3/puppy.jpg'

    kitten = imread(kitten_url)
    puppy = imread(puppy_url)
    # kitten is wide, and puppy is already square
    d = kitten.shape[1] - kitten.shape[0]
    kitten_cropped = kitten[:, d // 2:-d // 2, :]

    img_size = 200  # Make this smaller if it runs too slow
    resized_puppy = ToTensor()(Image.fromarray(puppy).resize((img_size, img_size)))
    resized_kitten = ToTensor()(Image.fromarray(
        kitten_cropped).resize((img_size, img_size)))
    x = torch.stack([resized_puppy, resized_kitten])

    # Set up a convolutional weights holding 2 filters, each 3x3
    w = torch.zeros(2, 3, 3, 3, dtype=x.dtype)

    # The first filter converts the image to grayscale.
    # Set up the red, green, and blue channels of the filter.
    w[0, 0, :, :] = torch.tensor([[0, 0, 0], [0, 0.3, 0], [0, 0, 0]])
    w[0, 1, :, :] = torch.tensor([[0, 0, 0], [0, 0.6, 0], [0, 0, 0]])
    w[0, 2, :, :] = torch.tensor([[0, 0, 0], [0, 0.1, 0], [0, 0, 0]])

    # Second filter detects horizontal edges in the blue channel.
    w[1, 2, :, :] = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # Vector of biases. We don't need any bias for the grayscale
    # filter, but for the edge detection filter we want to add 128
    # to each output so that nothing is negative.
    b = torch.tensor([0, 128], dtype=x.dtype)

    # Compute the result of convolving each input in x with each filter in w,
    # offsetting by b, and storing the results in out.
    out, _ = Conv.forward(x, w, b, {'stride': 1, 'pad': 1})

    def imshow_no_ax(img, normalize=True):
        """ Tiny helper to show images as uint8 and remove axis labels """
        if normalize:
            img_max, img_min = img.max(), img.min()
            img = 255.0 * (img - img_min) / (img_max - img_min)
        plt.imshow(img)
        plt.gca().axis('off')

    # Show the original images and the results of the conv operation
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


