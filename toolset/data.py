import os
import random
from typing import Tuple
import torch
import torchvision
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from toolset import utils
import matplotlib.pyplot as plt

def tensor_to_imggrid_show(X) ->list:
    classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    
    inverse_norm = transforms.Compose(
        [
            transforms.Normalize(mean=[0., 0., 0.], std=[1 / 0.4914, 1 / 0.4822, 1 / 0.4465]),
            transforms.Normalize(mean=[-0.247, -0.243, -0.261], std=[1., 1., 1.]),
        ]
    )
    
    utils.reset_seed(0)
    N = X.shape[0]
    
    img = torchvision.utils.make_grid(X, nrow=N)
    img = inverse_norm(img)
    plt.axis("off")
    plt.imshow(utils.tensor_to_image(img))
    plt.show()
    return

    for cls, name in enumerate(classes):
        # 手工调试，让classes name 处于一个较好的位置
        plt.text(-4, 34 * cls + 18, name, ha="right")  # ha=right让文字右对齐
        # @ 获取y_train中分类是cls的所有样本的下标（沿着N）
        # (y_train == cls) 返回一个布尔张量，元素值为 True 的位置对应 `y_train` 中等于 `cls` 的元素
        # .nonzero(as_tuple=True) 函数会返回一个包裹张量的元组，包含y_train的维度个张量（这里是1个），
        # 其中张量的值对应输入中非零元素（即布尔张量中为 True 的元素）的索引。
        # (idxs, ) 则将这个元组解包，得到一个包含所有 `y_train` 中等于 `cls` 的元素的索引的张量, shape (k, ), k<=N
        (idxs,) = (y == cls).nonzero(as_tuple=True)
        # 抽取sample_per_class个图片放入samples
        for _ in range(sample_per_class):
            idx = idxs[random.randrange(idxs.shape[0])].item()  # 随机抽取一个idx
            samples.append(X[idx])
    # 让多张tensor图片合并为一张tensor图片，以grid的形式展现
    img = torchvision.utils.make_grid(samples, nrow=sample_per_class)  # nrow是一行展示的图片数量
    # plt绘图，使用image格式((H, W, 3)的ndarray)
    plt.imshow(utils.tensor_to_image(img))
    plt.axis("off")
    plt.show()

    

def _extract_tensors(dset, num=None, x_dtype=torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将CIFAR10对象转为tensor返回。

    Args:
        dset: torcvision.datasets.CIFAR10对象
        num: [可选] 需要的样本数量
        x_dtype: [可选] 将图像转换成的数据格式，默认float32

    Returns:
        x: x_dtype类型的tensor，值的范围[0,1]，shape(N,3,32,32)
        y：int64 tensor，shape(N,)
    """
    x = torch.tensor(dset.data, dtype=x_dtype).permute(0, 3, 1, 2).div_(255)  # 使用div_将会在原地执行,节省内存
    y = torch.tensor(dset.targets, dtype=torch.int64)

    if num is not None:
        if num <= 0 or num > x.shape[0]:
            raise ValueError(f"无效num = {num},应该在[0, {x.shape[0]}]之间")
        x = x[:num]
        y = y[:num]
    return x, y


def cifar10(num_train=None, num_test=None, x_dtype=torch.float32) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    返回CIFAR10数据集的tensor形式，数据集下载到当前目录。
    首先尝试自动下载，有则直接加载。可以返回数据集的一部分。
    数据范围是浮点[0,1]。
    Args:
        num_train: [可选] 选择训练集的样本数，没有则返回全部
        num_test: [可选] 选择测试集的样本数，没有则返回全部
        x_dtype: [可选] 输入图像数据的格式

    Returns:
        x_train: x_dtype tensor, shape(num_train, 3, 32, 32)
        y_train: int64 tensor, shape(num_train, 3, 32, 32)
        x_test: x_dtype tensor, shape(num_test, 3, 32, 32)
        y_test: int64 tensor, shape(num_test, 3, 32, 32)
    """
    # 判断当前目录下是否存在名为 "cifar-10-batches-py" 的文件夹，存在则不需要下载
    download = not os.path.isdir("cifar-10-batches-py")

    # 下载训练集。如果已经存在则download=False，不会重新下载
    dset_train = CIFAR10(root=".", download=download, train=True)

    # 加载测试集。注意，这里没有下载选项，因为我们假定如果需要下载的话，上一行代码已经下载过了
    dset_test = CIFAR10(root=".", train=False)

    # 使用_extract_tensors函数将数据转换为tensors
    x_train, y_train = _extract_tensors(dset_train, num_train, x_dtype)
    x_test, y_test = _extract_tensors(dset_test, num_test, x_dtype)

    return x_train, y_train, x_test, y_test


def preprocess_cifar10(
        cuda=True,
        show_examples=True,
        bias_trick=False,
        flatten=True,
        validation_ratio=0.2,
        dtype=torch.float32,
        tpu=None
):
    r"""
    返回预处理后的CIFAR10tensor数据集、以字典的格式给出，如果需要的话会自动下载。我们执行以下步骤：

    (0) [可选] 可视化数据集中的一些图像
    (1) 通过减去均值对数据进行归一化处理
    (2) 将每个形状为(3，32，32)的图像重塑为形状为(3072，)的向量
    (3) [可选] 偏置技巧：在数据中添加一个额外的维度，值为1
    (4) 从训练集中划分出验证集
    Args:
        cuda: 如果为True，将整个数据集移动到GPU上
        show_examples: 布尔值，是否可视化数据样本
        bias_trick: 布尔值，指示是否应用偏置技巧
        flatten: 布尔值，是否将图像扁平化为向量
        validation_ratio: 介于(0, 1)之间的浮点数，表示验证集占xx的比例
        dtype: 输入图像X的数据类型

    Returns:
        返回一个字典，下面是键值:
        X_train: 形状为(N_train, D)的dtype张量，表示训练图像
        X_val: 形状为(N_val, D)的dtype张量，表示验证图像
        X_test: 形状为(N_test, D)的dtype张量，表示测试图像
        y_train: 形状为(N_train, )的int64张量，表示训练标签
        y_val: 形状为(N_val, )的int64张量，表示验证标签
        y_test: 形状为(N_test, )的int64张量，表示测试标签
    """
    X_train, y_train, X_test, y_test = cifar10(x_dtype=dtype)

    # 转向GPU
    if tpu is not None:
        X_train = X_train.to(tpu)
        y_train = y_train.to(tpu)
        X_test = X_test.to(tpu)
        y_test = y_test.to(tpu)
    elif cuda:
        X_train = X_train.cuda()
        y_train = y_train.cuda()
        X_test = X_test.cuda()
        y_test = y_test.cuda()

    # (0) 可视化一些例子
    if show_examples:
        classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        sample_per_class = 12
        samples = []
        utils.reset_seed(0)

        plt.text(31*sample_per_class, 31*sample_per_class, "AI Turing", ha="right")  # ha=right让文字右对齐
        
        for cls, name in enumerate(classes):
            # 手工调试，让classes name 处于一个较好的位置
            plt.text(-4, 34 * cls + 18, name, ha="right")  # ha=right让文字右对齐
            # @ 获取y_train中分类是cls的所有样本的下标（沿着N）
            # (y_train == cls) 返回一个布尔张量，元素值为 True 的位置对应 `y_train` 中等于 `cls` 的元素
            # .nonzero(as_tuple=True) 函数会返回一个包裹张量的元组，包含y_train的维度个张量（这里是1个），
            # 其中张量的值对应输入中非零元素（即布尔张量中为 True 的元素）的索引。
            # (idxs, ) 则将这个元组解包，得到一个包含所有 `y_train` 中等于 `cls` 的元素的索引的张量, shape (k, ), k<=N
            (idxs,) = (y_train == cls).nonzero(as_tuple=True)
            # 抽取sample_per_class个图片放入samples
            for _ in range(sample_per_class):
                idx = idxs[random.randrange(idxs.shape[0])].item()  # 随机抽取一个idx
                samples.append(X_train[idx])

        # 让多张tensor图片合并为一张tensor图片，以grid的形式展现
        img = torchvision.utils.make_grid(samples, nrow=sample_per_class)  # nrow是一行展示的图片数量
        # plt绘图，使用image格式((H, W, 3)的ndarray)
        plt.imshow(utils.tensor_to_image(img))
        plt.axis("off")
        plt.show()

    # (1) 通过减去均值对数据进行归一化处理
    # 这个过程就像是在一个三维的像素空间中，首先将所有图像叠加在一起（按 N 维度压缩），得到一个平均的“立体图像”。
    # 然后在这个平均的“立体图像”中，我们按照高度（H 维度）和宽度（W 维度）将所有的像素值压缩成一个平均值，
    # 得到一个在每个颜色通道上都有一个平均值的“点”（C 维度的向量, 这里就是3维度）。
    
    # CIFAR-10的预计算均值和标准差
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=X_train.device).view(1, 3, 1, 1)
    std = torch.tensor([0.247, 0.243, 0.261],  device=X_train.device).view(1, 3, 1, 1)
    # 归一化
    X_train.sub_(mean).div_(std)
    X_test.sub_(mean).div_(std)



    # (2) 将形状为(N, C, H, W)的张量重塑为(N, C*H*W)的向量
    if flatten:
        X_train = X_train.reshape(X_train.shape[0], -1)  # (N, 3072) 留了一些扩展性
        X_test = X_test.reshape(X_test.shape[0], -1)

    # (3) bias trick，在数据中添加一个额外的维度，把偏置塞进图像与标签 (之后权重也要加一个)
    if bias_trick:
        ones_train = torch.ones(X_train.shape[0], 1, device=X_train.device)  # (N, 1)
        X_train = torch.cat([X_train, ones_train], dim=1)  # (N, 3072)-(N,1) -> (N, 3073)
        ones_test = torch.ones(X_test.shape[0], 1, device=X_test.device)
        X_test = torch.cat([X_test, ones_test], dim=1)

    # (4) 从训练集中划分出验证集
    # 随机抽样使用torch.randperm或torch.randint
    # 不过这里采用slicing
    num_validation = int(X_train.shape[0] * validation_ratio)
    num_training = X_train.shape[0] - num_validation

    # 返回数据集
    data_dict = {'X_val': X_train[num_training:num_training + num_validation],
                 'y_val': y_train[num_training:num_training + num_validation],
                 'X_train': X_train[:num_training],
                 'y_train': y_train[:num_training],
                 'X_test': X_test,
                 'y_test': y_test}

    return data_dict
