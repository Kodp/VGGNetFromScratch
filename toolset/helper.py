import torch
import torchvision
from toolset import data
import matplotlib.pyplot as plt
import random
import math
from toolset.solver import Solver
from PIL import Image
import torchvision.transforms as transforms
import gc


def pil_to_tensor(image: Image, image_type: str) -> torch.Tensor:
    if image_type == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))   # mean, std for MNIST
        ])
    elif image_type == 'CIFAR':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # mean, std for CIFAR-10
        ])
    else:
        raise ValueError(f"Invalid image_type: {image_type}")

    return transform(image)



def get_mnist_data():
    """
    加载mnist数据集(全部)。

    Returns:
        train_data: tensor，训练数据
        train_labels: tensor，训练标签
        test_data: tensor，测试数据
        test_labels: tensor，测试标签
    """
    # 符合mnist的转换，将数据集转换为均值为0、方差为1的分布。
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),   # mean， std
    ])
    
    MNIST_train_set = torchvision.datasets.MNIST(
    "mnist_dataset/", train=True, download=True, transform=transform
    )
    MNIST_test_set = torchvision.datasets.MNIST(
        "mnist_dataset/", train=False, download=True, transform=transform
    )
    
    train_loader = torch.utils.data.DataLoader(MNIST_train_set, batch_size=len(MNIST_train_set), shuffle=True)
    test_loader = torch.utils.data.DataLoader(MNIST_test_set, batch_size=len(MNIST_test_set), shuffle=True)

    train_data, train_labels = next(iter(train_loader))
    test_data, test_labels = next(iter(test_loader))
    
    return train_data, train_labels, test_data, test_labels


def get_CIFAR10_data(validataion_ratio=0.02, flatten=False):
    """
    从磁盘加载CIFAR10数据。
    将使用cuda，展示图片。
    Args:
        validataion_ratio: 验证集的比例
        flatten: 是否将图像展平

    Returns:
        返回一个字典，下面是键值:
        X_train: 形状为(N_train, D)的dtype张量，表示训练图像
        X_val: 形状为(N_val, D)的dtype张量，表示验证图像
        X_test: 形状为(N_test, D)的dtype张量，表示测试图像
        y_train: 形状为(N_train, )的int64张量，表示训练标签
        y_val: 形状为(N_val, )的int64张量，表示验证标签
        y_test: 形状为(N_test, )的int64张量，表示测试标
    """
    return data.preprocess_cifar10(validation_ratio=validataion_ratio, flatten=flatten)

def plot_solver(solver:Solver):
    """
    绘制solver中的loss、train_acc、val_acc变换。
    Args:
        solver: 训练后的Solver对象。
    Returns:
        None
    """
    stat_dict = {'loss_history':solver.loss_history,
                 'train_acc_history':solver.train_acc_history,
                 'val_acc_history':solver.val_acc_history}
    
    lr_reg_str = ""
    try:
        lr_reg_str = f"lr:{solver.learning_rate}, reg:{solver.model.reg}"
    except:
        pass
    
    plot_stats(stat_dict, lr_reg_str)
    

def plot_stats(stat_dict, extra_str="",x=0.8,y=-0.11):
    """
    绘制loss函数变化，训练验证的准确率变化。
    Args:
        stat_dict: 字典:
            'loss_history':
            'train_acc_history':
            'val_acc_history':

    Returns:
        None
    """
    plt.subplot(1, 2, 1)
    plt.plot(stat_dict['loss_history'], 'o')
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(stat_dict['train_acc_history'], 'o-', label='train')
    plt.plot(stat_dict['val_acc_history'], 'o-', label='val')
    plt.title('Classification accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Classification accuracy')
    plt.legend()  # 给出图例
    ax = plt.gca()
    
    
    plt.text(x, y, s=extra_str, transform=ax.transAxes)
    plt.gcf().set_size_inches(14, 4)  # gcf get current figure
    plt.show()

def plot_stats_ax(stat_dict):
    """
    用面向对象的plot绘制损失函数变化和训练、验证的准确率。
    与plot_stats的功能相似*
    Args:
        stat_dict: 模型的训练数据字典:
            'loss_history': loss值列表
            'train_acc_history': 训练准确率列表
            'val_acc_history': 验证准确率列表

    Returns:
        None
    """
    fig, axs = plt.subplots(1, 2, figsize=(14, 4))

    axs[0].plot(stat_dict['loss_history'], 'o')
    axs[0].set_title('Loss history')
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Loss')

    axs[1].plot(stat_dict['train_acc_history'], 'o-', label='train')
    axs[1].plot(stat_dict['val_acc_history'], 'o-', label='val')
    axs[1].set_title('Classification accuracy history')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Classification accuracy')
    axs[1].legend()

    plt.show()


def plot_acc_curves(stat_dict):
    """
    绘制多个模型的loss变化和训练、验证的准确率的变化。
    Args:
        stat_dict: 多个模型的训练数据字典
        例如 stat_dict = {
            'model1': {'train_acc_history': [0.1, 0.2, 0.3, 0.4], 'val_acc_history': [0.15, 0.25, 0.35, 0.45]},
            'model2': {'train_acc_history': [0.1, 0.3, 0.5, 0.7], 'val_acc_history': [0.2, 0.4, 0.6, 0.8]}
        }

    Returns:
        None
    """
    plt.subplot(1, 2, 1)
    for key, single_stats in stat_dict.items():
        plt.plot(single_stats['train_acc_history'], label=str(key))
    plt.title('Train accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Clasification accuracy')

    plt.subplot(1, 2, 2)
    for key, single_stats in stat_dict.items():
        plt.plot(single_stats['val_acc_history'], label=str(key))
    plt.title('Validation accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Clasification accuracy')
    plt.legend()

    plt.gcf().set_size_inches(14, 5)
    plt.show()


def visualize_grid(x, upper_bound=255.0, padding=1):
    """
    重塑4D的张量x使得其能够容易的可视化。
    会将每一张2D矩阵映射到[0, upper_bound]的范围，浮点格式，使用 img - low / (high - low)。

    Args:
        x: (N, H, W, C) tensor, 值从[0, 1]
        upper_bound: 输出的gird的值将缩放到 [0, upperbound]
        padding: gird与gird之间的空白像素填充个数

    Returns:
        grid: 重塑之后的张量
    """
    N, H, W, C = x.shape
    grid_size = int(math.ceil(math.sqrt(N)))  # grid数量为总图像数量开根上取整
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = torch.zeros((grid_height, grid_width, C), device=x.device)  # 大画布
    idx = 0
    x1, x2 = 0, H  # 单个"图片" H*W, 图片的坐标以(x1, y1), (x2, y2) 给出

    for _ in range(grid_size):
        y1, y2 = 0, W
        for _ in range(grid_size):
            if idx < N:
                img = x[idx]
                low, high = torch.min(img), torch.max(img)
                grid[x1:x2, y1:y2] = upper_bound * (img - low) / (high - low)  # 原函数似乎写反了，这里对调了顺序
                idx += 1
            y1 += W + padding  # 水平移动
            y2 += W + padding
        x1 += H + padding  # 竖直移动
        x2 += H + padding

    return grid


def show_net_weights(net, key='W1'):
    """
    绘制指定权重的图像。
    Args:
        net: ？？？
        key: 权重的名称

    Returns:
        None
    """
    W = net.params[key] # (N, 3, 32, 32)
    W = W.reshape(3, 32, 32, -1).transpose(0, 3)  # (3, 32, 32,

    plt.imshow(visualize_grid(W, padding=3).type(torch.uint8).cpu())
    plt.gca().axis('off')
    plt.show()


def svm_loss(x, y):
    """
    对Multicalss SVM函数求梯度。
    Args:
        x: tensor, shape(N, C)， x[i, j] 表示第i个样本第j类的输出值
        y: tensor Vector, shape (N, ), 0 <= y[i] < C

    Returns:
        loss: int类型的loss值
        dx: x的梯度
    """
    N = x.shape[0]
    correct_class_scores = x[torch.arange(N), y]

    # 计算SVM的hinge loss，对于单个样本公式为∑max(0, s_j - s_yi + 1), j!=y[i]，其中s_j代表错误类别的分数，s_yi代表正确类别的分数
    # correct_class_scores[:, ]是(N, ), correct_class_scores[:, None]是(N, 1)
    margins = (x - correct_class_scores[:, None] + 1.0).clamp(min=0.)  # clamp 截断到0
    margins[torch.arange(N), y] = 0.  # 公式中j!=y[i]表示j==y[i]地方的值不统计到loss里面，故将这些地方的值变0防止影响结果
    loss = margins.sum() / N  # 平均损失

    # 计算每个样本有多少类别的间隔大于0
    num_pos = (margins > 0).sum(dim=1)  # (N, )

    dx = torch.zeros_like(x)

    dx[margins > 0] = 1. # 对于margin大于0的类别，表示有元素，求导导数为1
    dx[torch.arange(N), y] -= num_pos.to(dx.dtype)  # 样本有多少类别的间隔大于0, 则表达式中有多少个-y[i]，导数为-1，所以直接减去
    dx /= N  # 平均梯度
    return loss, dx



def softmax_loss(x, y):
    """
    对softmax+log函数求梯度。
    Args:
        x: tensor, shape(N, C)， x[i, j] 表示第i个样本第j类的输出值
        y: tensor Vector, shape (N, ), 0 <= y[i] < C

    Returns:
        loss: int类型的loss值
        dx: x的梯度
    """
    # 让数值稳定，不改变结果
    shifted_logits = x - x.max(dim=1, keepdim=True).values  # (N, C) - (N, 1)
    # 见笔记
    Z = shifted_logits.exp().sum(dim=1, keepdim=True) # (N, C)->(N, 1)
    log_probs = shifted_logits - Z.log()
    probs = log_probs.exp()
    N = x.shape[0]
    loss = (-1.0 / N) * log_probs[torch.arange(N), y].sum()
    dx = probs.clone()
    dx[torch.arange(N), y] -= 1
    dx /= N
    return loss, dx









