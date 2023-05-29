import random

import torch

from toolset.utils import *

""" 计算和检查梯度的工具，用来check反向传播的实现是否正确。"""


def grad_check_sparse(f, x, analytic_grad, num_checks=10, h=1e-7):
    """
    执行数值梯度检查(稀疏采样)。用中心差 分公式来计算数值导数：
    
        f'(x) =~ (f(x + h) - f(x - h)) / (2h)

    并未计算全部的数值梯度，而是而是采样部分维度进行数值导数计算。
    Args:
        f: 输入torch张量并返回torch标量的函数
        x: 用于评估数值梯度的点的torch张量
        analytic_grad: 在x处的f的解析梯度的torch张量
        num_checks: 需要检查的维度数量
        h: 计算数值导数的步长
    """
    reset_seed(0)
    for i in range(num_checks):

        ix = tuple([random.randrange(m) for m in x.shape])

        oldval = x[ix].item()
        x[ix] = oldval + h  # + h
        fxph = f(x).item()  # 计算 f(x + h)
        x[ix] = oldval - h  # - h 
        fxmh = f(x).item()  # 计算 f(x - h)
        x[ix] = oldval  # 重置

        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic = analytic_grad[ix]
        rel_error_top = abs(grad_numerical - grad_analytic)
        rel_error_bot = abs(grad_numerical) + abs(grad_analytic) + 1e-12
        rel_error = rel_error_top / rel_error_bot
        msg = "numerical: %f analytic: %f, relative error: %e"
        print(msg % (grad_numerical, grad_analytic, rel_error))


def compute_numeric_gradient(f, x, dLdf=None, h=1e-7):
    """ 
    执行数值梯度检查(全采样)。用中心差分公式来计算数值导数：
    
        df    f(x + h) - f(x - h)
        -- ~= -------------------
        dx          2 * h

    此函数还可以使用链式法则轻松扩展到中间层:

        dL   df   dL
        -- = -- * --
        dx   dx   df
        
    Args:
        f: 一个输入torch张量并返回torch标量的函数
        x: 一个给出计算梯度点的torch张量
        dLdf: 可选的，用于中间层的上游梯度
        h: 用于有限差分计算的eps
    Return:
        grad: 与x形状相同的张量，给出了在x处f的梯度
    """

    flat_x = x.contiguous().flatten()
    grad = torch.zeros_like(x)
    flat_grad = grad.flatten()

    # 没有上游梯度则初始化上游梯度全1
    if dLdf is None:
        y = f(x)
        dLdf = torch.ones_like(y)
    dLdf = dLdf.flatten()

    
    for i in range(flat_x.shape[0]):
        oldval = flat_x[i].item()  # 备份原始数据
        flat_x[i] = oldval + h  
        fxph = f(x).flatten()  # f(x + h)
        flat_x[i] = oldval - h  
        fxmh = f(x).flatten()  # f(x - h)
        flat_x[i] = oldval  

        # 中心差分
        dfdxi = (fxph - fxmh) / (2 * h)

        # 链式法则相乘
        flat_grad[i] = dLdf.dot(dfdxi).item()

    # flat_grad是grad的引用，所以可以直接返回grad
    return grad


def rel_error(x, y, eps=1e-10):
    """
    计算张量x和y之间的相对误差，定义为:
    
                                max_i |x_i - y_i|
        rel_error(x, y) = -------------------------------
                        max_i |x_i| + max_i |y_i| + eps

    输入:
    - x, y: 形状相同的张量
    - eps: 常数，用于数值稳定性

    返回:
    - rel_error: 标量，给出了x和y之间的相对误差
    """
    top = (x - y).abs().max().item()
    bot = (x.abs() + y.abs()).clamp(min=eps).max().item()
    return top / bot
