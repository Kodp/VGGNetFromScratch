"""
注意:
1. "层"类不可以改变输入数据(源数据)！ "层"类不可以改变输入数据(源数据)！ "层"类不可以改变输入数据(源数据)！

原因:
假设你正在使用一个神经网络进行训练，然后在ReLU层中，你决定不去克隆输入的数据`x`，而是直接在原数据上进行操作。

def forward(x):
    x[x < 0] = 0
    cache = x
    return x, cache

现在，考虑以下情况：

x = torch.tensor([-1, 2, -3, 4])
relu_layer = ReLU()
y, _ = relu_layer.forward(x)

在ReLU层的forward方法中，你直接在输入的数据`x`上进行了修改。现在，如果你去打印原始的输入`x`，你会发现它已经被修改了：

print(x)  # 输出：tensor([0, 2, 0, 4])

这就是一个具体的例子，原本我们期望`x`保持不变，但是因为ReLU层的实现直接修改了原始数据，我们的期望被破坏了。
这可能在实际应用中产生许多问题。比如说，如果你想要重复某个计算，或者你在后面的某个地方需要原始的`x`，
你就会发现你得到的不再是原始的输入，而是被ReLU修改过的输入。这将导致结果的不一致，使得你的程序行为变得难以预测，并且可能引入难以调试的错误。

2. 层类也不能包含to方法，数据的修改应该在更高的层上做

3. 层类实现正向传播和反向传播，但不实现梯度更新。梯度更新由solver类完成，其中包含了多种参数、优化方法可设置。

"""

import torch
from toolset.helper import softmax_loss
from fully_connected_networks import Linear_ReLU, Linear, Solver, adam, ReLU, Dropout
import math


class VggNet(object):
    """
    VggNet结构，可自定义任意数量卷积层的卷积神经网络，之后连接任意数量的全连接层(在Vgg中是3层)。

    所有卷积层将使用3x3的卷积核和填充1来保持特征图的大小，通俗来说就是卷积之后宽高不变。(F1,H,W)->(F2,H,W)
    所有池化层将是2x2的最大池化层，并且步幅为2，即每次减半特征图的大小。 (F,H,W)->(F,H/2,W/2)

    网络的架构如下所示：
    {卷积层 - 批归一化层 - ReLU - [池化层？]} x (filter) - {全连接-批归一化层 - [Dropout?]} x (FC-1) - 全连接层

    相比DeepConvNet类，固定加上了批归一化(批归一化=Batch Normalization=BN，虽然原论文没有加BN，但是大多数实现中为了加快收敛速度，都采用了BN)

    每个{...}结构都是一个"宏层"，包含一个卷积层、一个可选的批归一化层、一个ReLU非线性层和一个可选的池化层。
    在L-1个宏层之后，使用一个全连接层来预测类别得分。

    该网络对形状为(N, C, H, W)的数据小批量进行操作，其中N是图像数量，H和W分别是图像的高度和宽度，C是输入通道数。

    在该工程中，所有层的正向、方向传播均已实现，包含全连接、ReLu、BN、Dropout、卷积。
    也就是说网络可以用在这个工程文件里的所有层构成，完全不需要torch.nn。但是由于手写卷积的速度较慢，所以卷积在在这里采用了torch.nn。

    * 注意，参数不要用列表这种mutable对象。因为默认参数可能被永久改变导致错误:
        def func(key, value, a={}):
            a[key] = value
            return a

        print(func('a', 10))  # that's expected
        >>> {'a': 10}
        print(func('b', 20))  # that could be unexpected
        >>> {'b': 20, 'a': 10}
    * 这也表明，python的函数不是纯代码，而是一个对象(first-class object)

    """

    def __init__(self,
                 input_dims: tuple = (3, 32, 32),
                 num_filters: tuple = (32, 64, 128, 256, 256),
                 max_pools: tuple = (0, 1, 2, 3, 4),
                 num_FC: tuple = (128, 10),
                 dropout: float = 0,  # 标量，(0,1)
                 weight_scale=1e-3,
                 kaiming_ratio=1.,
                 reg=0.0,
                 print_params=False,
                 dtype=torch.float32,
                 device='cpu'):
        """
        初始化一个新的网络。

        输入：
        - input_dims：元组 (C, H, W)，给出输入数据的大小
        - num_filters：长度为 (L - 1) 的元组，给出每个宏层中要使用的卷积滤波器的数量
        - max_pools：一个整数元组，给出应该具有最大池化的宏层的索引（从零开始）
        - num_FC：一个整数元组，表示整个卷积层之后的FC层的层数和每层的神经元个数，元组最后一个值应是分类数num_classes
        - weight_scale：标量，给出权重随机初始化的标准差，或者使用字符串 "kaiming" 来使用 Kaiming 初始化
        - kaiming_ratio：kaiming初始化的系数。当网络的深度到达一定程度，原始的kaiming可能导致初始输出过大、变成NAN，这种情况下loss值会爆炸。
          使用kaiming_ratio可以进行缩放，使得初始输出变小，loss值回归正常。
          对于十分类而言，最理想的loss值是log(10)，这里可以通过调节kaiming_ratio使得初始loss降低到3以内。
          不需要特别小，对于vgg16只需要0.1~0.3就能降loss降低到3以下。
        - reg：标量，给出 L2 正则化强度系数。L2 正则化只应用于卷积层和全连接层的权重矩阵；不应用于偏置项或批归一化的缩放和偏移。
        - dtype：一个torch数据类型对象；所有计算将使用该数据类型进行。float 类型速度更快但精度较低，因此在数值梯度检查时应该使用 double 类型。
        - device：用于计算的设备。'cpu' 或 'cuda'
        """
        self.input_dims = input_dims
        self.num_filters = num_filters
        self.num_layers = len(num_filters) + len(num_FC)  # 计算总的层数=卷积层数+全连接层数
        self.num_FC = len(num_FC)  # 全连接层的数量
        self.max_pools = max_pools
        self.max_poolset = set(max_pools)
        self.reg = reg  # 正则化系数
        self.dtype = dtype  # 指定张量的数据类型
        self.use_dropout = abs(dropout) > 1e-9
        self.params = {}

        if device == 'cuda':
            device = 'cuda:0'  # 指定第0个GPU

        C, H, W = input_dims  # 通道数、高度和宽度

        # 1.初始化卷积参数和maxpool参数。
        filter_size = 3
        self.conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        self.pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        # 2.初始化BN参数。
        # List[Dict]字典列表self.bn_params。这个列表将用于保存每个卷积层的批量归一化(Batch Normalization)参数。
        # 对于批归一化，我们需要运行时的running_mean和running_var，所以需要将一个特殊的 bn_param 对象传递给每个BN层的前向传播。
        # 将 self.bn_params[0] 传递给第一个批归一化层的前向传播，将 self.bn_params[1] 传递给第二个批归一化层的前向传播，依此类推。
        self.bn_params = [{'mode': 'train'} for _ in range(self.num_layers - 1)]  # 最后一层不做BN

        # 3.初始化Dropout参数。
        # 当使用dropout时，我们需要向每个dropout层传递一个dropout_param字典，以便该层知道dropout概率和模式（train/test）。
        # 将相同的dropout_param传递给每个dropout层。
        self.dropout_param = {}
        if self.use_dropout:
            print("使用dropout。")
            self.dropout_param = {'mode': 'train', 'p': dropout}

        # 4.计算卷积层输出到全连接层的元素数量:
        # torch maxpool的规则是不够则不卷积，即下取整，。例如7x7，2x2下采样步伐为2，则采样出来为3x3而不是4x4
        # 注意2x2下采样，步伐为2，故分别在宽高上缩减2倍。不是拿总参数除以4倍，例如图像的HW都是奇数，下取整则7x7/4!=(7/2)x(7/2)
        shrink = 2 ** len(self.max_poolset)
        self.num_conv2fc = num_filters[-1] * (H // shrink) * (W // shrink)
        layer = 1  # 初始化层数计数器，下标从1开始
        ratio = kaiming_ratio

        # 5.初始化权重
        # 初始化卷积层
        for F in num_filters:
            if isinstance(weight_scale, str): # kaiming初始化
                self.params[f'W{layer}'] = kaiming_init(F, C, 3, ratio=ratio, dtype=dtype, device=device)
            else: # 正态分布初始化
                self.params[f'W{layer}'] = weight_scale * torch.randn(F, C, 3, 3, dtype=dtype, device=device)

            self.params[f'b{layer}'] = torch.zeros(F, dtype=dtype, device=device)
            self.params[f'gamma{layer}'] = torch.ones(F, dtype=dtype, device=device)
            self.params[f'beta{layer}'] = torch.zeros(F, dtype=dtype, device=device)
            C = F  # C为上一层的Channel
            layer += 1

        C = self.num_conv2fc  # 卷积层到全连接层的元素数量

        # 初始化全连接层+预测层
        for fc in num_FC:
            if isinstance(weight_scale, str):
                self.params[f'W{layer}'] = kaiming_init(C, fc, relu=layer != self.num_layers, ratio=ratio, dtype=dtype, device=device)
            else:
                self.params[f'W{layer}'] = weight_scale * torch.randn(C, fc, dtype=dtype, device=device)

            self.params[f'b{layer}'] = torch.zeros(fc, dtype=dtype, device=device)
            if layer != self.num_layers:
                self.params[f'gamma{layer}'] = torch.ones(fc, dtype=dtype, device=device)
                self.params[f'beta{layer}'] = torch.zeros(fc, dtype=dtype, device=device)
            C = fc
            layer += 1

        self._check_num_weights(device)

        if print_params:
            self.print_params()

    def print_params(self):
        """
        逐个打印权重的名称和shape。
        """
        print("参数:")
        for key, value in self.params.items():
            print(f"\t{key}: {value.shape}")

    def _check_num_weights(self, device):
        """
        检查权重的个数是否正确，用于排查错误。
        不要在类外部手动调用。
        Args:
            device: 权重所处的设备
        """

        # 为了确保我们得到了正确数量的参数，我们首先计算每个"宏"层(含有卷积-ReLU-BN的层)应该有多少参数。
        # 进行BN，那么每个宏层应有4个参数矩阵：权重、偏置、Bn的缩放因子和平移参数，即weight, bias, scale, shift
        params_per_macro_layer = 4  # W、b、γ、β

        # 计算模型中的总参数数量:
        # 每个宏层的参数数量乘以宏层的数量-1，再加上2（这个2表示最后一层线性层的权重和偏置）。
        num_params = params_per_macro_layer * (self.num_layers - 1) + 2

        # 构造一个错误消息字符串，该消息将在参数数量不正确时显示。
        msg = f'self.params has the wrong number of elements. Got {len(self.params)}; expected {num_params}'

        # 使用断言语句来检查参数数量是否正确。如果参数数量不正确，将抛出异常并显示上面构造的错误消息。
        assert len(self.params) == num_params, msg

        # 对模型中的每一个参数进行检查，确保它们都在正确的设备上，并且有正确的数据类型。
        for k, param in self.params.items():
            # 构造错误消息字符串，该消息将在参数的设备不正确时显示。
            msg = f'param "{k}" has device {param.device}; should be {device}'
            # 使用断言语句来检查参数的设备是否正确。如果设备不正确，将抛出异常并显示上面构造的错误消息。
            assert param.device == torch.device(device), msg

            # 构造错误消息字符串，该消息将在参数的数据类型不正确时显示。
            msg = f'param "{k}" has dtype {param.dtype}; should be {self.dtype}'
            # 使用断言语句来检查参数的数据类型是否正确。如果数据类型不正确，将抛出异常并显示上面构造的错误消息。
            assert param.dtype == self.dtype, msg

    # def _scale_init(self, weight_scale, num_filters, num_FC, device):
    #     """
    #     对模型权重的参数进行正态分布初始化，可以进行缩放。
    #     不要在类外部手动调用。
    #
    #     Args:
    #         weight_scale: 缩放的系数
    #         num_filters: 卷积层的数量
    #         num_FC: 全连接层的数量
    #         device: 权重应放的位置
    #     """
    #     dtype = self.dtype
    #     C, H, W = self.input_dims
    #     layer = 1  # 初始化层数计数器
    #     for F in num_filters:
    #         self.params[f'W{layer}'] = weight_scale * torch.randn(F, C, 3, 3, dtype=dtype, device=device)
    #         self.params[f'b{layer}'] = torch.zeros(F, dtype=dtype, device=device)
    #         self.params[f'gamma{layer}'] = torch.ones(F, dtype=dtype, device=device)
    #         self.params[f'beta{layer}'] = torch.zeros(F, dtype=dtype, device=device)
    #         C = F  # C为上一层的Channel
    #         layer += 1
    #
    #     C = self.num_conv2fc  # 卷积层到全连接层的元素数量
    #
    #     for fc in num_FC:
    #         self.params[f'W{layer}'] = weight_scale * torch.randn(C, fc, dtype=dtype, device=device)
    #         self.params[f'b{layer}'] = torch.zeros(fc, dtype=dtype, device=device)
    #         if layer != self.num_layers:
    #             self.params[f'gamma{layer}'] = torch.ones(fc, dtype=dtype, device=device)
    #             self.params[f'beta{layer}'] = torch.zeros(fc, dtype=dtype, device=device)
    #         C = fc
    #         layer += 1

    def check_loss(self, data_dict, num_samples=50, num_scores=10):
        """
        检查一批样本上的loss值(loss值的计算都是平均值)

        Args:
            data_dict: 数据
            num_samples: 采样个数
            num_scores: 打印的样本预测数量

        """
        scores = self.loss(data_dict['X_train'][:num_samples])
        print(scores[:num_scores])
        loss, _ = self.loss(data_dict['X_train'][:num_samples], data_dict['y_train'][:num_samples])
        print(f"loss:{loss:.6f}")

    def save(self, path):
        checkpoint = {
            'input_dims': self.input_dims,
            'num_filters': self.num_filters,
            'num_layers': self.num_layers,
            'num_FC': self.num_FC,
            'max_pools': self.max_pools,
            'max_poolset': self.max_poolset,
            'reg': self.reg,

            'use_dropout': self.use_dropout,
            'params': self.params,

            'conv_param': self.conv_param,
            'pool_param': self.pool_param,
            'bn_params': self.bn_params,
            'dropout_param': self.dropout_param,
            'num_conv2fc': self.num_conv2fc,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')

        self.input_dims = checkpoint['input_dims']
        self.num_filters = checkpoint['num_filters']
        self.num_layers = checkpoint['num_layers']
        self.num_FC = checkpoint['num_FC']
        self.max_pools = checkpoint['max_pools']
        self.max_poolset = checkpoint['max_poolset']
        self.reg = checkpoint['reg']

        self.use_dropout = checkpoint['use_dropout']  # 一定注意更新代码要更新save和load函数
        self.params = checkpoint['params']

        self.conv_param = checkpoint['conv_param']
        self.pool_param = checkpoint['pool_param']
        self.bn_params = checkpoint['bn_params']
        self.dropout_param = checkpoint['dropout_param']
        self.num_conv2fc = checkpoint['num_conv2fc']

        self.dtype = dtype
        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)

        for i in range(len(self.bn_params)):
            for p in ["running_mean", "running_var"]:
                try:
                    self.bn_params[i][p] = self.bn_params[i][p].type(dtype).to(device)
                except KeyError:
                    print(f"{i + 1}未使用BN？本网络设计成全部使用BN，哪里出了问题。")

        print(f"成功加载checkpoint文件: {path}")

    def loss(self, X, y=None):
        """
        计算深度卷积网络的损失和梯度，当y=None时仅预测。
        输入/输出：与 ThreeLayerConvNet 相同的 API。
        """
        X = X.to(self.dtype)
        mode = 'test' if y is None else 'train'
        if self.use_dropout:
            self.dropout_param['mode'] = mode

        for bn_param in self.bn_params:  # 遍历list，list里每个元素是一层的bn参数字典            
            bn_param['mode'] = mode

        # HINT  {conv - [batchnorm] - relu - [pool?]} x (L - 1) - linear
        cache_dict, dropout_cache = {}, {}
        filters = self.num_layers - self.num_FC  # 卷积层数量
        out = X

        # 1.卷积层正向传播
        for layer in range(1, filters + 1):
            W, b = self.params[f'W{layer}'], self.params[f'b{layer}']
            gamma, beta = self.params[f'gamma{layer}'], self.params[f'beta{layer}']
            bn_param = self.bn_params[layer - 1]  # 引用传递
            if layer - 1 in self.max_poolset:  # set O(1) 查询
                out, cache_dict[layer] = Conv_BatchNorm_ReLU_Pool.forward(out, W, b, gamma, beta,self.conv_param,bn_param,self.pool_param)
                # ! BatchNorm.forward会修改bn_param, 向里面添加running_mean和runing_var。
                # 导致下一轮计算的时候，bn_param的running参数不为空则取出，但是这是上一层添加的running参数、shape是上一层的，所以和当层的x参数shape不匹配、出错。
                # @ 但是如果每次传入的bn_param是每一层的bn_param，则正好同步修改，不会出现上述问题
            else:
                out, cache_dict[layer] = Conv_BatchNorm_ReLU.forward(out, W, b, gamma, beta, self.conv_param, bn_param)
            # print(out.shape)

        # 2.全连接层(除了预测层)正向传播
        for layer in range(filters + 1, self.num_layers):
            W, b = self.params[f'W{layer}'], self.params[f'b{layer}']
            gamma, beta = self.params[f'gamma{layer}'], self.params[f'beta{layer}']
            bn_param = self.bn_params[layer - 1]  # 引用传递
            out, cache_dict[layer] = Linear_BatchNorm_ReLU.forward(out, W, b, gamma, beta, bn_param)
            # 采用Dropout的处理
            if self.use_dropout:  # Dropout类里面已经区分开了train和test，例如test是直接输出
                out, dropout_cache[layer] = Dropout.forward(out, self.dropout_param)

        # 3.预测层正向传播
        L = self.num_layers
        out, cache_dict[L] = Linear.forward(out, self.params[f'W{L}'], self.params[f'b{L}'])

        # 仅推理则返回结果
        if y is None:
            return out

        # 计算loss
        loss, grads = 0, {}
        loss, dout = softmax_loss(out, y)
        for layer in range(1, L + 1):
            loss += self.reg * (self.params[f'W{layer}'] ** 2).sum()

        # 3.预测层反向传播
        dout, dw, db = Linear.backward(dout, cache_dict[L])
        grads[f'W{L}'], grads[f'b{L}'] = dw + 2 * self.reg * self.params[f'W{L}'], db

        # 2.全连接层(除了预测层)反向传播
        for layer in range(filters + 1, L)[::-1]:  # [L-1...filters+1]
            if self.use_dropout:
                dout = Dropout.backward(dout, dropout_cache[layer])
            dout, dw, db, dgamma, dbeta = Linear_BatchNorm_ReLU.backward(dout, cache_dict[layer])
            grads[f'W{layer}'], grads[f'b{layer}'] = dw + 2 * self.reg * self.params[f'W{layer}'], db
            grads[f'gamma{layer}'], grads[f'beta{layer}'] = dgamma, dbeta

        # 1.卷积层反向传播
        for layer in range(1, filters + 1)[::-1]:  # [filters...1]
            if layer - 1 in self.max_poolset:
                dout, dw, db, dgamma, dbeta = Conv_BatchNorm_ReLU_Pool.backward(dout, cache_dict[layer])
            else:
                dout, dw, db, dgamma, dbeta = Conv_BatchNorm_ReLU.backward(dout, cache_dict[layer])
            grads[f'W{layer}'], grads[f'b{layer}'] = dw + 2 * self.reg * self.params[f'W{layer}'], db
            grads[f'gamma{layer}'], grads[f'beta{layer}'] = dgamma, dbeta

        return loss, grads


class Conv(object):
    @staticmethod
    def forward(x, w, b, conv_param):
        """
        卷积层前向传播的朴素实现。
        输入由N个数据点组成，每个数据点具有C个通道，高度H和宽度W。我们使用F个不同的滤波器对每个输入进行卷积，
        其中每个滤波器跨越所有C个通道，并具有高度HH和宽度WW。

        输入：
        - x：形状为(N, C, H, W)的输入数据
        - w：形状为(F, C, HH, WW)的滤波器权重
        - b：形状为(F,)的偏置
        - conv_param：包含以下键的字典：
        - 'stride'：水平和垂直方向上相邻感受野之间的像素数。
        - 'pad'：用于对输入进行零填充的像素数。

        在填充过程中，'pad'个零应该对称地（即在高度和宽度轴上均匀地）放置在输入两侧。请注意不要直接修改原始输入x。

        返回一个元组：
        - out：形状为(N, F, H', W')的输出数据，其中H'和W'由以下公式给出：
        H' = 1 + (H + 2 * pad - HH) / stride
        W' = 1 + (W + 2 * pad - WW) / stride
        - cache：(x, w, b, conv_param)
        """
        out = None
        F, C, HH, WW = w.shape  # 卷积核的形状：F 为卷积核的数量，C 为通道数，HH 为卷积核的高度，WW 为卷积核的宽度
        N, C, H, W = x.shape  # 输入张量的形状：N 为样本数量（批次大小），H 为输入张量的高度，W 为输入张量的宽度
        s, pad = conv_param['stride'], conv_param['pad']  # 获取卷积参数：s 为步长，pad 为填充
        H_out, W_out = 1 + (H + 2 * pad - HH) // s, 1 + (W + 2 * pad - WW) // s  # 计算输出张量的高度和宽度
        padded_x = torch.nn.functional.pad(x, [pad] * 4, mode='constant', value=0)  # 对输入张量进行填充
        out = torch.zeros(N, F, H_out, W_out, dtype=x.dtype, device=x.device)  # 初始化输出张量

        for p_i in range(N):  # 遍历输入张量中的每个样本
            for f_i in range(F):  # 遍历每个卷积核
                filter = w[f_i]  # 提取当前卷积核（形状：(C, HH, WW)）
                # 两重循环的range的值(id,jd)为原图上卷积的左上角起点、其索引(i,j)为卷积结果out的下标
                for i, id in enumerate(range(0, H + 2 * pad - HH + 1, s)):
                    for j, jd in enumerate(range(0, W + 2 * pad - WW + 1, s)):
                        # 计算卷积操作：当前卷积核与输入张量的对应区域逐元素相乘，然后求和
                        out[p_i, f_i, i, j] = (filter * padded_x[p_i, :, id:id + HH, jd:jd + WW]).sum()
                # 将卷积核的偏置添加到对应的输出通道上
                out[p_i, f_i] += b[f_i]
        cache = (x, w, b, conv_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        卷积层反向传播的朴素实现。
        输入：
        - dout：上游导数。
        - cache：与conv_forward_naive中的(x, w, b, conv_param)相同的元组。

        返回一个元组：
        - dx：相对于x的梯度
        - dw：相对于w的梯度
        - db：相对于b的梯度
        """

        x, w, b, conv_param = cache
        dw, db = torch.zeros_like(w), torch.zeros_like(b)
        s, pad = conv_param['stride'], conv_param['pad']
        F, C, HH, WW = w.shape
        N, C, H, W = x.shape
        H_out, W_out = 1 + (H + 2 * pad - HH) // s, 1 + (W + 2 * pad - WW) // s

        padded_x = torch.nn.functional.pad(x, [pad] * 4, mode='constant', value=0)
        dp_x = torch.zeros_like(padded_x)  # 求padded_x的导数，然后取出dx的部分

        for p_i in range(N):
            for f_i in range(F):
                filter = w[f_i]  # (C, HH, WW)
                # range产生padded_x上的左上角，其下标是out上的对应位置
                # 不要忘记乘上游梯度
                for i, id in enumerate(range(0, H + 2 * pad - HH + 1, s)):
                    for j, jd in enumerate(range(0, W + 2 * pad - WW + 1, s)):
                        dw[f_i] += dout[p_i, f_i, i, j] * padded_x[p_i, :, id:id + HH:, jd:jd + WW]
                        dp_x[p_i, :, id:id + HH:, jd:jd + WW] += dout[p_i, f_i, i, j] * filter
                db[f_i] += dout[p_i, f_i].sum()  # 有多张图片卷积，用+=而不是=

        dx = dp_x[:, :, pad:pad + H, pad:pad + W]  # 取出dx部分

        return dx, dw, db


class MaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        """
        最大池化层前向传播的朴素实现。

        输入：
        - x：输入数据，形状为(N, C, H, W)
        - pool_param：字典，包含以下键：
        - 'pool_height'：每个池化区域的高度
        - 'pool_width'：每个池化区域的宽度
        - 'stride'：相邻池化区域之间的距离
        这里不需要填充。

        返回一个元组：
        - out：形状为(N, C, H', W')的输出，其中H'和W'由以下公式给出：
        H' = 1 + (H - pool_height) / stride
        W' = 1 + (W - pool_width) / stride
        - cache：(x, pool_param)
        """
        out = None

        N, C, H, W = x.shape
        ph, pw, s = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
        H_out, W_out = 1 + (H - ph) // s, 1 + (W - pw) // s
        out = torch.zeros(N, C, H_out, W_out, dtype=x.dtype, device=x.device)

        for p_i in range(N):
            for c_i in range(C):
                for i, id in enumerate(range(0, H - ph + 1, s)):
                    for j, jd in enumerate(range(0, W - pw + 1, s)):
                        # 注意x取p_i,c_i上卷积窗口大小的max
                        out[p_i, c_i, i, j] = x[p_i, c_i, id:id + ph:, jd:jd + pw].max()
        cache = (x, pool_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        最大池化层反向传播的朴素实现。
        输入：
        - dout：上游导数
        - cache：与前向传播中的(x, pool_param)相同的元组。
        返回：
        - dx：相对于x的梯度
        """
        dx = None
        x, pool_param = cache
        N, C, H, W = x.shape
        ph, pw, s = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
        H_out, W_out = 1 + (H - ph) // s, 1 + (W - pw) // s
        dx = torch.zeros_like(x)

        for p_i in range(N):
            for c_i in range(C):
                for i, id in enumerate(range(0, H - ph + 1, s)):
                    for j, jd in enumerate(range(0, W - pw + 1, s)):
                        # 注意p_i,c_i上的max
                        roi = x[p_i, c_i, id:id + ph:, jd:jd + pw]
                        pos = roi.argmax()
                        row, col = divmod(pos.item(), roi.shape[1])  # 吼吼
                        dx[p_i, c_i, id + row, jd + col] += dout[p_i, c_i, i, j]

        return dx


class ThreeLayerConvNet(object):
    """
    具有以下结构的三层卷积网络：
    conv - relu - 2x2最大池化 - linear - relu - linear - softmax
    该网络对具有形状(N, C, H, W)的数据进行操作，其中N表示图像数量，H和W表示每个图像的高度和宽度，C表示输入通道数。
    """

    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=32,
                 filter_size=7,
                 hidden_dim=100,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 dtype=torch.float,
                 device='cpu'):
        """
        初始化一个新的网络。
        输入:
        - input_dims: 元组 (C, H, W)，给出输入数据的大小
        - num_filters: 卷积层中使用的滤波器数量
        - filter_size: 卷积层中使用的滤波器的宽度/高度
        - hidden_dim: 全连接隐藏层中使用的单元数
        - num_classes: 最终线性层产生的分数数量
        - weight_scale: 用于随机初始化权重的标准差
        - reg: L2正则化的强度
        - dtype: torch数据类型对象；所有计算将使用此数据类型进行
        - device: 用于计算的设备。'cpu'或'cuda'
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        # 初始化三层卷积网络的权重和偏置。权重应从均值为0.0、标准差为weight_scale的高斯分布中进行初始化；
        # 偏置应初始化为零。所有的权重和偏置应存储在self.params字典中。
        # 使用键'W1'和'b1'存储卷积层的权重和偏置；使用键'W2'和'b2'存储隐藏线性层的权重和偏置；使用键'W3'和'b3'存储输出线性层的权重和偏置。

        # 假设第一层卷积层的填充和步幅被选择为**保持输入的宽度和高度不变**。
        # 在实现中使用快速/堆叠层。

        C, H, W = input_dims
        # 卷积核(F,C,HH,WW)
        F, HH, WW = num_filters, filter_size, filter_size
        self.params['W1'] = weight_scale * torch.randn(F, C, HH, WW, dtype=dtype, device=device)
        self.params['b1'] = torch.zeros(F, dtype=dtype, device=device)
        # 假设卷积之后的单张图片宽高不变(F,H,W)
        # 则maxpool相当于收缩一倍 (F,H,W)->(F,H/2,W/2)
        self.params['W2'] = weight_scale * torch.randn(F * H * W // 4, hidden_dim, dtype=dtype, device=device)
        self.params['b2'] = torch.zeros(hidden_dim, dtype=dtype, device=device)
        self.params['W3'] = weight_scale * torch.randn(hidden_dim, num_classes, dtype=dtype, device=device)
        self.params['b3'] = torch.zeros(num_classes, dtype=dtype, device=device)
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

    def save(self, path):
        checkpoint = {
            'reg': self.reg,
            'dtype': self.dtype,
            'params': self.params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = checkpoint['dtype']
        self.reg = checkpoint['reg']
        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        计算三层卷积网络的损失和梯度。输入/输出与TwoLayerNet相同。
        """
        X = X.to(self.dtype)
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # 将conv_param传递给卷积层的前向传播。选择padding和stride以保持输入的空间尺寸不变。
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None

        # HINT: conv - relu - 2x2 max pool - linear - relu - linear - softmax
        cache_dict = {}
        scores, cache_dict['CRP'] = Conv_ReLU_Pool.forward(X, W1, b1, conv_param, pool_param)
        scores, cache_dict['LR'] = Linear_ReLU.forward(scores, W2, b2)
        scores, cache_dict['L'] = Linear.forward(scores, W3, b3)
        if y is None:
            return scores

        loss, grads = 0.0, {}
        loss, dout = softmax_loss(scores, y)
        loss += self.reg * ((W1 ** 2).sum() + (W2 ** 2).sum() + (W3 ** 2).sum())

        dout, dw, db = Linear.backward(dout, cache_dict['L'])
        grads['W3'], grads['b3'] = dw + 2 * self.reg * W3, db
        dout, dw, db = Linear_ReLU.backward(dout, cache_dict['LR'])
        grads['W2'], grads['b2'] = dw + 2 * self.reg * W2, db
        dout, dw, db = Conv_ReLU_Pool.backward(dout, cache_dict['CRP'])
        grads['W1'], grads['b1'] = dw + 2 * self.reg * W1, db

        return loss, grads


class DeepConvNet(object):
    """
    具有任意数量卷积层的卷积神经网络，采用VGG-Net风格。
    所有卷积层将使用3x3的卷积核和填充1来保持特征图的大小，所有池化层将是2x2的最大池化层，并具有2的步幅来减半特征图的大小。

    网络的架构如下所示：
    {卷积层 - [批归一化层?] - ReLU - [池化层?]} x (L - 1) - 全连接层

    每个{...}结构都是一个"宏层"，包含一个卷积层、一个可选的批归一化层、一个ReLU非线性层和一个可选的池化层。
    在L-1个宏层之后，使用一个全连接层来预测类别得分。

    该网络对形状为(N, C, H, W)的数据小批量进行操作，其中N是图像数量，H和W分别是图像的高度和宽度，C是输入通道数。
    """

    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=[8, 8, 8, 8, 8],
                 max_pools=[0, 1, 2, 3, 4],
                 batchnorm=False,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 dtype=torch.float,
                 device='cpu'):
        """
        初始化一个新的网络。

        输入：
        - input_dims：元组 (C, H, W)，给出输入数据的大小
        - num_filters：长度为 (L - 1) 的列表，给出每个宏层中要使用的卷积滤波器的数量
        - max_pools：一个整数列表，给出应该具有最大池化的宏层的索引（从零开始）
        - batchnorm：是否在每个宏层中包括批归一化
        - num_classes：要从最后一个线性层产生的得分数量
        - weight_scale：标量，给出权重随机初始化的标准差，或者使用字符串 "kaiming" 来使用 Kaiming 初始化
        - reg：标量，给出 L2 正则化强度。L2 正则化只应用于卷积层和全连接层的权重矩阵；不应用于偏置项或批归一化的缩放和偏移。
        - dtype：一个torch数据类型对象；所有计算将使用该数据类型进行。float 类型速度更快但精度较低，因此在数值梯度检查时应该使用 double 类型。
        - device：用于计算的设备。'cpu' 或 'cuda'
        """
        self.params = {}
        self.num_layers = len(num_filters) + 1
        self.max_pools = max_pools
        self.batchnorm = batchnorm
        self.reg = reg
        self.dtype = dtype

        if device == 'cuda':
            device = 'cuda:0'

        # HINT:  {conv - [batchnorm?] - relu - [pool?]} x (L - 1) - linear
        # 卷积size=3x3,步伐为1->卷积之后宽高不变
        # 则maxpool相当于收缩一倍 (F,H,W)->(F,H/2,W/2)
        C, H, W = input_dims
        L = self.num_layers
        shrink = 4 ** len(set(max_pools))  # 计算执行所有maxpool之后的缩减比例，每一个pool都要在宽高上缩减2倍，所以是4倍

        if isinstance(weight_scale, str):  # kaiming
            print("使用 kaiming 初始化")
            for layer, F in enumerate(num_filters):
                self.params[f'W{layer + 1}'] = kaiming_init(F, C, 3, dtype=dtype, device=device)
                self.params[f'b{layer + 1}'] = torch.zeros(F, dtype=dtype, device=device)
                if self.batchnorm:
                    self.params[f'gamma{layer + 1}'] = torch.ones(F, dtype=dtype, device=device)
                    self.params[f'beta{layer + 1}'] = torch.zeros(F, dtype=dtype, device=device)
                C = F  # C为上一层的Channel

            self.params[f'W{L}'] = kaiming_init(C * H * W // shrink, num_classes, dtype=dtype, device=device)
        else:
            for layer, F in enumerate(num_filters):
                self.params[f'W{layer + 1}'] = weight_scale * torch.randn(F, C, 3, 3, dtype=dtype, device=device)
                self.params[f'b{layer + 1}'] = torch.zeros(F, dtype=dtype, device=device)
                if self.batchnorm:
                    self.params[f'gamma{layer + 1}'] = torch.ones(F, dtype=dtype, device=device)
                    self.params[f'beta{layer + 1}'] = torch.zeros(F, dtype=dtype, device=device)
                C = F  # C为上一层的Channel
            self.params[f'W{L}'] = weight_scale * torch.randn(C * H * W // shrink, num_classes, dtype=dtype,
                                                              device=device)

        self.params[f'b{L}'] = torch.zeros(num_classes, dtype=dtype, device=device)

        # 对于批归一化，我们需要跟踪运行的均值和方差，因此我们需要将一个特殊的 bn_param 对象传递给每个批归一化层的前向传播。
        # 将 self.bn_params[0] 传递给第一个批归一化层的前向传播，将 self.bn_params[1] 传递给第二个批归一化层的前向传播，依此类推。
        # 初始化空列表 self.bn_params。这个列表将用于保存每个卷积层的批量归一化(Batch Normalization)参数。
        self.bn_params = []

        # 判断是否需要进行批量归一化操作。
        if self.batchnorm:
            # 如果进行批量归一化，则对每个卷积层添加一个字典，其中包含模式(mode)键，
            # 它的值设置为'train'，表示在训练过程中使用这个批量归一化参数。
            self.bn_params = [{'mode': 'train'}
                              for _ in range(len(num_filters))]

        # 为了确保我们得到了正确数量的参数，我们首先计算每个"宏"层(含有卷积、ReLU和可能有的批量归一化操作的层)应该有多少参数。
        if not self.batchnorm:
            # 如果没有批量归一化，那么每个宏层应有2个参数：权重和偏置。
            params_per_macro_layer = 2  # weight and bias
        else:
            # 如果进行批量归一化，那么每个宏层应有4个参数：权重、偏置、批量归一化的缩放因子和平移参数。
            params_per_macro_layer = 4  # weight, bias, scale, shift

        # 计算模型中的总参数数量：每个宏层的参数数量乘以宏层的数量，再加上2（这个2代表全连接层的权重和偏置）。
        num_params = params_per_macro_layer * len(num_filters) + 2

        # 构造一个错误消息字符串，该消息将在参数数量不正确时显示。
        msg = 'self.params has the wrong number of ' \
              'elements. Got %d; expected %d'
        msg = msg % (len(self.params), num_params)

        # 使用断言语句来检查参数数量是否正确。如果参数数量不正确，将抛出异常并显示上面构造的错误消息。
        assert len(self.params) == num_params, msg

        # 对模型中的每一个参数进行检查，确保它们都在正确的设备上，并且有正确的数据类型。
        for k, param in self.params.items():
            # 构造错误消息字符串，该消息将在参数的设备不正确时显示。
            msg = 'param "%s" has device %r; should be %r' \
                  % (k, param.device, device)
            # 使用断言语句来检查参数的设备是否正确。如果设备不正确，将抛出异常并显示上面构造的错误消息。
            assert param.device == torch.device(device), msg

            # 构造错误消息字符串，该消息将在参数的数据类型不正确时显示。
            msg = 'param "%s" has dtype %r; should be %r' \
                  % (k, param.dtype, dtype)
            # 使用断言语句来检查参数的数据类型是否正确。如果数据类型不正确，将抛出异常并显示上面构造的错误消息。
            assert param.dtype == dtype, msg

    def save(self, path):
        checkpoint = {
            'reg': self.reg,
            'dtype': self.dtype,
            'params': self.params,
            'num_layers': self.num_layers,
            'max_pools': self.max_pools,
            'batchnorm': self.batchnorm,
            'bn_params': self.bn_params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = dtype
        self.reg = checkpoint['reg']
        self.num_layers = checkpoint['num_layers']
        self.max_pools = checkpoint['max_pools']
        self.batchnorm = checkpoint['batchnorm']
        self.bn_params = checkpoint['bn_params']

        for p in self.params:
            self.params[p] = \
                self.params[p].type(dtype).to(device)

        for i in range(len(self.bn_params)):
            for p in ["running_mean", "running_var"]:
                self.bn_params[i][p] = \
                    self.bn_params[i][p].type(dtype).to(device)

        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        # @当y=None可以用来预测
        """
        评估深度卷积网络的损失和梯度。
        输入/输出：与 ThreeLayerConvNet 相同的 API。
        """
        X = X.to(self.dtype)
        mode = 'test' if y is None else 'train'

        if self.batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        filter_size = 3
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        # HINT  {conv - [batchnorm?] - relu - [pool?]} x (L - 1) - linear
        cache_dict, L = {}, self.num_layers
        max_pools = set(self.max_pools)
        out = X

        if self.batchnorm:
            for layer in range(1, L):  # 遍历卷积层
                W, b = self.params[f'W{layer}'], self.params[f'b{layer}']
                gamma, beta = self.params[f'gamma{layer}'], self.params[f'beta{layer}']
                bn_param = self.bn_params[layer - 1]  # 引用传递
                if layer - 1 in max_pools:  # set O(1)
                    out, cache_dict[layer] = Conv_BatchNorm_ReLU_Pool.forward(out, W, b, gamma, beta, conv_param,
                                                                              bn_param,
                                                                              pool_param)
                    # ! BatchNorm.forward会修改bn_param, 向里面添加running_mean和runing_var。
                    # 导致下一轮计算的时候，bn_param的running参数不为空则取出，但是这是上一层添加的running参数、shape是上一层的，所以和当层的x参数shape不匹配、出错。
                    # @ 但是如果每次传入的bn_param是每一层的bn_param，则正好同步修改，不会出现上述问题
                else:
                    out, cache_dict[layer] = Conv_BatchNorm_ReLU.forward(out, W, b, gamma, beta, conv_param, bn_param)
        else:
            for layer in range(1, L):
                W, b = self.params[f'W{layer}'], self.params[f'b{layer}']
                if layer - 1 in max_pools:  # ! max_pools用的0index
                    out, cache_dict[layer] = Conv_ReLU_Pool.forward(out, W, b, conv_param, pool_param)
                else:
                    out, cache_dict[layer] = Conv_ReLU.forward(out, W, b, conv_param)

        out, cache_dict[L] = Linear.forward(out, self.params[f'W{L}'], self.params[f'b{L}'])
        scores = out
        if y is None:
            return scores

        loss, grads = 0, {}
        loss, dout = softmax_loss(scores, y)
        for i in range(1, L + 1):
            loss += self.reg * (self.params[f'W{i}'] ** 2).sum()

        dout, dw, db = Linear.backward(dout, cache_dict[L])
        grads[f'W{L}'], grads[f'b{L}'] = dw + 2 * self.reg * self.params[f'W{L}'], db

        if self.batchnorm:
            for layer in range(1, L)[::-1]:
                if layer - 1 in max_pools:
                    dout, dw, db, dgamma, dbeta = Conv_BatchNorm_ReLU_Pool.backward(dout, cache_dict[layer])
                else:
                    dout, dw, db, dgamma, dbeta = Conv_BatchNorm_ReLU.backward(dout, cache_dict[layer])
                grads[f'W{layer}'], grads[f'b{layer}'] = dw + 2 * self.reg * self.params[f'W{layer}'], db
                grads[f'gamma{layer}'], grads[f'beta{layer}'] = dgamma, dbeta
        else:
            for layer in range(1, L)[::-1]:
                if layer - 1 in max_pools:
                    dout, dw, db = Conv_ReLU_Pool.backward(dout, cache_dict[layer])
                else:
                    dout, dw, db = Conv_ReLU.backward(dout, cache_dict[layer])
                grads[f'W{layer}'], grads[f'b{layer}'] = dw + 2 * self.reg * self.params[f'W{layer}'], db

        return loss, grads


def find_overfit_parameters():
    weight_scale = 0.1
    # ! Naive初始化初值敏感性太高了，weightscale设成1e-4准确率动都不带动的
    learning_rate = 0.001
    return weight_scale, learning_rate


def create_convolutional_solver_instance(data_dict, dtype, device):
    model = None
    solver = None
    input_dims = data_dict['X_train'].shape[1:]
    weight_scale = 'kaiming'

    model = DeepConvNet(input_dims=input_dims, num_classes=10,
                        num_filters=([32] * 2) + ([64] * 2) + ([128] * 1),
                        max_pools=[1, 3, 4],
                        weight_scale=weight_scale,
                        reg=1e-5,
                        dtype=torch.float32,
                        device='cuda'
                        )
    solver = Solver(model, data_dict,
                    num_epochs=100, batch_size=128,
                    update_rule=adam,
                    optim_config={
                        'learning_rate': 0.002,
                    },
                    # lr_decay = 0.95,
                    print_every=50, device='cuda')
    # solver帮你实现了batch_size,不过不是cuda友好类型
    return solver


def kaiming_init(Din, Dout, K=None, relu=True, ratio=1., device='cpu',
                 dtype=torch.float32):
    """
    为线性层和卷积层实现 Kaiming 初始化。

    输入:
    - Din, Dout: 给出该层的输入和输出维度数量的整数。
    - K: 如果 K 为 None，则初始化具有 Din 个输入维度和 Dout 个输出维度的线性层的权重。
    如果 K 是非负整数，则初始化具有 Din 个输入通道，Dout 个输出通道和大小为 KxK 的卷积层的权重。
    - relu: 如果 relu=True，则使用增益因子 2 初始化权重以适应 ReLU 非线性性（Kaiming 初始化）；否则使用增益因子 1 初始化权重（Xavier 初始化）。
    - device, dtype: 输出张量的设备和数据类型。
    返回:
    - weight: 一个 torch 张量，给出此层的初始化权重。对于线性层，它应该具有形状 (Din, Dout)；对于卷积层，它应该具有形状 (Dout, Din, K, K)。
    """
    gain = 2. if relu else 1.
    weight = None
    if K is None:
        # 线性层的Kaiming初始化。权重的缩放因子为sqrt(gain / fan_in)，
        # 其中gain为2（如果ReLU在该层之后）或者1（如果没有ReLU），fan_in为输入通道数（等于Din）。
        # 返回的权重张量应具有指定的大小、数据类型和设备。
        std = torch.sqrt(gain / torch.tensor(Din))
        weight = torch.randn(Din, Dout, dtype=dtype, device=device) * std * ratio
    else:

        # 卷积层的Kaiming初始化。
        # 权重的尺度为sqrt(gain / fan_in)，
        # 其中gain为2（如果ReLU在该层后面），否则为1，
        # 而fan_in = num_in_channels (= Din) * K * K 
        # 输出应为在指定的大小、dtype和device上的张量。

        std = torch.sqrt(gain / torch.tensor(K * K * Din))
        weight = torch.randn(Din, Dout, K, K, dtype=dtype, device=device) * std * ratio

    return weight


class BatchNorm(object):

    @staticmethod
    def forward(x, gamma, beta, bn_param):
        """
        批量归一化的前向传播。

        在训练过程中，从小批量统计中计算样本均值和（未校正的）样本方差，并用它们来归一化输入数据。
        在训练过程中，我们还使用指数衰减的方式来维护每个特征的均值和方差的运行平均值，这些平均值用于在测试时归一化数据。

        在每个时间步骤中，我们使用基于动量参数的指数衰减来更新均值和方差的运行平均值：

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        注意，批量归一化论文建议在测试时使用不同的方法：它们使用大量训练图像来计算每个特征的样本均值和方差，而不是使用运行平均值。
        对于这个实现，我们选择使用运行平均值，因为它们不需要额外的估计步骤；PyTorch中的批量归一化实现也使用运行平均值。

        输入:
        - x: 形状为（N，D）的数据
        - gamma: 形状为（D，）的缩放参数
        - beta: 形状为（D，）的平移参数
        - bn_param: 字典，包含以下键:
          - mode: 'train'或'test'；必需的
          - eps: 数值稳定性的常数
          - momentum: 运行均值/方差的常数。
          - running_mean: 形状为（D，）的特征运行均值数组
          - running_var: 形状为（D，）的特征运行方差数组

        返回一个元组:
        - out: 形状为（N，D）的输出
        - cache: 反向传播过程中需要的值的元组
        """

        mode = bn_param['mode']
        eps = bn_param.get('eps', 1e-5)
        momentum = bn_param.get('momentum', 0.9)

        N, D = x.shape
        running_mean = bn_param.get('running_mean', torch.zeros(D, dtype=x.dtype, device=x.device))
        running_var = bn_param.get('running_var', torch.zeros(D, dtype=x.dtype, device=x.device))

        out, cache = None, None
        if mode == 'train':
            # Hint Lecture7 公式
            # Hint running_mean,running_var计算在Train，用在Test，在Test不用再算
            # sample_var, sample_mean = torch.var_mean(x, dim = 0)
            # ! 巨坑，torch.var_mean的var是无偏估计，用的系数1/(N-1)，
            # ! 不是我们这里用的1/N有偏, 不能用

            mean = 1. / N * x.sum(dim=0)
            var = 1. / N * ((x - mean) ** 2).sum(dim=0)

            running_mean = momentum * running_mean + (1 - momentum) * mean
            running_var = momentum * running_var + (1 - momentum) * var
            rsqrt = 1. / (var + eps).sqrt()
            x_hat = (x - mean) * rsqrt
            out = gamma * x_hat + beta
            cache = (x, x_hat, mean, var, gamma, rsqrt, eps)
        elif mode == 'test':
            out = gamma * ((x - running_mean) / (running_var + eps).sqrt()) + beta
        else:
            raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

        # 在PyTorch中，detach()方法用于创建一个新的张量，该张量与原始张量共享相同的底层数据，但不需要梯度。
        # 换句话说，它将张量与计算图“分离”，因此在反向传播过程中不会通过此张量进行梯度信息的反向传播。
        # 将更新后的运行均值存储回bn_param

        # $ 修改源数据
        bn_param['running_mean'] = running_mean.detach()
        bn_param['running_var'] = running_var.detach()
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        批归一化的更快的反向传播方法。
        输入/输出: 与batchnorm_backward相同
        """
        x, x_hat, mean, var, gamma, rsqrt, eps = cache
        N, D = x.shape
        dx_hat = gamma * dout
        # 可以看稿纸，或者 https://kevinzakka.github.io/2016/09/14/batch_normalization/
        dx = 1. / N * rsqrt * (N * dx_hat - dx_hat.sum(dim=0) - x_hat * (dx_hat * x_hat).sum(dim=0))  # 74字符除去空格
        dgamma, dbeta = (x_hat * dout).sum(dim=0), dout.sum(dim=0)
        return dx, dgamma, dbeta

    @staticmethod
    def backward_origin(dout, cache):
        """
        原始批归一化的反向传播。

        输入:
        - dout: 上游导数，形状为(N, D)
        - cache: 从batchnorm_forward得到的中间变量

        返回一个元组:
        - dx: 相对于输入x的梯度，形状为(N, D)
        - dgamma: 相对于缩放参数gamma的梯度，形状为(D,)
        - dbeta: 相对于平移参数beta的梯度，形状为(D,)
        """
        dx, dgamma, dbeta = None, None, None
        # original paper (https://arxiv.org/abs/1502.03167) 

        # @ 读论文，这个实现不要求速度
        x, x_hat, mean, var, gamma, rsqrt, eps = cache
        N, D = x.shape
        dx = torch.zeros_like(x)
        # ! 一定要注意单位、设备相同，不然有误差
        dsigma2, dmu = torch.zeros([D], dtype=dout.dtype, device=dout.device), torch.zeros([D], dtype=dout.dtype,
                                                                                           device=dout.device)
        # @ gamma和beta都是行向量，每一列对应一个
        dgamma = (x_hat * dout).sum(dim=0)
        dbeta = dout.sum(dim=0)
        dx_hat = gamma * dout

        # # 向量化
        dsigma2 = 0.5 * ((var + eps) ** (-1.5)) * (dx_hat * (mean - x)).sum(dim=0)
        dmu = -rsqrt * dx_hat.sum(dim=0) + dsigma2 * (-2 / N) * (x - mean).sum(dim=0)
        dx = dx_hat * rsqrt + dsigma2 * (2. / N) * (x - mean) + 1. / N * dmu

        # 来自纸上推导
        return dx, dgamma, dbeta


class SpatialBatchNorm(object):
    @staticmethod
    def forward(x, gamma, beta, bn_param):
        """
        计算空间批归一化的前向传播。

        输入:
        - x: 输入数据，形状为(N, C, H, W)
        - gamma: 缩放参数，形状为(C,)
        - beta: 平移参数，形状为(C,)
        - bn_param: 字典，包含以下键值:
        - mode: 'train' 或 'test'；必需
        - eps: 数值稳定性的常数
        - momentum: 运行均值/方差的常数。momentum=0 表示旧信息在每个时间步骤完全丢弃，而 momentum=1 表示不会合并新信息。默认的 momentum=0.9 在大多数情况下效果很好。
        - running_mean: 形状为(C,)的数组，给出特征的运行均值
        - running_var: 形状为(C,)的数组，给出特征的运行方差

        返回一个元组:
        - out: 输出数据，形状为(N, C, H, W)
        - cache: 反向传播所需的值
        """
        N, C, H, W = x.shape
        out, cache = BatchNorm.forward(x.reshape(-1, C), gamma, beta, bn_param)  # out.shape == x.shape
        out = out.reshape(N, C, H, W)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        计算空间批归一化的反向传播。

        输入:
        - dout: 上游导数，形状为(N, C, H, W)
        - cache: 前向传播过程中的值

        返回一个元组:
        - dx: 相对于输入的梯度，形状为(N, C, H, W)
        - dgamma: 相对于缩放参数的梯度，形状为(C,)
        - dbeta: 相对于平移参数的梯度，形状为(C,)
        """
        dx, dgamma, dbeta = None, None, None

        N, C, H, W = dout.shape
        dx, dgamma, dbeta = BatchNorm.backward(dout.view(-1, C), cache)
        dx = dx.view(N, C, H, W)

        return dx, dgamma, dbeta


# 用torch的快速实现
class FastConv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        stride, pad = conv_param['stride'], conv_param['pad']
        layer = torch.nn.Conv2d(C, F, (HH, WW), stride=stride, padding=pad)
        layer.weight = torch.nn.Parameter(w)
        layer.bias = torch.nn.Parameter(b)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, w, b, conv_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, _, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
            dw = layer.weight.grad.detach()
            db = layer.bias.grad.detach()
            layer.weight.grad = layer.bias.grad = None
        except RuntimeError:
            dx, dw, db = torch.zeros_like(tx), \
                         torch.zeros_like(layer.weight), \
                         torch.zeros_like(layer.bias)
        return dx, dw, db


class FastMaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        N, C, H, W = x.shape
        pool_height, pool_width = \
            pool_param['pool_height'], pool_param['pool_width']
        stride = pool_param['stride']
        layer = torch.nn.MaxPool2d(kernel_size=(pool_height, pool_width),
                                   stride=stride)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, pool_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
        except RuntimeError:
            dx = torch.zeros_like(tx)
        return dx


class Conv_ReLU(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        """
        一个方便的层，执行卷积，然后是ReLU。
        输入:
        - x: 卷积层的输入
        - w, b, conv_param: 卷积层的权重和参数
        返回一个元组:
        - out: ReLU的输出
        - cache: 用于反向传播的对象
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        out, relu_cache = ReLU.forward(a)
        cache = (conv_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        反向传播计算梯度
        """
        conv_cache, relu_cache = cache
        da = ReLU.backward(dout, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db


class Conv_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, b, conv_param, pool_param):
        """
        执行卷积，ReLU和池化。
        输入:
        - x: 卷积层的输入
        - w, b, conv_param: 卷积层的权重和参数
        - pool_param: 池化层的参数
        返回一个元组:
        - out: 池化层的输出
        - cache: 用于反向传播的对象
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        s, relu_cache = ReLU.forward(a)
        out, pool_cache = FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        反向传播计算梯度
        """
        conv_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        da = ReLU.backward(ds, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db


class Linear_BatchNorm_ReLU(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, bn_param):
        """
        执行线性变换，批归一化和ReLU。
        输入:
        - x: 形状为(N, D1)的数组；线性层的输入
        - w, b: 形状分别为(D1, D2)和(D2,)的数组，给出线性变换的权重和偏差。
        - gamma, beta: 形状为(D2,)的数组，给出批归一化的缩放和平移参数。
        - bn_param: 批归一化的参数字典。
        返回:
        - out: ReLU的输出，形状为(N, D2)
        - cache: 用于反向传播的对象
        """
        a, fc_cache = Linear.forward(x, w, b)
        a_bn, bn_cache = BatchNorm.forward(a, gamma, beta, bn_param)
        out, relu_cache = ReLU.forward(a_bn)
        cache = (fc_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        反向传播计算梯度
        """
        fc_cache, bn_cache, relu_cache = cache
        da_bn = ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = BatchNorm.backward(da_bn, bn_cache)
        dx, dw, db = Linear.backward(da, fc_cache)
        return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        an, bn_cache = SpatialBatchNorm.forward(a, gamma,
                                                beta, bn_param)
        out, relu_cache = ReLU.forward(an)
        cache = (conv_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache = cache
        dan = ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param, pool_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        an, bn_cache = SpatialBatchNorm.forward(a, gamma, beta, bn_param)
        s, relu_cache = ReLU.forward(an)
        out, pool_cache = FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, bn_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        dan = ReLU.backward(ds, relu_cache)
        da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db, dgamma, dbeta
