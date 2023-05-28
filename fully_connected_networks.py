"""
注意:
1. "层"类除了BN层不可以改变输入数据(源数据)！

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
from toolset.helper import softmax_loss, svm_loss


class Linear(object):

    @staticmethod
    def forward(x, w, b):
        """
        计算线性（全连接）层的前向传播。
        输入x的形状为(N, d_1, ..., d_k)，其中包含了N个示例的小批量数据，每个示例x[i]的形状为(d_1, ..., d_k)。
        我们将每个输入重新调整为维度为D = d_1 * ... * d_k的向量，然后将其转换为维度为M的输出向量。
        输入：
        - x：包含输入数据的张量，形状为(N, d_1, ..., d_k)
        - w：权重张量，形状为(D, M)
        - b：偏置张量，形状为(M,)
        返回一个元组：
        - out：输出，形状为(N, M)
        - cache：元组(x, w, b)
        """
        out = x.view(x.shape[0], -1).mm(w) + b
        cache = (x, w, b)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        计算线性层的反向传播。
        输入：
        - dout：上游导数，形状为(N, M)
        - cache：元组，包含以下内容：
          - x：输入数据，形状为(N, d_1, ... d_k)
          - w：权重，形状为(D, M)
          - b：偏置，形状为(M,)
        返回一个元组：
        - dx：相对于x的梯度，形状为(N, d1, ..., d_k)
        - dw：相对于w的梯度，形状为(D, M)
        - db：相对于b的梯度，形状为(M,)
        """
        x, w, b = cache
        dx = dout.mm(w.T).view(x.shape)  # 转成x的形态
        dw = x.view(x.shape[0], -1).t().mm(dout)  # 先转成(N,D)
        db = dout.sum(dim=0)
        return dx, dw, db


class ReLU(object):

    @staticmethod
    def forward(x):
        """
        计算修正线性单元（ReLU）层的前向传播。
        输入：
        - x：输入；任意形状的张量
        返回一个元组：
        - out：输出，与x形状相同的张量
        - cache：x
        """
        out = x.clone()
        out[out < 0] = 0
        cache = x
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        计算修正线性单元（ReLU）层的反向传播。

        输入：
        - dout：上游导数，任意形状的张量
        - cache：输入x，与dout形状相同
        返回：
        - dx：相对于x的梯度
        """
        dx, x = dout.clone(), cache
        dx[x < 0] = 0
        return dx


class Linear_ReLU(object):

    @staticmethod
    def forward(x, w, b):
        """
        先进行线性变换，然后进行ReLU操作。

        输入：
        - x：线性层的输入
        - w, b：线性层的权重
        返回一个元组：
        - out：ReLU的输出
        - cache：传递给反向传播的对象
        """
        a, fc_cache = Linear.forward(x, w, b)
        out, relu_cache = ReLU.forward(a)  # relu_cache是输入relu之前的东西，和a相同
        cache = (fc_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        线性-ReLU层的反向传播
        """
        fc_cache, relu_cache = cache
        da = ReLU.backward(dout, relu_cache)
        dx, dw, db = Linear.backward(da, fc_cache)
        return dx, dw, db



class TwoLayerNet(object):
    """
    一个具有ReLU和softmax的两层全连接神经网络，采用模块化层设计。
    我们假设输入维度为D，隐藏维度为H，并在C个类别上进行分类。
    网络的架构应为线性层 - ReLU激活 - 线性层 - softmax激活。

    注意，该类不实现梯度下降；将与负责运行优化的单独的Solver对象进行交互。

    模型的可学习参数存储在self.params字典中，将参数名称映射到PyTorch张量。
    """

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0001,
                 dtype=torch.float32, device='cpu'):  # ! 这里源代码 是cpu
        """
        初始化一个新的神经网络。
        输入：
        - input_dim：输入的大小，一个整数
        - hidden_dim：隐藏层的大小，一个整数
        - num_classes：要分类的类别数量，一个整数
        - weight_scale：权重随机初始化的标准差，一个标量
        - reg：L2正则化的强度，一个标量
        - dtype：torch数据类型对象；所有计算将使用此数据类型进行。float速度更快但精度较低，因此在数值梯度检查时应使用double类型。
        - device：用于计算的设备。'cpu'或'cuda'
        """
        self.params = {}
        self.reg = reg

        # 初始化两层网络的权重和偏置项。
        # 权重应从均值为0.0、标准差为weight_scale的高斯分布中初始化，
        # 偏置项初始化为0。所有权重和偏置项应存储在self.params字典中，
        # 第一层的权重和偏置项使用键'W1'和'b1'，第二层的权重和偏置项使用键'W2'和'b2'。

        self.params['W1'] = weight_scale * torch.randn(input_dim, hidden_dim, dtype=dtype, device=device)
        self.params['b1'] = weight_scale * torch.zeros(hidden_dim, dtype=dtype, device=device)
        self.params['W2'] = weight_scale * torch.randn(hidden_dim, num_classes, dtype=dtype, device=device)
        self.params['b2'] = weight_scale * torch.zeros(num_classes, dtype=dtype, device=device)

    def save(self, path):
        checkpoint = {
            'reg': self.reg,
            'params': self.params,
        }  # 数据字典

        torch.save(checkpoint, path)  # 用了pickle，对tensor进行了优化处理
        # https://pytorch.org/docs/stable/generated/torch.save.html?highlight=save#torch.save
        # https://pytorch.org/docs/stable/generated/torch.load.html?highlight=load#torch.load
        # https://docs.python.org/3/library/pickle.html
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.reg = checkpoint['reg']
        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)
        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        计算一个小批量数据的损失和梯度。

        输入：
        - X：输入数据的张量，形状为(N, d_1, ..., d_k)
        - y：标签的int64型张量，形状为(N,)。y[i]表示X[i]的标签。

        返回：
        如果y为None，则运行模型的测试时前向传播，并返回：
        - scores：形状为(N, C)的张量，给出分类分数，其中scores[i, c]是X[i]和类别c的分类分数。
        如果y不为None，则运行训练时的前向传播和反向传播，并返回一个元组：
        - loss：标量值，表示损失
        - grads：字典，具有与self.params相同的键，将参数名称映射到相对于这些参数的损失的梯度。
        """
        scores = None

        # $ linear - relu - linear, softmax是计算loss的时候才加，毕竟加不加softmax预测结果都不变
        a_out, cache_LR_1 = Linear_ReLU.forward(X, self.params['W1'], self.params['b1'])
        scores, cache_L_2 = Linear.forward(a_out, self.params['W2'], self.params['b2'])

        if y is None:
            return scores  # （N, C)

        grads = {}
        loss, dS = softmax_loss(scores, y)
        loss += self.reg * ((self.params['W1'] ** 2).sum() + (self.params['W2'] ** 2).sum())
        dR, dW2, db2 = Linear.backward(dout=dS, cache=cache_L_2)
        grads['W2'] = dW2 + 2 * self.reg * self.params['W2']
        grads['b2'] = db2

        dx, dW1, db1 = Linear_ReLU.backward(dR, cache_LR_1)
        grads['W1'] = dW1 + 2 * self.reg * self.params['W1']
        grads['b1'] = db1
        return loss, grads

    # @ 这是不用函数手动计算loss的函数，主要用于回顾，同时还发现了之前实现中的bug
    def loss_manual(self, X, y=None):
        """
        计算一个小批量数据的损失和梯度。

        输入：
        - X：输入数据的张量，形状为(N, d_1, ..., d_k)
        - y：标签的int64型张量，形状为(N,)。y[i]表示X[i]的标签。

        返回：
        如果y为None，则运行模型的测试时前向传播，并返回：
        - scores：形状为(N, C)的张量，给出分类分数，其中scores[i, c]是X[i]和类别c的分类分数。
        如果y不为None，则运行训练时的前向传播和反向传播，并返回一个元组：
        - loss：标量值，表示损失
        - grads：字典，具有与self.params相同的键，将参数名称映射到相对于这些参数的损失的梯度。
        """
        # $ linear - relu - linear, softmax是计算loss的时候才加，毕竟加不加softmax预测结果都不变
        a_out, cache_LR_1 = Linear_ReLU.forward(X, self.params['W1'], self.params['b1'])
        scores, cache_L_2 = Linear.forward(a_out, self.params['W2'], self.params['b2'])

        # 如果y为None，则处于测试模式，只需返回分数
        if y is None:
            return scores  # （N, C)
        loss, grads = 0, {}
        # 两层网络的反向传播。
        # 将损失存储在loss变量中，梯度存储在grads字典中。
        # 使用softmax计算数据损失，并确保grads[k]保存了self.params[k]的梯度。
        # 添加L2正则化。
        N = scores.shape[0]
        shifted_logits = scores - scores.max(dim=1, keepdim=True).values
        Z = shifted_logits.exp().sum(dim=1, keepdim=True)
        log_probs = shifted_logits - Z.log()
        probs = log_probs.exp()
        L1 = -1.0 / N * log_probs[torch.arange(N), y].sum()  # 损失就是-log probs
        L2 = self.reg * \
             ((self.params['W1'] ** 2).sum() + (self.params['W2'] ** 2).sum())
        loss = L1 + L2
        dS = probs
        dS[torch.arange(N), y] -= 1
        dS /= N

        fc_cache, relu_cache = cache_LR_1

        # NOTE
        # ! bug: grads['W2'] = relu_cache.t().mm(dS) + 2 * self.reg * self.params['W2']
        # @ 修复就是将relu_cache修改为a_out，因为计算W2的梯度需要用的是relu之后的值而非relu之前的值relu_cache
        # 这个bug找了好久，其中一个原因是这个bug只会导致网络的性能比正确的实现仅仅低3~5个百分点左右，
        # 实际上还是能到50%正确率，你可以自己训练一下。我觉得一般人都不会注意到这个问题，
        # 或者不把他视为一个问题，但实际上任何细微的变化都应该被察觉发现解决。
        # 如果这个东西写在框架里了并且实现了一些应用，之后再查只会难度指数增加
        grads['W2'] = a_out.t().mm(dS) + 2 * self.reg * self.params['W2']
        grads['b2'] = dS.sum(dim=0)

        grad_h1 = dS.mm(self.params['W2'].T)  # h1
        grad_h0 = grad_h1.clone()

        grad_h0[a_out == 0] = 0
        # ! 之前grad_h0[relu_cache < 1e-6] 实现是不对的，可能relu_cache里面有真的非0但是<1e-6的值
        # $ a_out == 0可能存在隐患，虽然relu之后有0，因为在实际操作中，计算机可能无法准确地表示浮点数，
        # 从而导致误差。虽然这种方法在某些情况下可能有效，但为了避免潜在问题，最好使用ReLU激活函数的输入值来计算梯度，即y=xw+b。
        # grad_h0[relu_cache == 0] = 0
        grads['W1'] = X.T.mm(grad_h0) + 2 * self.reg * self.params['W1']
        grads['b1'] = grad_h0.sum(dim=0)
        return loss, grads

class FullyConnectedNet(object):
    """
    一个具有任意数量隐藏层、ReLU非线性激活和softmax损失函数的全连接神经网络。
    对于具有L层的网络，其架构如下：

    { 线性层 - ReLU激活 - [dropout] } x (L - 1) - 线性层 - softmax激活

    其中，dropout是可选的，{...}块重复出现L - 1次。

    与上面的TwoLayerNet类似，可学习参数存储在self.params字典中，并将使用Solver类进行学习。
    """

    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
                 dropout=0.0, reg=0.0, weight_scale=1e-2, seed=None,
                 dtype=torch.float, device='cpu'):
        """
        初始化一个新的全连接神经网络。

        输入：
        - hidden_dims：一个整数列表，给出每个隐藏层的大小。
        - input_dim：输入的大小，一个整数。
        - num_classes：要分类的类别数量，一个整数。
        - dropout：在具有dropout的网络中，介于0和1之间的标量，表示丢弃的概率。
          如果dropout=0，则网络不应使用dropout。
        - reg：L2正则化的强度，一个标量。
        - weight_scale：权重随机初始化的标准差，一个标量。
        - seed：如果不为None，则将此随机种子传递给dropout层。
          这将使dropout层确定性，以便我们可以对模型进行梯度检查。
        - dtype：torch数据类型对象；所有计算将使用此数据类型进行。
          float类型速度更快但精度较低，因此在数值梯度检查时应使用double类型。
        - device：用于计算的设备。'cpu'或'cuda'
        """
        self.use_dropout = dropout != 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        # 初始化网络的参数，将所有值存储在self.params字典中。
        # 将第一层的权重和偏置项存储在W1和b1中；将第二层的权重和偏置项存储在W2和b2中，
        # 依此类推。权重从均值为0、标准差为weight_scale的正态分布中进行初始化，
        # 偏置项初始化为0。

        cnt_hidden = len(hidden_dims)
        for i in range(cnt_hidden):
            idx = str(i + 1)
            if i == 0:
                self.params['W' + idx] = weight_scale * torch.randn(input_dim, hidden_dims[i], dtype=dtype,
                                                                    device=device)
            else:
                self.params['W' + idx] = weight_scale * \
                                         torch.randn(hidden_dims[i - 1], hidden_dims[i], dtype=dtype, device=device)
            self.params['b' + idx] = weight_scale * torch.zeros(hidden_dims[i], dtype=dtype, device=device)

        self.params['W' + str(cnt_hidden + 1)] = \
            weight_scale * torch.randn(hidden_dims[-1], num_classes, dtype=dtype, device=device)
        self.params['b' + str(cnt_hidden + 1)] = weight_scale * torch.zeros(num_classes, dtype=dtype, device=device)


        # 当使用dropout时，我们需要向每个dropout层传递一个dropout_param字典，
        # 以便该层了解dropout的概率和模式（训练/测试）。可以将相同的dropout_param传递给每个dropout层。
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

    def save(self, path):
        checkpoint = {
            'reg': self.reg,
            'dtype': self.dtype,
            'params': self.params,
            'num_layers': self.num_layers,
            'use_dropout': self.use_dropout,
            'dropout_param': self.dropout_param,
        }

        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = dtype
        self.reg = checkpoint['reg']
        self.num_layers = checkpoint['num_layers']
        self.use_dropout = checkpoint['use_dropout']
        self.dropout_param = checkpoint['dropout_param']

        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)

        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        计算全连接神经网络的损失和梯度。
        输入：
        - X：输入数据的张量，形状为(N, d_1, ..., d_k)
        - y：标签的int64型张量，形状为(N,)。y[i]表示X[i]的标签。

        返回：
        如果y为None，则运行模型的测试时前向传播，并返回：
        - scores：形状为(N, C)的张量，给出分类分数，其中scores[i, c]是X[i]和类别c的分类分数。
        如果y不为None，则运行训练时的前向传播和反向传播，并返回一个元组：
        - loss：标量值，表示损失
        - grads：字典，具有与self.params相同的键，将参数名称映射到相对于这些参数的损失的梯度。
        """
        X = X.to(self.dtype)
        mode = 'test' if y is None else 'train'  # 没有y就是测试推理模式(不使用dropout)

        # 设置批归一化参数和dropout参数的训练/测试模式，因为它们在训练和测试过程中的行为不同。
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        scores = None

        # 前向传播，计算X的类别分数，存储在scores变量中
        # 当使用dropout时，用self.dropout_param传递给每个dropout层的前向传播

        cnt_layer = self.num_layers
        last_out = X
        fc_relu_cache = {}
        dropout_cache = {}

        for i in range(1, cnt_layer):
            w, b = self.params['W{}'.format(i)], self.params['b{}'.format(i)]
            last_out, cache = Linear_ReLU.forward(last_out, w, b)
            if self.use_dropout:
                # 模式(train,test)的区别，Dropout类里面处理好了，test就直接输出
                last_out, d_cache = Dropout.forward(
                    last_out, self.dropout_param)
                dropout_cache[i] = d_cache
            fc_relu_cache[i] = cache

        # 最后一层前向传播
        last_out, fc_relu_cache[cnt_layer] = Linear.forward(
            last_out, self.params['W{}'.format(cnt_layer)], self.params['b{}'.format(cnt_layer)])
        scores = last_out

        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        # 反向传播
        # 损失存储在loss变量中，梯度存储在grads字典中
        # 用softmax计算数据损失，并确保grads[k]保存了self.params[k]的梯度
        # 添加L2正则化。

        loss, dS = softmax_loss(scores, y)
        reg_loss = 0
        for i in range(1, cnt_layer + 1):
            reg_loss += self.reg * (self.params['W{}'.format(i)] ** 2).sum()
        loss += reg_loss

        dout, dw, db = Linear.backward(dout=dS, cache=fc_relu_cache[cnt_layer])
        grads['W{}'.format(cnt_layer)] = dw + 2 * self.reg * self.params['W{}'.format(cnt_layer)]
        grads['b{}'.format(cnt_layer)] = db

        for i in range(cnt_layer - 1, 0, -1):
            if self.use_dropout:
                dout = Dropout.backward(dout, dropout_cache[i])
            dout, dw, db = Linear_ReLU.backward(
                dout=dout, cache=fc_relu_cache[i])  # ! dout=dout而不是dS
            grads['W{}'.format(i)] = dw + 2 * self.reg * self.params['W' + str(i)]
            grads['b{}'.format(i)] = db

        return loss, grads


class Dropout(object):

    @staticmethod
    def forward(x, dropout_param):
        """
        执行（反向）dropout的前向传播。
        输入：
        - x：输入数据，任意形状的张量。
        - dropout_param：包含以下键的字典：
          - p：dropout参数。我们以概率p**丢弃**每个神经元的输出。
          - mode：'test'或'train'。如果模式为'train'，则执行dropout；
            如果模式为'test'，则只返回输入。
        输出：
        - out：与x形状相同的张量。
        - cache：元组（dropout_param，mask）。在训练模式下，mask是用于乘以输入的dropout掩码；
          在测试模式下，mask为None。
        注意：p是丢弃神经元的概率，不是保留神经元的概率
        """
        p, mode = dropout_param['p'], dropout_param['mode']
        out, mask = None, None

        if mode == 'train':
            # 可以对bool数组做除法操作，True为1
            mask = (torch.rand(x.shape, device=x.device) > p) / (1 - p)  # 这里p是指丢掉的输出比例，所以大于p保留，除以1-p保持缩放
            out = x * mask
        elif mode == 'test':
            out = x
        cache = (dropout_param, mask)

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        执行（反向）dropout的反向传播。
        输入：
        - dout：上游导数，任意形状的张量。
        - cache：来自Dropout.forward的（dropout_param，mask）元组。
        """
        dropout_param, mask = cache
        mode = dropout_param['mode']

        dx = None
        if mode == 'train':
            dx = dout * mask
        elif mode == 'test':
            dx = dout
        return dx



