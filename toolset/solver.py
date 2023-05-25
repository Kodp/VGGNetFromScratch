import pickle  # 序列化反序列化
import time
import torch
import copy

class Solver(object):  # 求解器
    """

    """
    def __init__(self, model, data, **kwargs):
        """
        构造求解器实例。

        Args:
            model: model，包含上述API
            data: 训练数据和验证数据的字典
                'X_train': Array, shape (N_train, d_1, ..., d_k) of training images
                'X_val': Array, shape (N_val, d_1, ..., d_k) of validation images
                'y_train': Array, shape (N_train,) of labels for training images
                'y_val': Array, shape (N_val,) of labels for validation images

            **kwargs: 可选参数:
            - update_rule: 更新规则，默认sgd。
            - optim_config: 包含对应于更新规则的字典，不同的更新规则需要不同的参数，但是所有的规则都需要包含`learning_rate`键。
            - lr_decay: 标量，每轮(epoch)之后learning_rate会乘以这个值。
            - batch_size: 训练/loss计算的一批的数量。
            - num_epochs: 训练轮数。
            - pring_every: 整数，表示每多少次迭代就打印信息(iteration)
            - print_acc_every: 整数，表示每多少次迭代就打印准确率(iteration)
            - verbose: 布尔类型，如果设置为False则去除所有打印信息，默认为True
            - num_train_samples: 用于计算训练准确率的样本数，默认1000，传入None则为所有训练数据
            - num_val_sample: 用于计算验证准确率的样本数，设为None则用所有的验证数据
            - checkpoint_name: 如果不为None则每轮都保存model
        """
        self.model = model
        self.X_train, self.y_train = data["X_train"], data["y_train"]
        self.X_val, self.y_val = data["X_val"], data["y_val"]
        # 解包kwargs
        self.update_rule = kwargs.pop("update_rule", self.sgd)  # pop是有则取，没有则取第二项。然后删除kwargs中的对应键。
        self.optim_config = kwargs.pop("optim_config", {})
        self.optim_config.setdefault('learning_rate', 1e-4)  # 设置了lr就用，没有则默认1e-4
        self.learning_rate = self.optim_config['learning_rate']
        self.lr_decay = kwargs.pop("lr_decay", 1.0)
        self.batch_size = kwargs.pop("batch_size", 100)
        self.num_epochs = kwargs.pop("num_epochs", 10)
        self.num_train_samples = kwargs.pop("num_train_samples", 1000)
        self.num_val_samples = kwargs.pop("num_train_samples", None)

        self.device = kwargs.pop("device", "cpu")
        self.checkpoint_name = kwargs.pop("checkpoint_name", None)
        self.print_every = kwargs.pop("print_every", 10)
        self.print_acc_every = kwargs.pop("print_acc_every", 1)
        self.verbose = kwargs.pop("verbose", True)
        self.check_batch_size = kwargs.pop("check_batch_size", 200)

        # 如果存在未识别的参数则报错
        if len(kwargs) > 0:
            extra = ", ".join(f'"{key}"' for key in kwargs.keys())
            raise ValueError(f"未识别的求解器字典参数: {extra}")
        self._reset()

    def _reset(self):
        """
        设定一些临时变量。不要手动调用
        """
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}  # 会在训练中赋给model，没有保存
        self.best_bn_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        """
           在许多优化算法中，如包含动量（momentum）的随机梯度下降（SGD）、Adam等，每个参数矩阵（权重或偏置）都会有一些独特的状态需要维护。
           这些状态可能包括上一次的梯度、历史梯度的移动平均值等。因此，虽然全局的学习率是一样的，但是每个参数的其他优化信息可能需要单独保存。
           所以，我们为每个权重(W1,b1,W2,b2,etc.)创建一个优化配置字典来保存这些信息。

           举个例子，如果我们使用包含动量的SGD，动量是根据过去的梯度来影响当前的权重更新。
           所以我们需要保存每个权重在之前的迭代中的梯度。这就是为什么每个参数需要自己的优化配置字典。
           尽管学习率和动量的系数可能对所有的参数都是相同的，但是动量的“速度”（也就是过去的梯度）是对每个参数独立的。

           这也是为什么代码中会创建一个新的优化配置字典self.optim_configs[p] = d的原因。
           其中，p是参数的名称，d是对应的优化配置。因此，self.optim_configs是一个字典，其键是参数的名称，值是对应参数的优化配置。
        """

        # 将模型的参数深拷贝到optim_configs 注意不是 optim_config
        self.optim_configs = {}  # k:v, k为单个权重的名称(W1,b1,etc.)，v为初始优化配置字典，每个权重都有优化配置字典
        for p in self.model.params:  # 遍历key，也是model的参数权重的名称
            d = {k: v for k, v in self.optim_config.items()}  # 拷贝字典到d
            self.optim_configs[p] = d # 每个权重一个配置字典

    def _step(self):
        """
        单步梯度更新，由train调用，不要手动调用
        """
        num_train = self.X_train.shape[0]
        batch_mask = torch.randperm(num_train)[: self.batch_size]  # [0,num_train-1]的随机序列，取batchsize个

        X_batch = self.X_train[batch_mask].to(self.device)
        y_batch = self.y_train[batch_mask].to(self.device)

        # 计算loss和梯度
        loss, grads = self.model.loss(X_batch, y_batch)
        self.loss_history.append(loss.item())
        # 更新参数
        with torch.no_grad():
            for w_name, w in self.model.params.items():  # 权重名称，权重
                dw = grads[w_name]
                config = self.optim_configs[w_name]
                next_w, next_config = self.update_rule(w, dw, config)
                # model.params将要取新的w，所以update_rule的实现可以in_place
                self.model.params[w_name] = next_w  # 更新权重
                self.optim_configs[w_name] = next_config  # 更新优化配置字典


    def _save_checkpoint(self):
        """
        保存对象到字节流，包含model，还有训练记录
        """
        if self.checkpoint_name is None:
            return
        checkpoint = {
            'model': self.model,
            'update_rule': self.update_rule,
            'optim_config': self.optim_config,
            'learning_rate': self.learning_rate,
            'lr_decay': self.lr_decay,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'num_train_samples': self.num_train_samples,
            'num_val_samples': self.num_val_samples,
            
            'epoch': self.epoch,
            'best_val_acc': self.best_val_acc,
            'loss_history': self.loss_history,
            'train_acc_history': self.train_acc_history,
            'val_acc_history': self.val_acc_history,
        }
        filename = f"{self.checkpoint_name}_epoch_{self.epoch}.pkl"
        if self.verbose:
            print(f"保存检查点到:{filename}")
        with open(filename, "wb") as f:
            pickle.dump(checkpoint, f)

    @staticmethod
    def sgd(w, dw, config=None):
        """
        执行随机梯度下降(sgd)算法。
        config:

        Args:
            w: 权重
            dw: 权重的梯度
            config: 字典
                - learing_rate: 标量学习率

        Returns:
            w: 更新后的权重
            config: 更新后的字典
                一些更新方法例如SGDMomentum，每个梯度都会有自己不同的"平均速度",这个是自己独享的，所以要有
        """
        if config is None:
            config = {}
        config.setdefault("learning_rate", 1e-2)
        w -= config["learning_rate"] * dw
        return w, config
    

    def check_accuracy(self, X, y, num_samples=None) -> float:
        r"""
        提供数据，返回准确率
        Args:
            X: tensor，shape (N, d_1, ..., d_k)
            y: tensor，shape (N,)
            num_samples: 选择指定数量的样本进行测试，为空则全部样本

        Returns:
            accs: 标量，预测正确的比例(准确率)
        """
        batch_size = self.check_batch_size  # 默认200张
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = torch.randperm(N, device=self.device)[:num_samples]
            N = num_samples
            X = X[mask]
            y = y[mask]
        X, y = X.to(self.device), y.to(self.device)

        num_batches = N // batch_size
        num_batches += (N % batch_size) != 0  # 除不尽则加一个batch
        y_pred = []

        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.loss(X[start:end])  # y为None就只推理
            y_pred.append(torch.argmax(scores, dim = 1))

        # cat用法示例:
        #>>> y = [torch.tensor([1,2,1]),torch.tensor([1,4,1]),torch.tensor([1,3,1])]
        #>>> torch.cat(y)
        #>>> tensor([1, 2, 1, 1, 4, 1, 1, 3, 1])
        # 连接所有预测结果
        y_pred = torch.cat(y_pred) # (N,)
        acc = (y_pred == y).to(torch.float).mean()

        return acc.item()

    def train(self, time_limit=None, return_best_params=True):
        """
        训练模型(model)。

        Args:
            time_limit: 训练时间限制，默认没有
            return_best_params: 若设为True，则将model的参数设为self.best_params

        Returns:
            None
        """
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)  # 每个epoch走的迭代次数, 一个epoch将会遍历num_train个样本
        num_iterations = self.num_epochs * iterations_per_epoch
        prev_time = start_time = time.time()

        for t in range(num_iterations):
            cur_time = time.time()

            if (time_limit is not None) and (t > 0):
                prev_time_use = cur_time - prev_time  # 上一轮用的时间
                if cur_time - start_time + prev_time_use > time_limit:  # 已用时间+上一轮的时间，估计为经过下一轮后的总用时
                    print(f"(用时 {cur_time - start_time:.2f}s; 迭代次数{t} / {num_iterations}) loss: {self.loss_history[-1]:.6f}")
                    print("下一轮会超时，结束训练。")
                    break
            prev_time = cur_time

            self._step() # 单步梯度更新

            if self.verbose and t % self.print_every == 0:
                print(f"(用时 {time.time() - start_time:.2f}s; 迭代次数{t + 1} / {num_iterations}) loss: {self.loss_history[-1]:.6f}")


            # 当前epoch结束标记
            epoch_end = (t + 1) % iterations_per_epoch == 0

            if epoch_end:
                self.epoch += 1
                # 更新学习率
                for k in self.optim_configs:
                    self.optim_configs[k]["learning_rate"] *= self.lr_decay

            # 检查train和val准确率，每轮epoch计算
            
            first_it = t == 0
            last_it = t == num_iterations - 1
            
            if epoch_end or first_it or last_it:
                with torch.no_grad():
                    train_acc = self.check_accuracy(self.X_train, self.y_train, self.num_train_samples)
                    val_acc = self.check_accuracy(self.X_val, self.y_val, self.num_val_samples)

                    self.train_acc_history.append(train_acc)
                    self.val_acc_history.append(val_acc)
                    self._save_checkpoint()

                    if self.verbose and self.epoch % self.print_acc_every == 0:
                        print(f"(Epoch {self.epoch} / {self.num_epochs}, train 准确率: {train_acc*100:.2f}%, val准确率: {val_acc*100:.2f}%")

                    # 保存最好的模型
                    if val_acc > self.best_val_acc:
                        print(f"更新:当前准确率{val_acc*100:.2f}, 大于之前的最佳准确率{self.best_val_acc*100:.2f}\n")
                        self.best_val_acc = val_acc
                        for k, v in self.model.params.items():
                            self.best_params[k] = v.clone()
                            try:
                                self.best_bn_params = copy.deepcopy(self.model.bn_params)
                                print("复制BN参数成功")
                            except AttributeError:
                                pass
        #TODO 检查有效性
        if return_best_params:
            #! 由于带有bn的model需要额外的运行时BN参数，如果直接把self.best_params给到self.model.params,则会导致BN参数不匹配
            # 进而导致造成准确率低下。
            self.model.params = self.best_params
            try:
                self.model.bn_params = copy.deepcopy(self.best_bn_params)
            except AttributeError:
                pass





