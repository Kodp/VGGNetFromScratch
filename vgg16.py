import torch
from torch import Tensor
import matplotlib.pyplot as plt
import numpy as np
import toolset as ts
from toolset import *
from toolset.utils import *
from toolset.data import *
from toolset.helper import *
from toolset.solver import *

plt.rcParams['figure.figsize'] = (6, 4)
plt.rcParams['font.size'] = 10


reset_seed(0)
data_dict = ts.data.preprocess_cifar10(cuda=True, dtype=torch.float64, flatten=False,show_examples=False)
from convolutional_networks import DeepConvNet, VggNet
from fully_connected_networks import adam
num_train = 50
small_data = {
  'X_train': data_dict['X_train'][:num_train],
  'y_train': data_dict['y_train'][:num_train],
  'X_val': data_dict['X_val'][:5],
  'y_val': data_dict['y_val'][:5],
}
input_dims = data_dict['X_train'].shape[1:]

reset_seed(0)  # 模型加载时也用了torch，保证相同
net = VggNet(
    num_filters = (64, 64, 
                   128, 128, 
                   256, 256, 256, 256,
                   512, 512, 512, 512,
                   512, 512, 512, 512
                   ),
    max_pools = (1, 3, 7, 11, 15),
    num_FC = (4096, 4096, 10),
    weight_scale = 'kaiming',
    dropout=0.5,
    reg = 1e-5,
    device = 'cuda'
)
solver = Solver(net, small_data,
                num_epochs=1000, batch_size=50,
                optim_config={
                'learning_rate': 1e-4,
                },
                update_rule=adam,
                verbose=True, device='cuda',print_every=4, lr_decay=0.95)

solver.train(time_limit=60)