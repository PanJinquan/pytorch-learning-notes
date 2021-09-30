# -*-coding: utf-8 -*-
"""
    @Project: pytorch-learning-tutorials
    @File   : vision_test01.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-06-16 10:45:23
    @url    :《深度学习tensorwatch可视化工具》https://blog.csdn.net/qq_29592829/article/details/90517303
    conda install graphviz
    conda install torchvision
    conda install tensorwatch
"""

import torch
from torch import nn
from torchviz import make_dot, make_dot_from_trace
from torch.autograd import Variable

model = nn.Sequential()
model.add_module('W0', nn.Linear(8, 16))
model.add_module('tanh', nn.Tanh())
model.add_module('W1', nn.Linear(16, 1))

# x = Variable(torch.rand(1, 3,640, 640))
x = Variable(torch.rand(1, 8))
dot=make_dot(model(x), params=dict(model.named_parameters()))
dot.save("torchviz.gv","data") # save *.gv file
