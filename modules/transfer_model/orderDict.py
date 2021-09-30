# -*-coding: utf-8 -*-
"""
    @Project: pytorch-learning-tutorials
    @File   : orderDict.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-07-07 15:42:11
"""
import torch.nn as nn
from collections import OrderedDict
named_children=OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ])
model = nn.Sequential(named_children)
print(model)