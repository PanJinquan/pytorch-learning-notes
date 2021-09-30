# -*-coding: utf-8 -*-
"""
    @Project: pytorch-learning-tutorials
    @File   : get_cnn_featuremap.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-06-27 18:57:36
"""
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable
import torchvision.transforms as transforms


def my_resnet18():
    conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    relu = nn.ReLU()
    maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    layer1 =nn.ModuleList([conv1, bn1, relu, maxpool])
    # BasicBlock=[nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
    #             nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    #             nn.ReLU(),
    #             nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
    #             nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]

    BasicBlock =nn.ModuleList( [nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                  nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                  nn.ReLU(),
                  nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                  nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)])
    layer1 += BasicBlock
    layer1 += BasicBlock
    out = nn.ModuleList(layer1)
    return out

if __name__ == '__main__':
    # net=my_resnet18()
    # print(net)
    resnet = models.resnet18(False)
    # resnet = models.resnet152(pretrained=True)
    modules = list(resnet.children())[:-2]  # delete the last fc layer.
    print(modules)
    print("----------------------------------------")
    convnet = nn.Sequential(*modules)
    print(convnet)
