# -*-coding: utf-8 -*-
"""
    @Project: pytorch-learning-tutorials
    @File   : main.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-06-27 13:46:20
"""
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
import os
from modules.performance.core import resnet
from modules.performance.core import custom_resnet

from utils import debug

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device='cpu'
print("-----device:{}".format(device))
print("-----Pytorch version:{}".format(torch.__version__))

if __name__ == "__main__":
    input_tensor = torch.randn(1, 3, 640, 640).to(device)
    print('input_tensor:', input_tensor.shape)
    vgg16 = models.vgg16(pretrained=False).to(device)
    resnet18 = models.resnet18(pretrained=False).to(device)
    pretrained_file = "/media/dm/dm2/project/pytorch-learning-tutorials/pretrained/resnet18-5c106cde.pth"
    # custom_resnet18 = resnet.resnet18(False).to(device)
    squeeze=models.SqueezeNet(version=1.0).to(device)

    print("squeeze:",squeeze)
    out16 = vgg16.forward(input_tensor)
    out16 = vgg16.forward(input_tensor)
    # test vgg16
    torch.cuda.synchronize()
    vgg16_t0 = debug.TIME()
    out16 = vgg16.forward(input_tensor)
    torch.cuda.synchronize()
    vgg16_t1 = debug.TIME()

    # test custom_resnet18
    torch.cuda.synchronize()
    custom_t0 = debug.TIME()
    out_custom18 = squeeze.forward(input_tensor)
    torch.cuda.synchronize()
    custom_t1 = debug.TIME()

    # test resnet18
    torch.cuda.synchronize()
    resnet_t0 = debug.TIME()
    out18 = resnet18.forward(input_tensor)
    torch.cuda.synchronize()
    resnet_t1 = debug.TIME()

    print("vgg16          :{},run time:{}ms".format(out16.shape, debug.RUN_TIME(vgg16_t1 - vgg16_t0)))
    print("resnet18       :{},run time:{}ms".format(out18[1:5], debug.RUN_TIME(resnet_t1 - resnet_t0)))
    print("custom_resnet18:{},run time:{}ms".format(out_custom18[1:5], debug.RUN_TIME(custom_t1 - custom_t0)))
