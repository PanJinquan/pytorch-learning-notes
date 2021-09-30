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
from transfer_model.core import resnet
from utils import debug

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device='cpu'
print("-----device:{}".format(device))
print("-----Pytorch version:{}".format(torch.__version__))

if __name__ == "__main__":
    input_tensor = torch.randn(1, 3, 100, 100).to(device)
    print('input_tensor:', input_tensor.shape)
    pretrained_file = "/media/dm/dm2/project/pytorch-learning-tutorials/pretrained/resnet18-5c106cde.pth"
    custom_resnet18 = resnet.resnet18(pretrained_file).to(device)
    print(custom_resnet18)
