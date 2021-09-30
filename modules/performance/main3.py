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


from utils import debug

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device='cpu'
print("-----device:{}".format(device))
print("-----Pytorch version:{}".format(torch.__version__))

if __name__ == "__main__":
    batch_size=1
    input_tensor = torch.randn(batch_size, 3, 112, 112).to(device)
    resnet18 = models.resnet18(pretrained=False).to(device)
    out18 = resnet18.forward(input_tensor)

    # test resnet18
    torch.cuda.synchronize()
    resnet_t0 = debug.TIME()
    for i in range(10):
        out18 = resnet18.forward(input_tensor)
    torch.cuda.synchronize()
    resnet_t1 = debug.TIME()
    print("resnet18       :{},run time:{}ms".format(out18.shape, debug.RUN_TIME(resnet_t1 - resnet_t0)/10))
