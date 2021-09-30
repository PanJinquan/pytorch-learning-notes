# -*-coding: utf-8 -*-
"""
    @Project: pytorch-learning-tutorials
    @File   : main2.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-07-07 13:20:23
"""
import torch
import torchvision.models as models
from collections import OrderedDict

if __name__=="__main__":
    resnet18 = models.resnet18(False)
    print("resnet18",resnet18)

    # use named_children()
    resnet18_v1 = OrderedDict(resnet18.named_children())
    # remove avgpool,fc
    resnet18_v1.pop("avgpool")
    resnet18_v1.pop("fc")
    resnet18_v1 = torch.nn.Sequential(resnet18_v1)
    print("resnet18_v1",resnet18_v1)
    # use children
    resnet18_v2 = torch.nn.Sequential(*list(resnet18.children())[:-2])
    print(resnet18_v2,resnet18_v2)
