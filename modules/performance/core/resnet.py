# -*-coding: utf-8 -*-
"""
    @Project: pytorch-learning-tutorials
    @File   : resnet.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-06-27 14:38:54
"""
import torch.nn as nn
import torch
from .torch_resnet import BasicBlock, Bottleneck, ResNet

def resnet18(pretrained_file=None, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    for name, child in model._modules.items():
        if child is not None:
            print(child)
    if pretrained_file:
        model.load_state_dict(torch.load(pretrained_file))
    return model

if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    print("-----device:{}".format(device))
    print("-----Pytorch version:{}".format(torch.__version__))

    input_tensor = torch.randn(1, 3, 100, 100).to(device)
    print('input_tensor:', input_tensor.shape)
    pretrained_file = "/media/dm/dm2/project/pytorch-learning-tutorials/pretrained/resnet18-5c106cde.pth"
    custom_resnet18 = resnet18(pretrained_file).to(device)
    # print(custom_resnet18)
