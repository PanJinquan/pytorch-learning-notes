# -*-coding: utf-8 -*-
"""
    @Project: pytorch-learning-tutorials
    @File   : efficientnet.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-07-06 12:49:38
    @url    : https://github.com/rwightman/gen-efficientnet-pytorch
"""
import torch
if __name__=="__main__":
    torch.hub.list('rwightman/gen-efficientnet-pytorch')
    model = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnet_b0', pretrained=True)
    model.eval()
    output = model(torch.randn(1,3,224,224))
    print(output)