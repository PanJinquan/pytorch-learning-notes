# -*-coding: utf-8 -*-
"""
    @Project: pytorch-learning-tutorials
    @File   : simpleNet.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-03-09 13:59:49
"""
import torch
from  torch import  nn
from  torch.nn import functional as F

class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # 卷积层conv1
        self.conv1 = torch.nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2))
        # 卷积层conv2
        self.conv2 = torch.nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2))

        # 卷积层conv3
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2))
        # 全连接层dense
        self.dense = nn.Sequential(
            nn.Linear(64 * 3 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        res = conv3_out.view(conv3_out.size(0), -1)
        out = self.dense(res)
        return out

def test_net():
    model = SimpleNet()
    tmp = torch.randn(2, 3, 28, 28)
    out = model(tmp)
    print('resnet:', out.shape)


if __name__ == '__main__':
    test_net()