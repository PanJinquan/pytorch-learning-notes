# -*-coding: utf-8 -*-
"""
    @Project: pytorch-learning-tutorials
    @File   : train_mobileNet_V1.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-04-09 08:59:27
    @url    : https://github.com/marvis/pytorch-mobilenet
"""
import time
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MobileNetV1(nn.Module):
    def __init__(self,num_classes):
        super(MobileNetV1, self).__init__()
        self.num_classes=num_classes

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(  3,  32, 2), 
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, self.num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

def speed(model, name):
    t0 = time.time()
    input = torch.rand(1,3,224,224).to(device)
    input = Variable(input, volatile = True)
    t1 = time.time()

    model(input)
    t2 = time.time()

    model(input)
    t3 = time.time()
    
    print('%10s : %f' % (name, t3 - t2))

if __name__ == '__main__':
    #cudnn.benchmark = True # This will make network slow ??
    resnet18 = models.resnet18().to(device)
    alexnet = models.alexnet().to(device)
    vgg16 = models.vgg16().to(device)
    squeezenet = models.squeezenet1_0().to(device)
    mobilenet = MobileNetV1().to(device)
    print(mobilenet)

    speed(resnet18, 'resnet18')
    speed(alexnet, 'alexnet')
    speed(vgg16, 'vgg16')
    speed(squeezenet, 'squeezenet')
    speed(mobilenet, 'mobilenet')
