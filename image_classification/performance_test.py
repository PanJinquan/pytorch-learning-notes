# -*-coding: utf-8 -*-
"""
    @Project: pytorch-learning-tutorials
    @File   : performance_test.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-04-09 14:58:48
"""
import time
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from core import mobileNet_V2,mobileNet_V1

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device='cpu'
# device='cuda'

def speed(model, name):
    t0 = time.time()
    input = torch.rand(1, 3, 224, 224).to(device)
    input = Variable(input, volatile=True)
    t1 = time.time()

    model(input)
    t2 = time.time()

    model(input)
    t3 = time.time()
    print('%10s : %fs' % (name, t3 - t2))


if __name__ == '__main__':
    '''
    各种模型性能测试
    '''
    # cudnn.benchmark = True # This will make network slow ??
    num_classes=1000
    resnet18 = models.resnet18().to(device)
    alexnet = models.alexnet().to(device)
    vgg16 = models.vgg16().to(device)
    squeezenet = models.squeezenet1_0().to(device)
    mobilenet_v1 = mobileNet_V1.MobileNetV1(num_classes=num_classes).to(device)
    mobilenet_v2 = mobileNet_V2.MobileNetV2(num_classes=num_classes).to(device)

    speed(resnet18, 'resnet18')
    speed(alexnet, 'alexnet')
    speed(vgg16, 'vgg16')
    speed(squeezenet, 'squeezenet')
    speed(mobilenet_v1, 'mobilenet_v1')
    speed(mobilenet_v2, 'mobilenet_v2')