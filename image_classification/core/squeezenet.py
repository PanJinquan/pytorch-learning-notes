# -*-coding: utf-8 -*-
"""
    @Project: pytorch-learning-tutorials
    @File   : squeezenet.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-03-19 14:34:14
"""
import torch
from  torch import  nn
from  torch.nn import functional as F
import torch.nn.init as init

class Fire(nn.Module):
    def __init__(self,inchn,sqzout_chn,exp1x1out_chn,exp3x3out_chn):
        super(Fire,self).__init__()
        self.inchn = inchn
        self.squeeze = nn.Conv2d(inchn,sqzout_chn,kernel_size=1)
        self.squeeze_act = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(sqzout_chn,exp1x1out_chn,kernel_size=1)
        self.expand1x1_act = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(sqzout_chn,exp3x3out_chn,kernel_size=3, padding=1)
        self.expand3x3_act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.squeeze_act(self.squeeze(x))
        return torch.cat([
                self.expand1x1_act(self.expand1x1(x)),
                self.expand3x3_act(self.expand3x3(x))
                ], 1)

class nets(nn.Module):
    def __init__(self, num_classes=7):
        super(nets, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,stride=2),
            nn.ReLU(inplace=True),
            # 这里ceil_mode一定要设成False，不然finetune会报错，
            #  后面你会看到我finetune时也改了这里，
            #  因为目前onnx不支持squeezenet的 ceil_mode=True！！
            nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=False),
            Fire(64,16,64,64),
            Fire(128,16,64,64),
            nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=False),
            Fire(128,32,128,128),
            Fire(256,32,128,128),
            nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=False),
            Fire(256,48,192,192),
            Fire(384,48,192,192),
            Fire(384,64,256,256),
            Fire(512,64,256,256),
        )
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            # nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AvgPool2d(13)
        )
        # 这里参考了官网的实现，就是做参数初始化
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         if m is final_conv:
        #             init.normal(m.weight.data, mean=0.0, std=0.01)
        #         else:
        #             init.kaiming_uniform(m.weight.data)
        #         if m.bias is not None:
        #             m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)