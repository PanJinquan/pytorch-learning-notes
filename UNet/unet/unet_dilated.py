# -*-coding: utf-8 -*-
"""
    @Project: UNet
    @File   : unet_dilated.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-05-07 11:05:00
"""
import torch.nn as nn
import torch
import torch.nn.functional as F

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class double_dilation(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch,dilation=1):
        super(double_dilation, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1,stride=1,dilation=1), #不会改变尺寸
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=2,stride=2,dilation=dilation),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class down_layer(nn.Module):
    '''
    MaxPool2d+double_conv(2个卷积)
    '''
    def __init__(self, in_ch, out_ch, downType="pool"):
        super(down_layer, self).__init__()
        if downType== "pool":
            self.mpconv = nn.Sequential(
                nn.MaxPool2d(2),
                double_conv(in_ch, out_ch)
            )
        elif downType== "dilation":
            self.mpconv = nn.Sequential(
                double_dilation(in_ch, out_ch,dilation=2)
            )
    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()
        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # print("x1,x2.shape:{},{}".format(x1.shape,x2.shape))
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.input_layer = double_conv(n_channels, 64)
        self.down1 = down_layer(64, 128 ,downType="pool")
        self.down2 = down_layer(128, 256,downType="pool")
        self.down3 = down_layer(256, 512,downType="pool")
        self.down4 = down_layer(512, 512,downType="pool")
        # self.down1 = down_layer(64, 128 ,downType="dilation")
        # self.down2 = down_layer(128, 256,downType="dilation")
        # self.down3 = down_layer(256, 512,downType="dilation")
        # self.down4 = down_layer(512, 512,downType="dilation")
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.output_layer = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x1 = self.input_layer(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.output_layer(x)
        return torch.sigmoid(out)

def net_test():
    net = UNet(n_channels=3, n_classes=1)
    tmp = torch.randn(2, 3,50, 50)
    out = net(tmp)
    print('Unet', out.shape)

if __name__ == '__main__':
    net_test()