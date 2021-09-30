# -*-coding: utf-8 -*-
"""
    @Project: pytorch-learning-tutorials
    @File   : ResNetBatchNorm.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-03-11 13:55:21
"""
import  torch
from    torch import  nn
from    torch.nn import functional as functional

class ResBlk(nn.Module):
    """
    resnet block
    """
    def __init__(self, ch_in, ch_out):
        """

        :param ch_in:
        :param ch_out:
        """
        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.relu= nn.ReLU()

        self.extra = nn.Sequential()
        if ch_out != ch_in:
            # [b, ch_in, h, w] => [b, ch_out, h, w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1),
                nn.BatchNorm2d(ch_out)
            )


    def forward(self, x):
        """

        :param x: [b, ch, h, w]
        :return:
        """
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # short cut.
        # extra module: [b, ch_in, h, w] => [b, ch_out, h, w]
        # element-wise add:
        out = self.extra(x) + out

        return out

class conv2d(nn.Module):

    """
    conv2d
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,activation_fn=None,batch_norm=None):
        '''

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param dilation:
        :param groups:
        :param bias:
        :param activation_fn: 激活函数
        :param batch_norm: 批规范化
        '''
        super(conv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride,padding, dilation, groups, bias)
        # self.bn1 = nn.BatchNorm2d(out_channels)
        # self.relu= nn.ReLU()
        self.activation_fn = activation_fn
        self.batch_norm = batch_norm


    def forward(self, input):
        """

        :param input: [b, ch, h, w]
        :return:
        """
        net = self.conv1(input)
        if self.batch_norm is not None:
            net = self.batch_norm(net)
        if self.activation_fn is not None:
            net = self.activation_fn(net)
        return net



class nets(nn.Module):

    def __init__(self,num_classes):
        super(nets, self).__init__()
        self.num_classes=num_classes
        # 卷积层conv1
        self.conv1 = torch.nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1,padding=0),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        # 卷积层conv2
        self.conv2 = torch.nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1,padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        #
        self.res1 =ResBlk(64, 64)
        self.res2 =ResBlk(64, 128)
        self.res3 =ResBlk(128, 256)

        self.conv3 = torch.nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1,padding=0),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        # 全连接层dense
        self.dense = nn.Sequential(
            nn.Linear(in_features=512*25*25, out_features=self.num_classes)
        )
    def forward(self, input):
        """
        :param x:
        :return:
        """
        net = self.conv1(input)
        net = self.conv2(net)
        net = self.res1(net)
        net = self.res2(net)
        net = self.res3(net)
        net = self.conv3(net)
        # print("x.shape:{}".format(net.shape))
        net = net.view(net.size(0), -1)
        out = self.dense(net)

        return out

def test_net():
    # 检查GPU是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device='cpu'
    print("-----device:{}".format(device))
    print("-----Pytorch version:{}".format(torch.__version__))

    # blk = ResBlk(64, 128).to(device)
    # tmp = torch.randn(2, 64,224, 224)
    # tmp=tmp.to(device)
    # out = blk(tmp)
    # print('blkk', out.shape)


    model = nets(num_classes=10).to(device)
    tmp = torch.randn(2, 3, 224, 224).to(device)
    out = model(tmp).to(device)
    print('resnet:', out.shape)


if __name__ == '__main__':
    test_net()