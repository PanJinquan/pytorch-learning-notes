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


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)

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
                 padding=0, dilation=1,activation_fn=None,batch_norm=None):
        '''

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param dilation:
        :param groups:
        :param bias:
        :param activation_fn:
        :param batch_norm:
        '''
        super(conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        # self.bn1 = nn.BatchNorm2d(out_channels)
        # self.relu= nn.ReLU()
        if batch_norm is not None:
            self.batch_norm = batch_norm(out_channels)
        else:
            self.batch_norm =None

        if activation_fn is not None:
            self.activation_fn = activation_fn()
        else:
            self.activation_fn =None

    def forward(self, input):
        """

        :param input: [b, ch, h, w]
        :return:
        """
        net = self.conv(input)
        if self.batch_norm is not None:
            net = self.batch_norm(net)
        if self.activation_fn is not None:
            net = self.activation_fn(net)
        return net

class Dense(nn.Module):

    """
    Dense
    """
    def __init__(self, in_features, out_features,activation_fn=None,batch_norm=None,dropout_prob=None):
        '''

        :param in_features:
        :param out_features:
        :param activation_fn:
        :param batch_norm:
        :param dropout_prob:
        '''
        super(Dense, self).__init__()

        self.dense = nn.Linear(in_features=in_features, out_features=out_features)
        # self.bn1 = nn.BatchNorm1d(out_channels)
        # self.relu= nn.ReLU()
        if batch_norm is not None:
            self.batch_norm = batch_norm(out_features)
        else:
            self.batch_norm =None

        if activation_fn is not None:
            self.activation_fn = activation_fn()
        else:
            self.activation_fn =None

        if dropout_prob is not None:
            self.drop_out = nn.Dropout(p=dropout_prob)
        else:
            self.drop_out=None


    def forward(self, input):
        """
        :param input: [b, ch, h, w]
        :return:
        """
        net = self.dense(input)
        if self.batch_norm is not None:
            net = self.batch_norm(net)
        if self.activation_fn is not None:
            net = self.activation_fn(net)
        if self.drop_out is not None:
            net = self.drop_out(net)
        return net


class nets(nn.Module):

    def __init__(self,num_classes):
        super(nets, self).__init__()
        self.num_classes=num_classes
        activation_fn=nn.ReLU
        batch_norm=nn.BatchNorm2d

        # 卷积层conv1
        self.conv_1=conv2d(in_channels=3,
                           out_channels=32,
                           kernel_size=3,
                           stride=1,padding=0,
                           activation_fn=activation_fn,
                           batch_norm=batch_norm)
        self.pool_1=nn.MaxPool2d(kernel_size=3, stride=2)

        # 卷积层conv2
        self.conv_2=conv2d(in_channels=32,
                           out_channels=64,
                           kernel_size=3,
                           stride=1,padding=0,
                           activation_fn=activation_fn,
                           batch_norm=batch_norm)
        self.pool_2=nn.MaxPool2d(kernel_size=3, stride=2)
        #
        self.res1 =ResBlk(64, 64)
        self.res2 =ResBlk(64, 128)
        self.res3 =ResBlk(128, 256)

        self.conv_3=conv2d(in_channels=256,
                           out_channels=512,
                           kernel_size=3,
                           stride=1,padding=0,
                           activation_fn=activation_fn,
                           batch_norm=batch_norm)
        self.pool_3=nn.MaxPool2d(kernel_size=3, stride=2)

        self.flatten = Flatten()
        # 全连接层dense
        # self.dense = nn.Sequential(
        #     nn.Linear(in_features=512*25*25, out_features=self.num_classes)
        # )
        # self.fc_1 = Dense(in_features=512*25*25,
        #                    out_features=512,
        #                    activation_fn=activation_fn,
        #                    batch_norm=nn.BatchNorm1d,
        #                    dropout_prob=0.5)
        #
        # self.dense = Dense(in_features=512,
        #                    out_features=num_classes,
        #                    activation_fn=None,
        #                    batch_norm=None,
        #                    dropout_prob=None)
        self.dense = Dense(in_features=512*25*25,
                           out_features=num_classes,
                           activation_fn=None,
                           batch_norm=None,
                           dropout_prob=None)
    def forward(self, input):
        """
        :param x:
        :return:
        """
        net = self.conv_1(input)
        net = self.pool_1(net)

        net = self.conv_2(net)
        net = self.pool_2(net)


        net = self.res1(net)
        net = self.res2(net)
        net = self.res3(net)

        net = self.conv_3(net)
        net = self.pool_3(net)

        # print("x.shape:{}".format(net.shape))
        net=self.flatten(net)#net = net.view(net.size(0), -1)
        # net = self.fc_1(net)
        out = self.dense(net)
        return out

def test_net():
    # 检查GPU是否可用
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device='cpu'
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