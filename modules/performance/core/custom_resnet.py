import torch
from torch import nn
from torch.nn import functional as F


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
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # short cut.
        # extra module: [b, ch_in, h, w] => [b, ch_out, h, w]
        # element-wise add:
        out = self.extra(x) + out

        return out


class resnet18(nn.Module):

    def __init__(self, num_classes):
        super(resnet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(32)
        )
        self.num_classes = num_classes
        # followed 4 blocks
        # self.blk1 = ResBlk(16, 32)
        # self.blk2 = ResBlk(32, 64)
        # self.blk3 = ResBlk(64, 128)
        # self.blk4 = ResBlk(128, 256)
        self.blk1 = ResBlk(32, 256)
        self.blk2 = ResBlk(256, 512)
        self.blk3 = ResBlk(512, 512)
        self.blk4 = ResBlk(512, 1024)
        # self.outlayer = nn.Linear(256*74*74,  self.num_classes)
        # self.outlayer = nn.Linear(256 * 33 * 33, self.num_classes)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = F.relu(self.conv1(x))

        # [b, 64, h, w] => [b, 1024, h, w]
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)

        print(x.shape)
        # x = x.view(x.size(0), -1)
        # x = self.outlayer(x)

        return x


def test_net():
    blk = ResBlk(64, 128)
    tmp = torch.randn(2, 64, 224, 224)
    out = blk(tmp)
    print('blkk', out.shape)

    model = resnet18(num_classes=5)
    tmp = torch.randn(2, 3, 100, 100)
    out = model(tmp)
    print('resnet:', out.shape)


if __name__ == '__main__':
    test_net()
