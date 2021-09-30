Implementation of MobileNet, modified from https://github.com/pytorch/examples/tree/master/imagenet.
imagenet data is processed [as described here](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset)


nohup python main.py -a mobilenet ImageNet-Folder  > log.txt &

Results
- sgd :                    top1 68.848 top5 88.740 [download](https://pan.baidu.com/s/1nuRcK3Z)
- rmsprop:                top1 0.104  top5 0.494
- rmsprop init from sgd :  top1 69.526 top5 88.978 [donwload](https://pan.baidu.com/s/1eRCxYKU)
- paper:                  top1 70.6

Benchmark:

Titan-X, batchsize = 16
```
  resnet18 : 0.004030
   alexnet : 0.001395
     vgg16 : 0.002310
squeezenet : 0.009848
 mobilenet : 0.073611
```
Titan-X, batchsize = 1
```
  resnet18 : 0.003688
   alexnet : 0.001179
     vgg16 : 0.002055
squeezenet : 0.003385
 mobilenet : 0.076977
```

---------

```
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

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
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x
```
