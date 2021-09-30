# -*-coding: utf-8 -*-
"""
    @Project: pytorch-learning-tutorials
    @File   : main.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-06-27 13:46:20
"""
import torch
from torchvision import models
from utils import debug
import performance.core.mixnet as mixnet
from performance.core import torch_resnet

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'
print("-----device:{}".format(device))
print("-----Pytorch version:{}".format(torch.__version__))


# @debug.run_time_decorator()
def model_forward(model, input_tensor):
    with torch.no_grad():
        T0 = debug.TIME()
        out = model(input_tensor)
        torch.cuda.synchronize()
        T1 = debug.TIME()
        time = debug.RUN_TIME(T1 - T0)
    return out, time


def iter_model(model, input_tensor, iter):
    out, time = model_forward(model, input_tensor)
    all_time = 0
    for i in range(iter):
        out, time = model_forward(model, input_tensor)
        all_time += time
    return all_time


def squeezenet1_0(input_tensor, out_features, iter=10):
    model = models.squeezenet.squeezenet1_0(pretrained=False, num_classes=out_features).to(device)
    model.eval()
    all_time = iter_model(model, input_tensor, iter)
    print("squeezenet1_0,mean run time :{:.3f}".format(all_time / iter))


def squeezenet1_1(input_tensor, out_features, iter=10):
    model = models.squeezenet.squeezenet1_1(pretrained=False, num_classes=out_features).to(device)
    model.eval()
    all_time = iter_model(model, input_tensor, iter)
    print("squeezenet1_1,mean run time :{:.3f}".format(all_time / iter))


def mnasnet1_0(input_tensor, out_features, iter=10):
    model = models.mnasnet.mnasnet1_0(pretrained=False, num_classes=out_features).to(device)
    model.eval()
    all_time = iter_model(model, input_tensor, iter)
    print("mnasnet1_0,mean run time :{:.3f}".format(all_time / iter))


def shufflenet_v2_x1_0(input_tensor, out_features, iter=10):
    model = models.shufflenetv2.shufflenet_v2_x1_0(pretrained=False, num_classes=out_features).to(device)
    model.eval()
    all_time = iter_model(model, input_tensor, iter)
    print("shufflenet_v2_x1_0,mean run time :{:.3f}".format(all_time / iter))


def mobilenet_v2(input_tensor, out_features, iter=10):
    model = models.mobilenet_v2(pretrained=False, num_classes=out_features).to(device)
    model.eval()
    all_time = iter_model(model, input_tensor, iter)
    print("mobilenet_v2,mean run time :{:.3f}".format(all_time / iter))


def resnet18(input_tensor, out_features, iter=10):
    model = models.resnet18(pretrained=False, num_classes=out_features).to(device)
    model.eval()
    all_time = iter_model(model, input_tensor, iter)
    print("reset18,mean run time :{:.3f}".format(all_time / iter))


def torch_resnet18(input_tensor, out_features, iter=10):
    model = torch_resnet.resnet18(pretrained=False, num_classes=out_features).to(device)
    model.eval()
    # Specify quantization configuration
    # Start with simple min/max range estimation and per-tensor quantization of weights
    model.qconfig = torch.quantization.default_qconfig
    print(model.qconfig)
    torch.quantization.prepare(model, inplace=True)

    all_time = iter_model(model, input_tensor, iter)
    print("reset18,mean run time :{:.3f}".format(all_time / iter))



def resnet34(input_tensor, out_features, iter=10):
    model = models.resnet34(pretrained=False, num_classes=out_features).to(device)
    model.eval()
    all_time = iter_model(model, input_tensor, iter)
    print("resnet34,mean run time :{:.3f}".format(all_time / iter))



def resnet50(input_tensor, out_features, iter=10):
    model = models.resnet50(pretrained=False, num_classes=out_features).to(device)
    model.eval()
    all_time = iter_model(model, input_tensor, iter)
    print("resnet50,mean run time :{:.3f}".format(all_time / iter))


def vgg16(input_tensor, out_features, iter=10):
    model = models.vgg16(pretrained=False, num_classes=out_features).to(device)
    model.eval()
    all_time = iter_model(model, input_tensor, iter)
    print("vgg16,mean run time :{:.3f}".format(all_time / iter))


def MixNet_L(input_tensor, input_size, out_features, iter=10):
    model = mixnet.MixNet_L(input_size, out_features).to(device)
    model.eval()
    all_time = iter_model(model, input_tensor, iter)
    print("MixNet_L,mean run time :{:.3f}".format(all_time / iter))


def MixNet_M(input_tensor, input_size, out_features, iter=10):
    model = mixnet.MixNet_M(input_size, out_features).to(device)
    model.eval()
    all_time = iter_model(model, input_tensor, iter)
    print("MixNet_M,mean run time :{:.3f}".format(all_time / iter))


def MixNet_S(input_tensor, input_size, out_features, iter=10):
    model = mixnet.MixNet_S(input_size, out_features).to(device)
    model.eval()
    all_time = iter_model(model, input_tensor, iter)
    print("MixNet_S,mean run time :{:.3f}".format(all_time / iter))


def inception_v3(input_tensor, out_features, iter=10):
    model = models.inception.inception_v3(pretrained=False, num_classes=out_features).to(device)
    model.eval()
    all_time = iter_model(model, input_tensor, iter)
    print("inception_v3,mean run time :{:.3f}".format(all_time / iter))

def googlenet(input_tensor, out_features, iter=10):
    model = models.googlenet(pretrained=False, num_classes=out_features).to(device)
    model.eval()
    all_time = iter_model(model, input_tensor, iter)
    print("googlenet,mean run time :{:.3f}".format(all_time / iter))

if __name__ == "__main__":
    input_size = [112, 112]
    out_features = 4
    input_tensor = torch.randn(1, 3, input_size[0], input_size[1]).to(device)
    print('input_tensor:', input_tensor.shape)
    iter = 10
    mobilenet_v2(input_tensor, out_features, iter)
    # resnet18(input_tensor, out_features, iter)
    resnet18(input_tensor, out_features, iter)
    resnet34(input_tensor, out_features, iter)
    resnet50(input_tensor, out_features, iter)
    # vgg16(input_tensor, out_features, iter)
    # squeezenet1_0(input_tensor, out_features, iter)
    # squeezenet1_1(input_tensor, out_features, iter)
    # inception_v3(input_tensor, out_features, iter)
    # googlenet(input_tensor, out_features, iter)

    # mnasnet1_0(input_tensor, out_features, iter)
    # shufflenet_v2_x1_0(input_tensor, out_features, iter)
    # MixNet_S(input_tensor, input_size, out_features, iter)
    # MixNet_M(input_tensor, input_size, out_features, iter)
    # MixNet_L(input_tensor, input_size, out_features, iter)

