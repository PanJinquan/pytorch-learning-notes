# -*-coding: utf-8 -*-
"""
    @Project: pytorch-learning-tutorials
    @File   : resnet.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-06-27 14:38:54
"""
import torch
from torchvision import models

from transfer_model.core.torch_resnet import BasicBlock, Bottleneck, ResNet


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model



def transfer_model(pretrained_file, model):
    '''
    只导入pretrained_file部分模型参数
    tensor([-0.7119,  0.0688, -1.7247, -1.7182, -1.2161, -0.7323, -2.1065, -0.5433,-1.5893, -0.5562]
    update:
        D.update([E, ]**F) -> None.  Update D from dict/iterable E and F.
        If E is present and has a .keys() method, then does:  for k in E: D[k] = E[k]
        If E is present and lacks a .keys() method, then does:  for k, v in E: D[k] = v
        In either case, this is followed by: for k in F:  D[k] = F[k]
    :param pretrained_file:
    :param model:
    :return:
    '''
    pretrained_dict = torch.load(pretrained_file)  # get pretrained dict
    model_dict = model.state_dict()  # get model dict
    # 在合并前(update),需要去除pretrained_dict一些不需要的参数
    pretrained_dict = transfer_state_dict(pretrained_dict, model_dict)
    model_dict.update(pretrained_dict)  # 更新(合并)模型的参数
    model.load_state_dict(model_dict)
    return model


def transfer_state_dict(pretrained_dict, model_dict):
    '''
    根据model_dict,去除pretrained_dict一些不需要的参数,以便迁移到新的网络
    url: https://blog.csdn.net/qq_34914551/article/details/87871134
    :param pretrained_dict:
    :param model_dict:
    :return:
    '''
    # state_dict2 = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    state_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys():
            # state_dict.setdefault(k, v)
            state_dict[k] = v
        else:
            print("Missing key(s) in state_dict :{}".format(k))
    return state_dict


def string_rename(old_string, new_string, start, end):
    new_string = old_string[:start] + new_string + old_string[end:]
    return new_string


def modify_model(pretrained_file, model, old_prefix, new_prefix):
    '''
    :param pretrained_file:
    :param model:
    :param old_prefix:
    :param new_prefix:
    :return:
    '''
    pretrained_dict = torch.load(pretrained_file)
    model_dict = model.state_dict()
    state_dict = modify_state_dict(pretrained_dict, model_dict, old_prefix, new_prefix)
    model.load_state_dict(state_dict)
    return model


def modify_state_dict(pretrained_dict, model_dict, old_prefix, new_prefix):
    '''
    修改model dict
    :param pretrained_dict:
    :param model_dict:
    :param old_prefix:
    :param new_prefix:
    :return:
    '''
    state_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys():
            # state_dict.setdefault(k, v)
            state_dict[k] = v
        else:
            for o, n in zip(old_prefix, new_prefix):
                prefix = k[:len(o)]
                if prefix == o:
                    kk = string_rename(old_string=k, new_string=n, start=0, end=len(o))
                    print("rename layer modules:{}-->{}".format(k, kk))
                    state_dict[kk] = v
    return state_dict


if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    print("-----device:{}".format(device))
    print("-----Pytorch version:{}".format(torch.__version__))

    input_tensor = torch.zeros(1, 3, 100, 100)
    print('input_tensor:', input_tensor.shape)
    pretrained_file = "/media/dm/dm2/project/pytorch-learning-tutorials/pretrained/resnet18-5c106cde.pth"
    # model = models.resnet18()
    # model.load_state_dict(torch.load(pretrained_file))
    # model.eval()
    # out = model(input_tensor)
    # print("out:", out.shape, out[0, 0:10])
    #
    # model1 = resnet18()
    # model1 = transfer_model(pretrained_file, model1)
    # out1 = model1(input_tensor)
    # print("out1:", out1.shape, out1[0, 0:10])
    #
    new_file = "new_model.pth"
    model = resnet18()
    new_model = modify_model(pretrained_file, model, old_prefix=["layer4"], new_prefix=["layer44"])
    torch.save(new_model.state_dict(), new_file)

    model2 = resnet18()
    model2.load_state_dict(torch.load(new_file))
    model2.eval()
    out2 = model2(input_tensor)
    print("out2:", out2.shape, out2[0, 0:10])
