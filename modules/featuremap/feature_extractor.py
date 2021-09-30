# -*-coding: utf-8 -*-
"""
    @Project: pytorch-learning-tutorials
    @File   : feature_extractor.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-06-28 15:20:53
"""

import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
from utils import image_processing


class FeatureExtractor(nn.Module):
    '''
    中间特征提取
    '''

    def __init__(self, submodule, extracted_layers):
        '''
        :param submodule:
        :param extracted_layers: extracted layer name or layer index list
        '''

        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        '''
        add fc layer adapter
        :param x:
        :return:
        '''
        outputs = []
        for index, (name, module) in enumerate(self.submodule._modules.items()):
            if name is "fc":
                x = x.view(x.size(0), -1)
            x = module(x)
            print(index,name)
            if name in self.extracted_layers:
                outputs.append(x)
            if index in self.extracted_layers:
                outputs.append(x)
        return outputs


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device='cuda'
    print("-----device:{}".format(device))
    print("-----Pytorch version:{}".format(torch.__version__))

    # 特征输出
    model_path = "/media/dm/dm2/project/pytorch-learning-tutorials/pretrained/resnet18-5c106cde.pth"
    myresnet = models.resnet18(pretrained=False)
    myresnet.load_state_dict(torch.load(model_path))
    print("myresnet:",myresnet)
    # exact_list = ["conv1", "layer1", "avgpool"]
    exact_list = [0, 4, 8]

    exactor = FeatureExtractor(myresnet, exact_list)

    ##
    input_tensor = torch.randn(size=(1, 3, 640, 640))
    feature_maps = exactor(input_tensor)

    feature_map = feature_maps[0]
    # image = image.numpy()  #
    feature_map = feature_map[0, :].detach().numpy()
    size = feature_map.shape[0]
    # for i in range(size):
    #     feature = feature_map[i, :]
    #     # feature = feature.transpose(1, 2, 0)  # 通道由[c,h,w]->[h,w,c]
    #     image_processing.cv_show_image("image", feature)
    #     print("feature_map.shape:{}}".format(feature_map.shape))
