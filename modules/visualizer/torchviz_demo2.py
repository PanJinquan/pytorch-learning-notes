# -*-coding: utf-8 -*-
"""
    @Project: pytorch-learning-tutorials
    @File   : vision_test01.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-06-16 10:45:23
    @url    :《深度学习tensorwatch可视化工具》https://blog.csdn.net/qq_29592829/article/details/90517303
    conda install graphviz
    conda install torchvision
    conda install tensorwatch
"""
import sys
import torch
import tensorwatch as tw
import torchvision.models

# 添加graphviz环境路径
import os

os.environ["PATH"] += os.pathsep + 'D:/ProgramData/Anaconda3/envs/pytorch-py36/Library/bin/graphviz/'  # windows


def tensorwatch_resnet18():
    resnet_model = torchvision.models.resnet18(True)
    draw_image = tw.draw_model(resnet_model, [1, 3, 224, 224])
    tw.model_stats(resnet_model, [1, 3, 224, 224])

def tensorwatch_model():
    model_path="/media/dm/dm2/project/pytorch-learning-tutorials/image_classification/models/pretrain/resnet18-class5.pth"
    model = torch.load(model_path)
    # state_dict = checkpoint.get("state_dict")
    # model.load_state_dict(checkpoint)
    draw_image = tw.draw_model(model, [1, 3, 224, 224])
    tw.model_stats(model, [1, 3, 224, 224])

if __name__ == "__main__":
    torch.hub.list('rwightman/gen-efficientnet-pytorch')
    # tensorwatch_resnet18()