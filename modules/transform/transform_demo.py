# -*-coding: utf-8 -*-
"""
    @Project: pytorch-learning-tutorials
    @File   : transform_demo.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-08-05 15:37:24
"""
import cv2
import torch
from torchvision import transforms




def custom_transform(transform_type, input_size, RGB_MEAN, RGB_STD):
    if transform_type == "default":
        transform = transforms.Compose([
            # refer to https://pytorch.org/docs/stable/torchvision/transforms.html for more build-in online data augmentation
            transforms.Resize([int(128 * input_size[0] / 112), int(128 * input_size[0] / 112)]),  # smaller side resized
            transforms.RandomCrop([input_size[0], input_size[1]]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=RGB_MEAN, std=RGB_STD),
        ])
    elif transform_type == "scale20_50":
        transform = transforms.Compose([
            transforms.RandomResizedCrop(),
            transforms.Resize([int(128 * input_size[0] / 112), int(128 * input_size[0] / 112)]),  # smaller side resized
            transforms.RandomCrop([input_size[0], input_size[1]]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=RGB_MEAN, std=RGB_STD),
        ])

    elif transform_type == "scale30":
        transform = transforms.Compose([
            transforms.Resize([30, 30]),
            transforms.Resize([int(128 * input_size[0] / 112), int(128 * input_size[0] / 112)]),  # smaller side resized
            transforms.RandomCrop([input_size[0], input_size[1]]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=RGB_MEAN, std=RGB_STD),
        ])
    else:
        raise Exception("transform_type ERROR:{}".format(transform_type))
    return transform




if __name__=="__main__":
    pass