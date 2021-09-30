# -*-coding: utf-8 -*-
"""
    @Project: pytorch-learning-tutorials
    @File   : pytorch_imagefolder.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-08-05 10:04:15
"""

import os
import torch
import random
import PIL.Image as Image
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from core import imagefolder_dataset
from utils import image_processing


class RandomResize(object):
    def __init__(self, range_size, interpolation=Image.BILINEAR):
        self.interpolation = interpolation
        self.range_size = range_size

    def __call__(self, img):
        r = int(random.uniform(self.range_size[0], self.range_size[1]))
        size = (r, r)
        print("RandomResize:{}".format(size))
        return transforms.functional.resize(img, size, self.interpolation)

    def __repr__(self):
        interpolate_str = transforms._pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


def custom_transform(input_size, RGB_MEAN, RGB_STD, transform_type):
    '''
    :param input_size:
    :param RGB_MEAN:
    :param RGB_STD:
    :param transform_type: [default,scale20_50,scale30]
    :return:
    '''
    # train_transform = transforms.Compose([
    #     # refer to https://pytorch.org/docs/stable/torchvision/transforms.html for more build-in online data augmentation
    #     transforms.Resize([int(128 * INPUT_SIZE[0] / 112), int(128 * INPUT_SIZE[0] / 112)]),  # smaller side resized
    #     transforms.RandomCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=RGB_MEAN,
    #                          std=RGB_STD),
    # ])
    if transform_type == "default":
        transform = transforms.Compose([
            transforms.Resize([int(128 * input_size[0] / 112), int(128 * input_size[0] / 112)]),  # smaller side resized
            transforms.RandomCrop([input_size[0], input_size[1]]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=RGB_MEAN, std=RGB_STD),
        ])

    elif transform_type == "scale20_50":
        transform = transforms.Compose([
            RandomResize(range_size=(20, 50)),
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


if __name__ == '__main__':
    image_dir1 = "/media/dm/dm1/project/InsightFace_Pytorch/custom_insightFace/data/facebank"
    image_dir2 = "/media/dm/dm1/project/InsightFace_Pytorch/custom_insightFace/data/faces_emore/imgs"
    # 图像预处理Rescale，RandomCrop，ToTensor
    input_size = [112, 112]
    image_dir_list = [image_dir1]
    train_transform = custom_transform(input_size, RGB_MEAN=[0.5, 0.5, 0.5], RGB_STD=[0.5, 0.5, 0.5],
                                       transform_type="scale20_50")
    PIN_MEMORY = True
    NUM_WORKERS = 2
    DROP_LAST = True
    dataset_train = datasets.ImageFolder(image_dir1, transform=train_transform)
    # dataset_train = imagefolder_dataset.ImageFolderDataset(image_dir_list=image_dir_list, transform=train_transform)

    print("num images:{},num classs:{}".format(len(dataset_train.imgs), len(dataset_train.classes)))
    weights = imagefolder_dataset.make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))
    weights = torch.DoubleTensor(weights)
    # 自定义从数据集中取样本的策略，如果指定这个参数，那么shuffle必须为False
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    # dataloader = DataLoader(dataset_train, batch_size=8, sampler=sampler, pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS, drop_last=DROP_LAST, shuffle=False)
    dataloader = DataLoader(dataset_train, batch_size=1, pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS,
                            drop_last=DROP_LAST, shuffle=False)

    for batch_image, batch_label in iter(dataloader):
        image = batch_image[0, :]
        # image = image.numpy()  #
        image = np.array(image, dtype=np.float32)
        image = image.transpose(1, 2, 0)  # 通道由[c,h,w]->[h,w,c]
        print("batch_image.shape:{},batch_label:{}".format(batch_image.shape, batch_label))
        image_processing.cv_show_image("image", image)
        # batch_x, batch_y = Variable(batch_x), Variable(batch_y)
