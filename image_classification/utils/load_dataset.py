# -*-coding: utf-8 -*-
"""
    @Project: pytorch-learning-tutorials
    @File   : load_dataset.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-03-07 15:55:21
"""
from torch.utils.data import DataLoader, Dataset
from skimage import io, transform
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms
import numpy as np
from utils import image_processing


class PytorchDataset(Dataset):
    def __init__(self, filename,image_dir,resize_height, resize_width, transform=None):
        self.transform = transform
        self.resize_height = resize_height
        self.resize_width = resize_width

        self.image_label_list = self.read_file(filename)
        self.image_dir = image_dir
        self.len = len(self.image_label_list)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        image_name, label = self.image_label_list[idx]
        image_path = os.path.join(self.image_dir, image_name)
        img = self.load_data(image_path, self.resize_height, self.resize_width, normalization=False)
        label = np.array(label)
        sample = {'image': img, 'lable': label}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def read_file(self, filename):
        image_label_list = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # rstrip：用来去除结尾字符、空白符(包括\n、\r、\t、' '，即：换行、回车、制表符、空格)
                content = line.rstrip().split(' ')
                name = content[0]
                labels = []
                for value in content[1:]:
                    labels.append(int(value))
                image_label_list.append((name, labels))
        return image_label_list
    def load_data(self, path, resize_height, resize_width, normalization):
        '''
        加载数据
        :param path:
        :param resize_height:
        :param resize_width:
        :param normalization: 是否归一化
        :return:
        '''
        image = image_processing.read_image(path, resize_height, resize_width, normalization)
        return image


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, lable = sample['image'], sample['lable']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w),mode='constant')

        # h and w are swapped for lable because for images,
        # x and y axes are axis 1 and 0 respectively
        # lable = lable * [new_w / w, new_h / h]

        return {'image': img, 'lable': lable}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, lable = sample['image'], sample['lable']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,left: left + new_w]

        # lable = lable - [left, top]
        return {'image': image, 'lable': lable}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, lable = sample['image'], sample['lable']
        # print lable
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'lable': torch.from_numpy(lable)}


if __name__ == '__main__':
    train_filename="../dataset/test_images/train.txt"
    # test_filename="../dataset/test_images/test.txt"
    image_dir='../dataset/test_images/images'

    # 图像预处理Rescale，RandomCrop，ToTensor
    transform = transforms.Compose([Rescale(256),RandomCrop(224),ToTensor()])
    data = PytorchDataset(train_filename, image_dir, resize_height=None, resize_width=None, transform=transform)
    dataloader = DataLoader(data, batch_size=4, shuffle=False, num_workers=4)
    for i, bach_data in enumerate(dataloader):
        batch_image=bach_data['image']
        batch_label=bach_data['lable']

        image = batch_image[0, :]
        # image = image.numpy()  #
        image = np.array(image,dtype=np.float32)
        image = image.transpose(1, 2, 0)  # 通道由[c,h,w]->[h,w,c]
        image_processing.cv_show_image("image", image)
        print("batch_image.shape:{},batch_label:{}".format(batch_image.shape, batch_label))
        # batch_x, batch_y = Variable(batch_x), Variable(batch_y)

