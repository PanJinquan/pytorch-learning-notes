# -*-coding: utf-8 -*-
"""
    @Project: pytorch-learning-tutorials
    @File   : dataset.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-03-07 18:45:06
"""
import os
import PIL.Image as Image
import numpy as np
import cv2
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from utils import image_processing, file_processing


def make_weights_for_balanced_classes(images, nclasses):
    '''
        Make a vector of weights for each image in the dataset, based
        on class frequency. The returned vector of weights can be used
        to create a WeightedRandomSampler for a DataLoader to have
        class balancing when sampling for a training batch.
            images - torchvisionDataset.imgs
            nclasses - len(torchvisionDataset.classes)
        https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
    '''
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1  # item is (img-data, label-id)
    weight_per_class = [0.] * nclasses
    N = float(sum(count))  # total number of images
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]

    return weight


class ImageFolderDataset(Dataset):
    '''
    Pytorch Dataset
    '''

    def __init__(self, image_dir_list, transform=None, repeat=1):
        '''
        :param image_dir_list: [image_dir]->list or `path/to/image_dir`->str
        :param repeat: 所有样本数据重复次数，默认循环一次，当repeat为None时，表示无限循环<sys.maxsize
        '''
        self.classes, self.class_to_idx, self.imgs = self.get_imgs_classes(image_dir_list)
        self.transform = transform
        self.repeat = repeat

    def __getitem__(self, idx):
        image_path = self.imgs[idx][0]
        label_id = self.imgs[idx][1]
        image = self.read_image(image_path)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        return image, label_id

    def __len__(self):
        if self.repeat is None:
            data_len = 10000000
        else:
            data_len = len(self.imgs) * self.repeat
        return data_len

    def get_numclass(self):
        return len(self.classes)

    @staticmethod
    def get_imgs_classes(image_dir_list):
        if isinstance(image_dir_list, str):
            image_dir_list = [image_dir_list]

        image_lists = []
        image_labels = []
        for image_dir in image_dir_list:
            print("loading image from:{}".format(image_dir))
            dir_id = os.path.basename(image_dir)
            image_list, label_list = file_processing.get_files_labels(image_dir, postfix=["*.jpg"])
            label_list = [os.path.join(dir_id, l) for l in label_list]
            print("----have images:{},lable set:{}".format(len(image_list), len(set(label_list))))
            image_lists += image_list
            image_labels += label_list

        classes = list(set(image_labels))
        classes.sort()
        # print( self.label_set)
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        imgs = ImageFolderDataset.get_imgs(image_lists, image_labels, class_to_idx)
        print("Dataset have images:{},classes:{}".format(len(image_lists), len(classes)))
        return classes, class_to_idx, imgs

    @staticmethod
    def get_imgs(image_lists, image_labels, class_to_idx):
        imgs = []
        for image_path, label in zip(image_lists, image_labels):
            label_id = class_to_idx[label]
            imgs.append((image_path, label_id))
        return imgs

    @staticmethod
    def read_image(path, mode='RGB'):
        '''
        读取图片的函数
        :param path:
        :param mode: RGB or L
        :return:
        '''
        try:
            # print("read image:{}".format(path))
            # image = image_processing.read_image(path)
            image = image_processing.read_image(path, colorSpace=mode)
        except Exception as e:
            print(e)
            image = None
        return image

    def data_preproccess(self, image):
        '''
        数据预处理
        :param data:
        :return:
        '''
        image = self.transform(image)
        return image


if __name__ == '__main__':
    image_dir1 = "/media/dm/dm1/project/InsightFace_Pytorch/custom_insightFace/data/facebank"
    image_dir2 = "/media/dm/dm1/project/InsightFace_Pytorch/custom_insightFace/data/faces_emore/imgs"
    # 图像预处理Rescale，RandomCrop，ToTensor
    input_size = [112, 112]
    image_dir_list = [image_dir1, image_dir2]
    train_transform = transforms.Compose([
        transforms.Resize((input_size[0], input_size[1])),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    PIN_MEMORY = True
    NUM_WORKERS = 2
    DROP_LAST = True
    dataset_train = ImageFolderDataset(image_dir_list=image_dir_list, transform=train_transform)
    print("num classs:{}".format(dataset_train.get_numclass()))
    weights = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))
    weights = torch.DoubleTensor(weights)
    # 自定义从数据集中取样本的策略，如果指定这个参数，那么shuffle必须为False
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    dataloader = DataLoader(dataset_train, batch_size=8, sampler=sampler, pin_memory=PIN_MEMORY,
                            num_workers=NUM_WORKERS, drop_last=DROP_LAST, shuffle=False)
    for batch_image, batch_label in iter(dataloader):
        image = batch_image[0, :]
        # image = image.numpy()  #
        image = np.array(image, dtype=np.float32)
        image = image.transpose(1, 2, 0)  # 通道由[c,h,w]->[h,w,c]
        print("batch_image.shape:{},batch_label:{}".format(batch_image.shape, batch_label))
        image_processing.cv_show_image("image", image)
        # batch_x, batch_y = Variable(batch_x), Variable(batch_y)
