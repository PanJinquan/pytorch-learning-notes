# -*-coding: utf-8 -*-
"""
    @Project: pytorch-learning-tutorials
    @File   : dataset.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-03-07 18:45:06
"""
import os
import cv2
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from utils import image_processing, dataset_collate, custom_collate_fn


def getDataTransform(resize_height, resize_width):
    '''
    图像预处理
    transforms.RandomHorizontalFlip(),#随机翻转图像
    transforms.RandomCrop(size=(resize_height, resize_width), padding=4),  # 随机裁剪
    transforms.ToTensor(),  # 吧shape=(H,W,C)->换成shape=(C,H,W),并且归一化到[0.0, 1.0]的torch.FloatTensor类型
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))#给定均值(R,G,B) 方差（R，G，B），将会把Tensor正则化
    :param resize_height:
    :param resize_width:
    :return:
    '''
    data_transform = transforms.Compose([
        transforms.Resize(size=(resize_height, resize_width)),
    ])
    return data_transform


def getDataLoader(dataset, batch_size, shuffle=False, num_workers=0, collate_fn=dataset_collate.collate_fn):
    '''
    DataLoader,产生迭代数据
    :param dataset:
    :param batch_size:
    :param shuffle:
    :param num_workers
    :param collate_fn:
    :return:
    '''
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             collate_fn=collate_fn,
                             num_workers=num_workers)
    return data_loader


class TorchDataset(Dataset):
    '''
    Pytorch Dataset
    '''

    def __init__(self, image_id_list, image_dir=None, resize_height=None, resize_width=None, repeat=1):
        '''
        :param image_id_list: 图片id
        :param image_dir: 图片路径：image_dir+imge_name.jpg构成图片的完整路径
        :param resize_height 为None时，不进行缩放
        :param resize_width  为None时，不进行缩放，
                              PS：当参数resize_height或resize_width其中一个为None时，可实现等比例缩放
        :param repeat: 所有样本数据重复次数，默认循环一次，当repeat为None时，表示无限循环<sys.maxsize
        '''
        self.image_dir = image_dir
        self.image_id_list = image_id_list
        self.len = len(image_id_list)
        self.repeat = repeat
        self.resize_height = resize_height
        self.resize_width = resize_width

    def __getitem__(self, i):
        index = i % self.len
        # print("i={},index={}".format(i, index))
        image_id = self.image_id_list[index]
        if self.image_dir is None:
            image_path = image_id
        else:
            image_path = os.path.join(self.image_dir, image_id)
        image = self.read_image(image_path)
        image = self.data_preproccess(image)
        data = {"image": image, "id": image_id}
        return data

    def __len__(self):
        if self.repeat is None:
            data_len = 10000000
        else:
            data_len = len(self.image_id_list) * self.repeat
        return data_len

    @staticmethod
    def read_image(path, mode='RGB'):
        '''
        读取图片的函数
        :param path:
        :param mode: RGB or L
        :return:
        '''
        try:
            print("read image:{}".format(path))
            # image = image_processing.read_image(path)
            image = image_processing.read_images_url(path, colorSpace=mode)
            cv2.waitKey(1000)
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
        if image is not None:
            image = image_processing.resize_image(image, self.resize_height, self.resize_width)
        return image


if __name__ == '__main__':
    height = None
    width = None
    test_image_id_list = ["12.jpg", "22.jpg", "3.jpg", "4.jpg", "5.jpg", "6.jpg", "7.jpg", "8.jpg", "9.jpg",
                          "10.jpg"]
    test_image_dir = "F:/project/pytorch-learning-tutorials/image_classification/dataset/test_images/images"
    # test_image_id_list = ["2334/1828778894_271415878a_o.jpg", "3149/2355285447_290193393a_o.jpg",
    #                       "2090/1792526652_8f37410561_o.jpg", "2099/1791684639_044827f860_o.jpg"]
    # test_image_dir = "https://farm3.staticflickr.com/"
    epoch_num = 1  # 总样本循环次数
    test_batch_size = 4  # 训练时的一组数据的大小
    train_data_nums = 10
    num_workers = 4
    max_iterate = int((train_data_nums + test_batch_size - 1) / test_batch_size * epoch_num)  # 总迭代次数

    train_data = TorchDataset(image_id_list=test_image_id_list,
                              image_dir=test_image_dir,
                              resize_height=height,
                              resize_width=width,
                              repeat=1)
    # 使用默认的default_collate会报错
    # train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
    # 使用自定义的collate_fn
    train_loader = DataLoader(dataset=train_data,
                              batch_size=test_batch_size,
                              shuffle=False,
                              collate_fn=custom_collate_fn.collate_fn_dict)
    # [1]使用epoch方法迭代，TorchDataset的参数repeat=1
    dict_test = {}

    for epoch in range(epoch_num):
        for step, batch_data in enumerate(train_loader):
            if batch_data:
                # print("batch_data:{}".format(batch_data))
                img = batch_data["image"][0]
                cv2.imshow("image", img)
                cv2.waitKey()
