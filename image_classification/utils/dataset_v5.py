# -*-coding: utf-8 -*-
"""
    @Project: pytorch-learning-tutorials
    @File   : dataset.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-03-07 18:45:06
"""
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils import dataset_collate,image_processing
import os
import cv2


def getDataTransform(resize_height, resize_width):
    '''
    图像预处理
    :param resize_height:
    :param resize_width:
    :return:
    '''
    data_transform = transforms.Compose([
        transforms.Resize(size=(resize_height, resize_width)),
        # transforms.RandomHorizontalFlip(),#随机翻转图像
        # transforms.RandomCrop(size=(resize_height, resize_width), padding=4),  # 随机裁剪
        # transforms.ToTensor(),  # 吧shape=(H,W,C)->换成shape=(C,H,W),并且归一化到[0.0, 1.0]的torch.FloatTensor类型
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))#给定均值(R,G,B) 方差（R，G，B），将会把Tensor正则化
    ])
    return data_transform

def getDataLoader(dataset,batch_size,shuffle=False,num_workers=0,collate_fn=dataset_collate.collate_fn):
    '''
    DataLoader,产生迭代数据
    :param dataset:
    :param batch_size:
    :param shuffle:
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
    def __init__(self, image_id_list, image_dir, resize_height=None, resize_width=None, repeat=1):
        '''
        :param filename: 数据文件TXT：格式：imge_name.jpg label1_id labe2_id
        :param image_dir: 图片路径：image_dir+imge_name.jpg构成图片的完整路径
        :param resize_height 为None时，不进行缩放
        :param resize_width  为None时，不进行缩放，
                              PS：当参数resize_height或resize_width其中一个为None时，可实现等比例缩放
        :param repeat: 所有样本数据重复次数，默认循环一次，当repeat为None时，表示无限循环<sys.maxsize
        '''
        self.image_dir = image_dir
        self.image_id_list=image_id_list
        self.len = len(image_id_list)
        self.repeat = repeat
        self.resize_height = resize_height
        self.resize_width = resize_width

    def __getitem__(self, i):
        index = i % self.len
        # print("i={},index={}".format(i, index))
        image_id = self.image_id_list[index]
        if self.image_dir is None:
            image_path=image_id
        else:
            image_path = os.path.join(self.image_dir, image_id)
        img = self.load_data(image_path)

        if img is None:
            return None,image_id
        img = self.data_preproccess(img)
        if not isinstance(img,torch.Tensor):
            img=torch.from_numpy(img)
        return img,image_id

    def __len__(self):
        if self.repeat == None:
            data_len = 10000000
        else:
            data_len = len(self.image_id_list) * self.repeat
        return data_len

    def load_data(self, path, mode='BGR'):
        '''
        读取图片的函数
        :param path:
        :param mode: RGB or L
        :return:
        '''
        try:
            print("加载图片：{}".format(path))
            image = image_processing.read_image(path)
            cv2.waitKey(1000)
        except Exception as e:
            image=None
            print(e)
        return image

    def data_preproccess(self, image):
        '''
        数据预处理
        :param data:
        :return:
        '''
        if image is not None:
            image=image_processing.resize_image(image,self.resize_height, self.resize_width)
        return image

if __name__=='__main__':


    resize_height = 224
    resize_width = 224
    image_id_list=["1.jpg","2.jpg","3.jpg","4.jpg","5.jpg","6.jpg","7.jpg","8.jpg","9.jpg","10.jpg"]
    image_dir="../dataset/test_images/images"
    # 相关预处理的初始化
    '''class torchvision.transforms.ToTensor把shape=(H,W,C)的像素值范围为[0, 255]的PIL.Image或者numpy.ndarray数据
    # 转换成shape=(C,H,W)的像素数据，并且被归一化到[0.0, 1.0]的torch.FloatTensor类型。
    '''

    epoch_num=1   #总样本循环次数
    batch_size=2  #训练时的一组数据的大小
    train_data_nums=10
    max_iterate=int((train_data_nums+batch_size-1)/batch_size*epoch_num) #总迭代次数

    train_data = TorchDataset(image_id_list=image_id_list,
                              image_dir=image_dir,
                              resize_height=224,
                              resize_width=224,
                              repeat=1)
    # 使用默认的default_collate会报错
    # train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
    # 使用自定义的collate_fn
    # train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False, collate_fn=dataset_collate.collate_fn)
    train_loader =getDataLoader(dataset=train_data, batch_size=batch_size, shuffle=False, num_workers=2,collate_fn=dataset_collate.collate_fn)

    # [1]使用epoch方法迭代，TorchDataset的参数repeat=1
    for epoch in range(epoch_num):
        for step,(batch_image, batch_label) in enumerate(train_loader):
            if batch_image is None and batch_label is None:
                print("batch_image:{},batch_label:{}".format(batch_image, batch_label))
                continue
            image=batch_image[0]
            image=image.numpy()#image=np.array(image)
            cv2.imshow("image",image)
            cv2.waitKey()
            print("batch_image.shape:{},batch_label:{}".format(batch_image.shape,batch_label))
            # batch_x, batch_y = Variable(batch_x), Variable(batch_y)
