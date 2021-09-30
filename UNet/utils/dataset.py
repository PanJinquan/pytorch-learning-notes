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
from utils import image_processing,PairRandomCrop
import os,eval
from PIL import Image

def read_image(path):
    return Image.open(path).convert('RGB')
    # return Image.open(path).convert('L')

# normalize_transform=transforms.Compose([
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))#给定均值(R,G,B) 方差（R，G，B），将会把Tensor正则化
# ])

class TorchDataset(Dataset):
    def __init__(self, filename, orig_dir, dest_dir,resize_height=256, resize_width=256, repeat=1, transform=None):
        '''
        :param filename: 数据文件TXT：格式：imge_name.jpg label1_id labe2_id
        :param orig_dir: 图片路径：image_dir+imge_name.jpg构成图片的完整路径
        :param dest_dir
        :param resize_height 为None时，不进行缩放
        :param resize_width  为None时，不进行缩放，
                              PS：当参数resize_height或resize_width其中一个为None时，可实现等比例缩放
        :param repeat: 所有样本数据重复次数，默认循环一次，当repeat为None时，表示无限循环<sys.maxsize
        :param transform:预处理
        '''
        self.images_list = self.read_file(filename)
        self.orig_dir = orig_dir
        self.dest_dir = dest_dir
        self.len = len(self.images_list)
        self.repeat = repeat
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.transform= transform

    def __getitem__(self, i):
        index = i % self.len
        # print("i={},index={}".format(i, index))
        image_name = self.images_list[index]
        orig_image_path = os.path.join(self.orig_dir, image_name)
        dest_image_path = os.path.join(self.dest_dir, image_name)
        orig_image = self.load_data(orig_image_path)
        dest_image = self.load_data(dest_image_path)

        orig_image = self.data_preproccess(orig_image)
        # orig_image=normalize_transform(orig_image)     # 对输入的数据进行正则化
        dest_image = self.data_preproccess(dest_image) # 模型输出使用sigmoid函数，所以label不需要正则化
        dest_image=dest_image[0,:,:].transpose(0,1)

        return orig_image, dest_image

    def __len__(self):
        if self.repeat == None:
            data_len = 10000000
        else:
            data_len = len(self.images_list) * self.repeat
        return data_len

    def read_file(self,file):
        '''
        :param file:
        :return:
        '''
        with open(file, mode="r") as f:
            content_list = f.readlines()
            # 按空格分隔
            # content_list = [content.rstrip().split(" ") for content in content_list]
            content_list = [content.rstrip() for content in content_list]

        return content_list

    def load_data(self, path):
        '''
        加载数据
        :param path:
        :param resize_height:
        :param resize_width:
        :param normalization: 是否归一化
        :return:
        '''
        image = read_image(path)
        # image = image_processing.read_image(path)#用opencv读取图像
        return image

    def data_preproccess(self, data):
        '''
        数据预处理
        :param data:
        :return:
        '''
        if self.transform is not None:
            data = self.transform(data)
        return data

if __name__=='__main__':
    resize_height = 224
    resize_width = 224
    # 相关预处理的初始化
    '''
    class torchvision.transforms.ToTensor把shape=(H,W,C)的像素值范围为[0, 255]的PIL.Image或者numpy.ndarray数据
    转换成shape=(C,H,W)的像素数据，并且被归一化到[0.0, 1.0]的torch.FloatTensor类型。
    '''
    train_transform = transforms.Compose([
        transforms.Resize(size=(resize_height, resize_width)),
        # transforms.RandomHorizontalFlip(),#随机翻转图像
        # transforms.RandomCrop(size=(resize_height, resize_width), padding=4),      # 随机裁剪
        PairRandomCrop.PairRandomCrop(size=(resize_height, resize_width), padding=4),#配对裁剪
        transforms.ToTensor(),  # 吧shape=(H,W,C)->换成shape=(C,H,W),并且归一化到[0.0, 1.0]的torch.FloatTensor类型
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))#给定均值(R,G,B) 方差（R，G，B），将会把Tensor正则化
    ])

    # train_filename="../dataset/test_images/train2.txt"
    # image_dir='../dataset/test_images/images'

    train_filename="E:/git/dataset/tgs-salt-identification-challenge/train/my_train.txt"
    orig_dir= 'E:/git/dataset/tgs-salt-identification-challenge/train/my_images'
    masks_dir= 'E:/git/dataset/tgs-salt-identification-challenge/train/my_masks'


    epoch_num=1   #总样本循环次数
    batch_size=2  #训练时的一组数据的大小
    train_data_nums=10
    max_iterate=int((train_data_nums+batch_size-1)/batch_size*epoch_num) #总迭代次数

    train_data = TorchDataset(filename=train_filename,
                              orig_dir=orig_dir,
                              dest_dir=masks_dir,
                              resize_height=resize_height,
                              resize_width=resize_width,
                              repeat=1,
                              transform=train_transform)
    # test_data = TorchDataset(test_filename, image_dir,resize_height,resize_width,1,transform=None)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size,shuffle=True)
    # test_loader = DataLoader(dataset=test_data, batch_size=batch_size,shuffle=False)

    # [1]使用epoch方法迭代，TorchDataset的参数repeat=1
    for epoch in range(epoch_num):
        for step,(batch_orig_image, batch_mask_image) in enumerate(train_loader):
            image_processing.show_batch_image(batch_orig_image, index=0)
            image_processing.show_batch_image(batch_mask_image, index=0)

            # iou_metric=eval.get_batch_iou_mean(batch_mask_image.detach().cpu().numpy(),batch_mask_image.cpu().numpy())
            # iou=eval.get_iou(batch_mask_image.detach().cpu().numpy(),batch_mask_image.cpu().numpy())
            # print(iou_metric)
            # print(iou)


            if step==3:
                break
