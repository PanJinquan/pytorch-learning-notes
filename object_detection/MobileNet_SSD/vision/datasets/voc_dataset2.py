import numpy as np
import pathlib
import xml.etree.ElementTree as ET
import cv2
import os
from torch.utils.data import DataLoader
from vision.ssd.data_preprocessing import TrainAugmentation
from vision.ssd.config import mobilenetv1_ssd_config
from vision.ssd.ssd import MatchPrior
from utils import image_processing

class Dataset:
    def __init__(self,filename, transform=None, target_transform=None):
        """Dataset for VOC data.
        """
        self.transform = transform
        self.target_transform = target_transform
        self.image_path = Dataset._read_image_ids(filename)#获取所有图像名
        print("nums:{}".format(len(self.image_path)))
    def __getitem__(self, index):

        image_path = self.image_path[index]
        boxes, labels = self._get_annotation(image_path)
        image = self._read_image(image_path)
        # # 测试读取的图片
        # print(" boxes.shape:{},boxes:{}".format(boxes.shape,boxes))
        # print("labels.shape:{},labels:{}".format(labels.shape,labels))
        # image_processing.show_image_boxes("image",image,boxes)

        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image, boxes, labels

    def get_image(self, index):
        image_id = self.image_path[index]
        image = self._read_image(image_id)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        image_id = self.image_path[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.image_path)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids

    def _get_annotation(self, image_path):
        dirname = os.path.dirname(image_path)
        image_idx=os.path.basename(image_path)[:-len('.jpg')]
        labels_file=os.path.join(os.path.dirname(dirname),'labels',image_idx+'.txt')
        content_list =self.read_data(labels_file)
        boxes = []
        labels = []
        for content in content_list:
            class_id = content[0]
            # VOC dataset format follows Matlab, in which indexes start from 0
            x1 = float(content[1]) - 1
            y1 = float(content[2]) - 1
            w = float(content[3]) - 1
            h = float(content[4]) - 1
            x2=x1+w
            y2=y1+h
            boxes.append([x1, y1, x2, y2])
            labels.append(class_id)
        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64))

    def _read_image(self, image_file):
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def write_data(self,file, content_list, model):
        with open(file, mode=model) as f:
            for line in content_list:
                f.write(line + "\n")

    def read_data(self,file):
        '''
        :param file:
        :return:
        '''
        with open(file, mode="r") as f:
            content_list = f.readlines()
            #按空格分隔
            content_list = [content.rstrip().split(" ") for content in content_list]
        return content_list

if __name__=='__main__':
    device = 'cpu'
    config = mobilenetv1_ssd_config
    train_filename='E:/git/VOC0712_dataset/train.txt'
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)
    train_dataset = Dataset(train_filename, transform=train_transform,target_transform=target_transform)
    train_loader = DataLoader(train_dataset, batch_size=2,num_workers=0,shuffle=False)
    test_nums=10
    for i, data in enumerate(train_loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        image = images[0, :]
        image = image.numpy()  # image=np.array(image)
        image = image.transpose(1, 2, 0)  # 通道由[c,h,w]->[h,w,c]
        image_processing.cv_show_image("image", image,)
        # print("boxes:{},boxes:{}".format(boxes.shape,boxes))
        # print("labels:{},labels:{}".format(labels.shape,labels))

        if i==test_nums:
            break
