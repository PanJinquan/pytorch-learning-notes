# -*-coding: utf-8 -*-
"""
    @Project: pytorch-learning-tutorials
    @File   : predict.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-03-11 10:44:16
"""

import torch
import torch.optim as optim
from torch import  nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from utils import dataset,image_processing,fun
from core import resnet,resNetBatchNorm,resRegularBn,squeezenet
from PIL import Image
import os,glob
import numpy as np

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device='cpu'
print("-----device:{}".format(device))
print("-----Pytorch version:{}".format(torch.__version__))


def predict(model_path,image_dir,labels_filename):
    resize_height=224
    resize_width=224

    labels = np.loadtxt(labels_filename, str, delimiter='\t')

    model= squeezenet.nets(num_classes=5)

    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    # image = Image.open(image_path)
    # test_transform = transforms.Compose([
    #     transforms.Resize(size=(resize_height, resize_width)),
    #     transforms.ToTensor(),  # 吧shape=(H,W,C)->换成shape=(C,H,W),并且归一化到[0.0, 1.0]的torch.FloatTensor类型
    # ])
    test_transform = transforms.Compose([
        transforms.Resize(size=(resize_height, resize_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    images_list=glob.glob(os.path.join(image_dir,'*.jpg'))
    for image_path in images_list:
        image = Image.open(image_path).convert('RGB')
        image_tensor = test_transform(image).float()
        # Add an extra batch dimension since pytorch treats all images as batches
        image_tensor = image_tensor.unsqueeze_(0)
        image_tensor = image_tensor.to(device)
        # Turn the input into a Variable
        input = Variable(image_tensor)

        # Predict the class of the image
        output = model(input)
        output = output.cpu().data.numpy()#gpu:output.data.numpy()
        pre_score=fun.softmax(output,axis=1)
        pre_index =np.argmax(pre_score, axis=1)
        max_score = pre_score[:,pre_index]
        pre_label = labels[pre_index]
        print("{} is: pre labels:{},name:{} score: {}".format(image_path,pre_index,pre_label, max_score))



if __name__=='__main__':

    labels_filename='./dataset/images/label.txt'
    #
    image_dir='./dataset/images/test_image'
    model_path='./models/model_epoch81_step0.model'
    predict(model_path, image_dir,labels_filename)