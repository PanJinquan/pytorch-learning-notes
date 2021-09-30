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
from core import dataset
import numpy as np
import os
import train_resnet50
import glob
import PIL.Image as Image
from torchvision import datasets ,models , transforms
from sklearn.metrics import precision_score, f1_score

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device='cpu'
print("-----device:{}".format(device))
print("-----Pytorch version:{}".format(torch.__version__))

classLabels = dataset.classLabels




def predict(model_path,image_dir):
    resize_height=224
    resize_width=224
    num_classes = len(classLabels)
    threshold=0.4

    model = models.resnet50(pretrained=True)  # load the pretrained model
    num_features = model.fc.in_features  # get the no of on_features in last Linear unit
    print("num_features:{}".format(num_features))
    ## freeze the entire convolution base
    for param in model.parameters():
        param.requires_grad_(False)

    top_head = train_resnet50.create_head(num_features, num_classes)  # because ten classes
    model.fc = top_head  # replace the fully connected layer

    # ## Optimizer and Criterion
    model = model.to(device)

    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

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
        score = torch.sigmoid(output).data
        preds=score> threshold
        score = score.cpu().data.numpy()#gpu:output.data.numpy()


        # pre_label = classLabels[pre_index]
        print("{} is: score: {},preds :{}".format(image_path,score,preds))



if __name__=='__main__':
    #
    image_dir='./dataset/test_image'
    model_path='./models/model_epoch99_step0.model'
    predict(model_path, image_dir)