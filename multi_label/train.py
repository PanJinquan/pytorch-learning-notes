# -*-coding: utf-8 -*-
"""
    @Project: pytorch-learning-tutorials
    @File   : train.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-05-22 14:32:46
"""

import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets ,models , transforms
import json
from torch.utils.data import Dataset, DataLoader ,random_split
from core import pre_dataset
from PIL import Image
from pathlib import Path
from tqdm import trange
from sklearn.metrics import precision_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classLabels = pre_dataset.classLabels


def create_head(num_features, number_classes, dropout_prob=0.5, activation_func=nn.ReLU):
    features_lst = [num_features, num_features // 2, num_features // 4]
    layers = []
    for in_f, out_f in zip(features_lst[:-1], features_lst[1:]):
        layers.append(nn.Linear(in_f, out_f))
        layers.append(activation_func())
        layers.append(nn.BatchNorm1d(out_f))
        if dropout_prob != 0: layers.append(nn.Dropout(dropout_prob))
    layers.append(nn.Linear(features_lst[-1], number_classes))
    return nn.Sequential(*layers)


def step_train(img_dir,csv_file):
    batch_size = 32
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])

    dataset = pre_dataset.MyDataset(csv_file, Path(img_dir), transform)
    valid_no = int(len(dataset) * 0.12)
    trainset, valset = random_split(dataset, [len(dataset) - valid_no, valid_no])
    print(f"trainset len: {len(trainset)} valset len: {len(valset)}")
    dataloader = {"train": DataLoader(trainset, shuffle=True, batch_size=batch_size),
                  "val": DataLoader(valset, shuffle=True, batch_size=batch_size)}

    # ## Model Definition

    model = models.resnet50(pretrained=True)  # load the pretrained model
    num_features = model.fc.in_features       # get the no of on_features in last Linear unit
    print("num_features:{}".format(num_features))
    ## freeze the entire convolution base
    for param in model.parameters():
        param.requires_grad_(False)

    top_head = create_head(num_features, len(classLabels))  # because ten classes
    model.fc = top_head  # replace the fully connected layer

    # ## Optimizer and Criterion
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()

    # specify optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    sgdr_partial = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0.005)
    train(model, dataloader, criterion, optimizer, sgdr_partial, num_epochs=10)


def train(model, data_loader, criterion, optimizer, scheduler, num_epochs=5):
    for epoch in trange(num_epochs, desc="Epochs"):
        result = []
        for phase in ['train', 'val']:
            if phase == "train":  # put the model in training mode
                model.train()
                scheduler.step()
            else:  # put the model in validation mode
                model.eval()

            # keep track of training and validation loss
            running_loss = 0.0
            running_corrects = 0.0

            for data, target in data_loader[phase]:
                # load the data and target to respective device
                data, target = data.to(device), target.to(device)

                with torch.set_grad_enabled(phase == "train"):
                    # feed the input
                    output = model(data)
                    # calculate the loss
                    loss = criterion(output, target)
                    preds = torch.sigmoid(output).data > 0.5
                    preds = preds.to(torch.float32)

                    if phase == "train":
                        # backward pass: compute gradient of the loss with respect to model parameters
                        loss.backward()
                        # update the model parameters
                        optimizer.step()
                        # zero the grad to stop it from accumulating
                        optimizer.zero_grad()

                # statistics
                running_loss += loss.item() * data.size(0)
                running_corrects += f1_score(target.to("cpu").to(torch.int).numpy(),
                                             preds.to("cpu").to(torch.int).numpy(), average="samples") * data.size(0)

            epoch_loss = running_loss / len(data_loader[phase].dataset)
            epoch_acc = running_corrects / len(data_loader[phase].dataset)

            result.append('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
        print(result)


if __name__=="__main__":
    img_dir = 'dataset/trainval'
    csv_file = 'dataset/data.csv'
    step_train(img_dir, csv_file)
