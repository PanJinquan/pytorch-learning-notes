# -*-coding: utf-8 -*-
"""
    @Project: pytorch-learning-tutorials
    @File   : pre_dataset.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-05-22 13:46:27
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import json
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from pathlib import Path
from utils import image_processing

classLabels = ["desert", "mountains", "sea", "sunset", "trees"]
print(torch.__version__)

def convert_data(dataset_dir,labels_path,out_labels_path):
    df = pd.DataFrame({"image": sorted([int(x.name.strip(".jpg")) for x in Path(dataset_dir).iterdir()])})
    df.image = df.image.astype(np.str)
    print(df.dtypes)
    df.image = df.image.str.cat([".jpg"] * len(df))
    for label in classLabels:
        df[label] = 0
    with open(labels_path) as infile:
        s = "["
        s = s + ",".join(infile.readlines())
        s = s + "]"
        s = np.array(eval(s))
        s[s < 0] = 0
        df.iloc[:, 1:] = s
    df.to_csv(out_labels_path, index=False)
    print(df.head(10))
    del df

def visualizeImage(df,idx):
    fd = df.iloc[idx]
    image = fd.image
    label = fd[1:].tolist()
    print(image)
    image = Image.open(os.path.join(dataset_dir,image))
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.grid(False)
    classes = np.array(classLabels)[np.array(label, dtype=np.bool)]
    for i, s in enumerate(classes):
        ax.text(0, i * 20, s, verticalalignment='top', color="white", fontsize=16, weight='bold')
    plt.show()

def load_data(dataset_dir,csv_labels_path):
    # ## Visulaize the data
    # ### Data distribution
    df = pd.read_csv(csv_labels_path)
    fig1, ax1 = plt.subplots()
    df.iloc[:, 1:].sum(axis=0).plot.pie(autopct='%1.1f%%', shadow=True, startangle=90, ax=ax1)
    ax1.axis("equal")
    plt.show()

    # ### Correlation between different classes
    import seaborn as sns
    sns.heatmap(df.iloc[:, 1:].corr(), cmap="RdYlBu", vmin=-1, vmax=1)

    # looks like there is no correlation between the labels
    # ### Visualize images
    visualizeImage(df,52)

    # Images in the dataset have different sizes to lets take a mean size while resizing 224*224
    l = []
    for i in df.image:
        with Image.open(Path(dataset_dir) / i) as f:
            l.append(f.size)
    np.array(l).mean(axis=0), np.median(np.array(l), axis=0)
    return df


# Create Data pipeline
class MyDataset(Dataset):
    def __init__(self, csv_file, img_dir, transforms=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transforms = transforms

    def __getitem__(self, idx):
        # d = self.df.iloc[idx.item()]
        d = self.df.iloc[idx]
        image = Image.open(self.img_dir / d.image).convert("RGB")
        label = torch.tensor(d[1:].tolist(), dtype=torch.float32)

        if self.transforms is not None:
            image = self.transforms(image)
        return image, label

    def __len__(self):
        return len(self.df)


if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_dir = '../image_scene_data/original'
    labels_path = '../image_scene_data/labels.json'
    out_labels_path = '../image_scene_data/data.csv'
    convert_data(dataset_dir, labels_path, out_labels_path)
    df = load_data(dataset_dir,csv_labels_path=out_labels_path)

    batch_size = 5
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])

    dataset = MyDataset(csv_file=out_labels_path, img_dir=Path(dataset_dir), transforms=transform)
    valid_no = int(len(dataset) * 0.12)
    trainset, valset = random_split(dataset, [len(dataset) - valid_no, valid_no])
    print(f"trainset len: {len(trainset)} valset len: {len(valset)}")
    dataloader = {"train": DataLoader(trainset, shuffle=True, batch_size=batch_size),
                  "val": DataLoader(valset, shuffle=True, batch_size=batch_size)}
    phase='train'
    max_iterate=5
    for step,(batch_data, batch_target) in enumerate(dataloader[phase]):
        # load the data and target to respective device
        batch_data, batch_target = batch_data.to(device), batch_target.to(device)
        image=batch_data[0,:]
        image=image.cpu().numpy()#image=np.array(image)
        image = image.transpose(1, 2, 0)  # 通道由[c,h,w]->[h,w,c]
        image_processing.show_image("image",image)
        print("step:{},batch_image.shape:{},batch_label:{}".format(step,batch_data.shape,batch_target))
        # batch_x, batch_y = Variable(batch_x), Variable(batch_y)
        if step>=max_iterate:
            break