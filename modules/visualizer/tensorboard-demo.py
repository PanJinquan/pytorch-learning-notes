# -*-coding: utf-8 -*-
"""
    @Project: pytorch-learning-tutorials
    @File   : vision_test01.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-06-16 10:45:23
    @url    :《深度学习tensorwatch可视化工具》https://blog.csdn.net/qq_29592829/article/details/90517303
    conda install graphviz
    conda install torchvision
    conda install tensorwatch
"""
import time
import tensorboardX
import random
import torch
from tensorboardX import SummaryWriter
import os
import PIL.Image as Image
import numpy as np
import torchvision.transforms as transforms


def writer_add_scalars(writer, main_tag, line_name_list, line_val_list, global_step):
    tag_scalar_dict = {}
    for name, val in zip(line_name_list, line_val_list):
        tag_scalar_dict[name] = val
    writer.add_scalars(main_tag, tag_scalar_dict, global_step)


def writer_add_images(writer, img_name_list, img_tensor_list, global_step):
    # for i,img_tensor in enumerate(img_tensor_list):
    #     img_tensor_list[i] = img_tensor.unsqueeze(0)  # 增加一个维度
    # batch_image_data = torch.cat(img_tensor_list)
    # writer.add_images(main_tag, batch_image_data, global_step)
    for name, image_tensor in zip(img_name_list, img_tensor_list):
        writer.add_image(name, image_tensor, global_step)


def tensorwatch2(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)  # writer for buffering intermedium results
    for step in range(100):
        iteration = step
        face_loss = (step % 20) / 20
        head_loss = (step % 10) / 10
        loss = face_loss + head_loss
        writer.add_scalars('train-loss-epoch', {'loss': loss,
                                                'face_loss': face_loss,
                                                "head_loss": head_loss}, step)
        db_name_list = ["loss", "face_loss", "head_loss"]
        val_list = [loss, face_loss, head_loss]
        writer_add_scalars(writer, "train-loss-step", db_name_list, val_list, step)
        data1 = np.random.rand(100, 100, 3)
        data2 = np.random.rand(100, 100, 3)
        data1 = np.asarray(data1, dtype=np.uint8)
        data2 = np.asarray(data2, dtype=np.uint8)

        roc_curve1 = Image.fromarray(data1)
        roc_curve2 = Image.fromarray(data2)
        roc_curve_tensor1 = transforms.ToTensor()(roc_curve1)
        roc_curve_tensor2 = transforms.ToTensor()(roc_curve2)
        img_tensor_list = [roc_curve_tensor1, roc_curve_tensor2]
        writer_add_images(writer, db_name_list, img_tensor_list, step)
        # writer.add_image('ROC_Curve', roc_curve_tensor1, step)
        print("step:{},loss:{},face_loss:{}, head_loss:{}".format(iteration, loss, face_loss, head_loss))


if __name__ == "__main__":
    log_dir = "./work_dir"
    tensorwatch2(log_dir)
