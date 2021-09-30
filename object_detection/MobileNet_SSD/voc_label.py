# -*-coding: utf-8 -*-
"""
    @Project: PythonAPI
    @File   : vocDemo.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-05-09 19:10:16
"""
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
from utils import file_processing,image_processing,pascal_voc
#for SSD，the first label is BACKGROUND：
# classes = ["BACKGROUND","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

classes=["BACKGROUND",'PCwall']
print("class_name:{}".format(classes))
def convert_annotation_list(annotations_list, image_dir, label_out_dir, class_names, image_type='.jpg', show=True):
    '''

    :param annotations_list:annotations列表
    :param image_dir:图片所在路径
    :param label_out_dir:输出label目录
    :param class_names:
    :param image_type:图片的类型，如.jpg ,.png
    :param show:
    :return:
    '''
    if not os.path.exists(label_out_dir):
        os.makedirs(label_out_dir)
    name_id_list=[]
    nums=len(annotations_list)

    for i, annotations_file in enumerate(annotations_list):
        name_id = os.path.basename(annotations_file)[:-len(".xml")]
        image_name=name_id + image_type
        image_path = os.path.join(image_dir, image_name)
        if not os.path.exists(image_path) :
            print("no image:{}".format(image_path))
            continue
        if not os.path.exists(annotations_file):
            print("no annotations:{}".format(annotations_file))
            continue
        out_file = os.path.join(label_out_dir, name_id + ".txt")
        rects, class_name, class_id = pascal_voc.get_annotation(annotations_file, class_names)
        content_list = [[c] + r for c, r in zip(class_id, rects)]
        name_id_list.append(name_id)
        file_processing.write_data(out_file, content_list, mode='w')
        if show:
            image = image_processing.read_image(image_path)
            image_processing.show_image_rects_text("image", image, rects, class_name)
        if i % 100 == 0 or i==nums-1:
            print("processing {}/{}".format(i+1, nums))
    return name_id_list

def convert_annotation_image(image_list, annotations_dir, label_out_dir, class_names, show=True):
    '''

    :param image_list: 图片列表
    :param annotations_dir: 图片对应annotations所在目录
    :param label_out_dir: label输出目录
    :param class_names:
    :param show: 显示
    :return:
    '''
    if not os.path.exists(label_out_dir):
        os.makedirs(label_out_dir)
    name_id_list = []
    nums=len(image_list)
    for i, image_path in enumerate(image_list):
        name_id = os.path.basename(image_path)[:-len(".jpg")]
        ann_name = name_id + '.xml'
        annotations_file = os.path.join(annotations_dir, ann_name)

        if not os.path.exists(image_path) :
            print("no image:{}".format(image_path))
            continue
        if not os.path.exists(annotations_file):
            print("no annotations:{}".format(annotations_file))
            continue
        out_file = os.path.join(label_out_dir, name_id + ".txt")
        rects, class_name, class_id = pascal_voc.get_annotation(annotations_file, class_names)
        content_list = [[c] + r for c, r in zip(class_id, rects)]
        name_id_list.append(name_id)
        file_processing.write_data(out_file, content_list, mode='w')
        if show:
            image = image_processing.read_image(image_path)
            image_processing.show_image_rects_text("image", image, rects, class_name)
        if i % 100 == 0 or i==nums-1:
            print("processing {}/{}".format(i+1, nums))
    return name_id_list

def convert_voc_label_annotations(annotations_dir, image_dir, label_out_dir,out_train_val_path, class_names, show=True):
    annotations_list=file_processing.get_files_list(annotations_dir,postfix=["*.xml"])
    print("have {} annotations files".format(len(annotations_list)))
    # 分割成train和val数据集
    factor=0.8
    train_num=int(factor*len(annotations_list))
    train_annotations_list=annotations_list[:train_num]
    val_annotations_list=annotations_list[train_num:]

    # 转换label数据
    print("doing train data .....")
    train_image_id=convert_annotation_list(train_annotations_list, image_dir, label_out_dir, class_names, image_type=".jpg", show=show)
    print("doing val data .....")
    val_image_id=convert_annotation_list(val_annotations_list, image_dir, label_out_dir, class_names, image_type=".jpg", show=show)
    print("done...ok!")

    # 保存图片id数据
    train_id_path=os.path.join(out_train_val_path, "train.txt")
    val_id_path=os.path.join(out_train_val_path, "val.txt")
    save_id(train_id_path, train_image_id, val_id_path, val_image_id)

def convert_voc_label_for_image(annotations_dir, image_dir, label_out_dir, out_train_val_path, class_names, show=True):
    image_list=file_processing.get_files_list(image_dir,postfix=["*.bmp"])
    print("have {} images".format(len(image_list)))
    # 分割成train和val数据集
    factor=0.8
    train_num=int(factor*len(image_list))
    train_image_list=image_list[:train_num]
    val_image_list=image_list[train_num:]

    # 转换label数据
    print("doing train data .....")
    train_image_id=convert_annotation_image(train_image_list, annotations_dir, label_out_dir, class_names, show=show)
    print("doing val data .....")
    val_image_id=convert_annotation_image(val_image_list, annotations_dir, label_out_dir, class_names, show=show)
    print("done...ok!")

    # 保存图片id数据
    train_id_path=os.path.join(out_train_val_path, "train.txt")
    val_id_path=os.path.join(out_train_val_path, "val.txt")
    save_id(train_id_path, train_image_id, val_id_path, val_image_id)

def save_id(train_id_path,train_id,val_id_path,val_id):
    if not os.path.exists(os.path.dirname(train_id_path)):
        os.makedirs(os.path.dirname(train_id_path))
    if not os.path.exists(os.path.dirname(val_id_path)):
        os.makedirs(os.path.dirname(val_id_path))
    # 保存图片id数据
    file_processing.write_list_data(train_id_path,train_id,mode="w")

    file_processing.write_list_data(val_id_path,val_id,mode="w")
    print("train num:{},save path:{}".format(len(train_id),train_id_path))
    print("val   num:{},save path:{}".format(len(val_id),val_id_path))


def label_test(image_dir, filename,class_names):
    basename = os.path.basename(filename)[:-len('.txt')]+".bmp"
    image_path=os.path.join(image_dir, basename)
    image=image_processing.read_image_gbk(image_path)
    data=file_processing.read_data(filename,split=" ")
    label_list,rect_list = file_processing.split_list(data, split_index=1)
    label_list = [l[0] for l in label_list]
    name_list=file_processing.decode_label(label_list,class_names)
    image_processing.show_image_rects_text("object2", image, rect_list, name_list)

def batch_label_test(label_dir,image_dir,classes):
    file_list=file_processing.get_files_list(label_dir,postfix=[".txt"])
    for filename in file_list:
        label_test(image_dir, filename,class_names=classes)

if __name__=="__main__":
    annotations_dir='E:/git/PCwall/20190513/dataset'   # xml标准文件
    label_out_dir= 'E:/git/PCwall/20190513/labels'     # 输出label文件
    image_dir="E:/git/PCwall/20190513/dataset"         # 图像文件所在目录
    out_train_val_path= "E:/git/PCwall/20190513"       # 输出 train/val 文件
    show=False                                  # 显示
    convert_voc_label_annotations(annotations_dir, image_dir, label_out_dir, out_train_val_path, classes, show=show)
    convert_voc_label_for_image(annotations_dir, image_dir, label_out_dir, out_train_val_path, classes, show=show)
    # batch_label_test(label_out_dir, image_dir,classes)

