# -*-coding: utf-8 -*-
"""
    @Project: PythonAPI
    @File   : pascal_voc.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-05-09 20:39:21
"""
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
from utils import file_processing,image_processing

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return [x,y,w,h]

def convert_rect(size, box):
    #box=[xmin,xmax,ymin,ymax]
    dw = 1
    dh = 1
    x = box[0]
    y = box[2]
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return [x,y,w,h]

def get_annotation(annotations_file, classes):
    tree=ET.parse(annotations_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    rects=[]
    class_name=[]
    class_id=[]
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        # b=[xmin,xmax,ymin,ymax]
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        #bb = convert((w,h), b)
        rect = convert_rect((w,h), b)
        rects.append(rect)
        class_name.append(cls)
        class_id.append(cls_id)
    return rects,class_name,class_id

