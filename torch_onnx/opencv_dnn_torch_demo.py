# -*-coding: utf-8 -*-
"""
    @Project: pytorch-learning-tutorials
    @File   : opencv_onnx.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-05-23 10:55:12
"""
import cv2
import numpy as np
from utils import fun
import glob
import os

def getOutputsNames(net):
    '''
    Get the names of the output layers
    :param net:
    :return:
    '''
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    output_layers=[layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    print("output_layers:{}".format(output_layers))
    return output_layers

def opencv_dnn_torch_predict(model_path, image_dir, labels_name):
    net_width = 224
    net_height = 224
    # model = cv2.dnn.readNetFromONNX(model_path)
    model = cv2.dnn.readNetFromTorch(model_path,isBinary=False)
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL) # DNN_TARGET_CPU,DNN_TARGET_FPGA,DNN_TARGET_MYRIAD,DNN_TARGET_OPENCL
    output_layers=getOutputsNames(model)
    images_list=glob.glob(os.path.join(image_dir,'*.jpg'))
    for image_path in images_list:
        print("--------------------------------------")
        frame = cv2.imread(image_path)
        blob = cv2.dnn.blobFromImage(frame, scalefactor=1 / 255, size=(net_width, net_height), mean=[0, 0, 0], swapRB=True, crop=False)
        # Sets the input to the network
        model.setInput(blob)
        # Runs the forward pass to get output of the output layers
        output = model.forward(output_layers)
        score=output[0]
        # print("output shape: ", output.shape)
        pre_score = fun.softmax(score, axis=1)
        pre_index = np.argmax(pre_score, axis=1)
        max_score = pre_score[:, pre_index]
        pre_label = labels_name[pre_index]
        print(score)
        print("{} is: pre labels:{},name:{} score: {}".format(image_path, pre_index, pre_label, max_score))

if __name__=="__main__":
    '''
    '''
    # model_path = './models/resRegularBn/model.onnx'
    # model_path = './models/torch_models/trace_model.pth'
    model_path = './models/torch_models/model.pt'

    labels_filename = './dataset/label.txt'
    labels_name = np.loadtxt(labels_filename, str, delimiter='\t')
    image_dir='dataset/test_image'
    opencv_dnn_torch_predict(model_path, image_dir, labels_name)

