# -*-coding: utf-8 -*-
"""
    @Project: pytorch-learning-tutorials
    @File   : convert_torch_to_onnx_caffe2.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-03-18 18:59:09
"""
from torch.autograd import Variable
import torch
from core import resnet,resNetBatchNorm,resRegularBn,squeezenet
from utils import convert_onnx_to_caffe2,fun,image_processing

from caffe2.proto import caffe2_pb2
from caffe2.python import core, net_drawer, net_printer, visualize, workspace, utils
import numpy as np
import os
from PIL import Image
import glob
from torchvision import transforms

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device='cpu'
print("-----device:{}".format(device))
print("-----Pytorch version:{}".format(torch.__version__))

def convert_onnx(model_path,onnx_path,num_classes=5):
    batch_size=1  # 随便一个数
    # 导入模型
    model = resRegularBn.nets(num_classes=num_classes)

    model.load_state_dict(torch.load(model_path,map_location=device))
    # model.to(device)

    # set the train mode to false since we will only run the forward pass.
    model.train(False)

    # 将pytorch模型转为onnx格式
    x = Variable(torch.randn(batch_size,3,224,224), requires_grad=True)
    # y=model(x)
    torch_out = torch.onnx._export(model,
                                  x,
                                   onnx_path,
                                   export_params=True
                                  )


def caffe2_predictor_v1(init_net_path, predict_net_path, image_dir, labels_filename):
    '''
    https://discuss.pytorch.org/t/caffe2-mobilenetv2-quantized-using-caffe2-blobistensortype-blob-cpu-blob-is-not-a-cpu-tensor-325/29065
    :param init_net_path:
    :param predict_net_path:
    :param image_dir:
    :param labels_filename:
    :return:
    '''
    resize_height=224
    resize_width=224

    labels = np.loadtxt(labels_filename, str, delimiter='\t')
    test_transform = transforms.Compose([
        transforms.Resize(size=(resize_height, resize_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    init_def = caffe2_pb2.NetDef()
    with open(init_net_path, "rb") as f:
        init_def.ParseFromString(f.read())
        workspace.RunNetOnce(init_def.SerializeToString())

    predict_def = caffe2_pb2.NetDef()
    with open(predict_net_path, "rb") as f:
        predict_def.ParseFromString(f.read())
        workspace.CreateNet(predict_def.SerializeToString())
    print(net_printer.to_string(predict_def))
    print('------------------------Running net-------------------------')

    images_list=glob.glob(os.path.join(image_dir,'*.jpg'))
    for image_path in images_list:
        image = Image.open(image_path).convert('RGB')
        image_tensor = test_transform(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input=image_tensor.numpy()
        # print("input.shape:{}".format(input.shape))


        workspace.FeedBlob('0', input)       # Feed the inputArray into the input blob (index) of the network
        workspace.RunNetOnce(predict_def)
        output = workspace.FetchBlob("125")  # Fetch the result from the output blob (index) of the network
        # output = np.asarray(output)
        # print("output shape: ", output.shape)
        pre_score = fun.softmax(output, axis=1)
        pre_index = np.argmax(pre_score, axis=1)
        max_score = pre_score[:, pre_index]
        pre_label = labels[pre_index]
        print("{} is: pre labels:{},name:{} score: {}".format(image_path, pre_index, pre_label, max_score))

def caffe2_predictor_v2(init_net_path, predict_net_path, image_dir, labels_filename):
    '''
    https://github.com/caffe2/tutorials/blob/master/Loading_Pretrained_Models.ipynb
    :param init_net_path:
    :param predict_net_path:
    :param image_dir:
    :param labels_filename:
    :return:
    '''
    resize_height=224
    resize_width=224

    labels = np.loadtxt(labels_filename, str, delimiter='\t')
    test_transform = transforms.Compose([
        transforms.Resize(size=(resize_height, resize_width)),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    # Read the contents of the input protobufs into local variables
    with open(init_net_path, "rb") as f:
        init_net = f.read()
    with open(predict_net_path, "rb") as f:
        predict_net = f.read()

    predict_def = caffe2_pb2.NetDef()
    predict_def.ParseFromString(predict_net)
    print(net_printer.to_string(predict_def))
    # 加载图像
    # workspace.RunNetOnce(init_net)
    # workspace.CreateNet(predict_net)
    p = workspace.Predictor(init_net, predict_net)

    images_list=glob.glob(os.path.join(image_dir,'*.jpg'))
    for image_path in images_list:
        print("--------------------------------------")

        image = Image.open(image_path).convert('RGB')
        image_tensor = test_transform(image).float()
        # Add an extra batch dimension since pytorch treats all images as batches
        image_tensor = image_tensor.unsqueeze_(0)

        input=image_tensor.numpy()
        print("input.shape:{}".format(input.shape))
        # output = p.run({'0': input})
        output = p.run([input])
        #
        output = np.asarray(output)
        output=np.squeeze(output, axis=(0,))
        print(output)
        # print("output shape: ", output.shape)
        pre_score = fun.softmax(output, axis=1)
        pre_index = np.argmax(pre_score, axis=1)
        max_score = pre_score[:, pre_index]
        pre_label = labels[pre_index]
        print("{} is: pre labels:{},name:{} score: {}".format(image_path, pre_index, pre_label, max_score))


def caffe2_predictor_v3(init_net_path, predict_net_path, image_dir, labels_filename):
    '''
    https://github.com/caffe2/tutorials/blob/master/Loading_Pretrained_Models.ipynb
    :param init_net_path:
    :param predict_net_path:
    :param image_dir:
    :param labels_filename:
    :return:
    '''
    resize_height=224
    resize_width=224

    labels = np.loadtxt(labels_filename, str, delimiter='\t')
    test_transform = transforms.Compose([
        transforms.Resize(size=(resize_height, resize_width)),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    # Read the contents of the input protobufs into local variables
    with open(init_net_path, "rb") as f:
        init_net = f.read()
    with open(predict_net_path, "rb") as f:
        predict_net = f.read()

    predict_def = caffe2_pb2.NetDef()
    predict_def.ParseFromString(predict_net)
    print(net_printer.to_string(predict_def))
    # 加载图像
    # workspace.RunNetOnce(init_net)
    # workspace.CreateNet(predict_net)
    p = workspace.Predictor(init_net, predict_net)

    images_list=glob.glob(os.path.join(image_dir,'*.jpg'))
    for image_path in images_list:
        print("--------------------------------------")
        image = image_processing.read_image(image_path,resize_height,resize_width,normalization=True)
        # Add an extra batch dimension since pytorch treats all images as batches
        image=image.transpose(2, 0, 1)  # 通道由[h,w,c]->[c,h,w]
        input=image[np.newaxis,:]
        print("input.shape:{}".format(input.shape))
        output = p.run([input])

        output = np.asarray(output)
        output=np.squeeze(output, axis=(0,))
        print(output)
        # print("output shape: ", output.shape)
        pre_score = fun.softmax(output, axis=1)
        pre_index = np.argmax(pre_score, axis=1)
        max_score = pre_score[:, pre_index]
        pre_label = labels[pre_index]
        print("{} is: pre labels:{},name:{} score: {}".format(image_path, pre_index, pre_label, max_score))



if __name__=='__main__':

    model_path = './models/model_epoch53_step0.model'
    onnx_path = './models/pb/model.onnx'
    # convert_onnx(model_path, onnx_path)

    init_net_path = 'models/pb/init_net.pb'
    predict_net_path = 'models/pb/predict_net.pb'
    # convert_onnx_to_caffe2.convert_onnx_to_caffe2_v2(onnx_path, init_net_path, predict_net_path)

    labels_filename='./dataset/images/label.txt'
    image_dir='./dataset/images/test_image'
    caffe2_predictor_v3(init_net_path, predict_net_path, image_dir, labels_filename)