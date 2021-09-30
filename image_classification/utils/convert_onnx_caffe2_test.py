# -*-coding: utf-8 -*-
"""
    @Project: pytorch-learning-tutorials
    @File   : convert_onnx_caffe2.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-03-18 18:59:09
"""


from torch.autograd import Variable
import torch
from core import resnet,resNetBatchNorm,resRegularBn,squeezenet

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device='cpu'
print("-----device:{}".format(device))
print("-----Pytorch version:{}".format(torch.__version__))

model_path='./models/model_epoch81_step0.model'
batch_size=1  # 随便一个数
# 导入模型
model = squeezenet.nets(num_classes=5)
model.load_state_dict(torch.load(model_path))
model.to(device)

# set the train mode to false since we will only run the forward pass.
model.train(False)

# 将pytorch模型转为onnx格式
onnx_path='./models/pb/model.onnx'
x = Variable(torch.randn(batch_size,3,224,224), requires_grad=True)
torch_out = torch.onnx._export(model,
                              x,
                               onnx_path,
                               export_params=True
                              )

# 再将onnx转caffe2
import onnx
import caffe2.python.onnx.backend as onnx_caffe2_backend
# import onnx_caffe2.backend
# Load the ONNX ModelProto object. model is a standard Python protobuf object
model = onnx.load(onnx_path)

# prepare the caffe2 backend for executing the model this converts the ONNX model into a
# Caffe2 NetDef that can execute it. Other ONNX backends, like one for CNTK will be
# availiable soon.
prepared_backend = onnx_caffe2_backend.prepare(model)

# Construct a map from input names to Tensor data.
# The graph of the model itself contains inputs for all weight parameters, after the input image.
# Since the weights are already embedded, we just need to pass the input image.
# Set the first input.
W = {model.graph.input[0].name: x.data.numpy()}

# Run the Caffe2 net:
c2_out = prepared_backend.run(W)[0]

# extract the workspace and the model proto from the internal representation
c2_workspace = prepared_backend.workspace
c2_model = prepared_backend.predict_net

# Now import the caffe2 mobile exporter
from caffe2.python.predictor import mobile_exporter

# call the Export to get the predict_net, init_net. These nets are needed for running things on mobile
init_net, predict_net = mobile_exporter.Export(c2_workspace, c2_model, c2_model.external_input)

# Let's also save the init_net and predict_net to a file that we will later use for running them on mobile
init_net_path='models/pb/init_net.pb'
predict_net_path='models/pb/predict_net.pb'

with open(init_net_path, "wb") as fopen:
    fopen.write(init_net.SerializeToString())
with open(predict_net_path, "wb") as fopen:
    fopen.write(predict_net.SerializeToString())


######################################################################
# Some standard imports
from caffe2.proto import caffe2_pb2
from caffe2.python import core, net_drawer, net_printer, visualize, workspace, utils

import numpy as np
import os
import subprocess
from PIL import Image
from matplotlib import pyplot
from skimage import io, transform

# 加载图像
img_in = io.imread("/home/ubuntu/project/pytorch-learning-tutorials/Classification/dataset/images/test_image/animal.jpg")

# 设置图片分辨率为 224x224
img = transform.resize(img_in, [224, 224])

# Let's run the mobile nets that we generated above so that caffe2 workspace is properly initialized
workspace.RunNetOnce(init_net)
workspace.RunNetOnce(predict_net)

# Caffe2 has a nice net_printer to be able to inspect what the net looks like and identify
# what our input and output blob names are.
print(net_printer.to_string(predict_net))

# Now, let's also pass in the resized cat image for processing by the model.
workspace.FeedBlob("53", np.array(img)[np.newaxis, np.newaxis, :, :].astype(np.float32))

# run the predict_net to get the model output
workspace.RunNetOnce(predict_net)

# Now let's get the model output blob
img_out = workspace.FetchBlob("127")
print("img_out:{}".format(img_out))