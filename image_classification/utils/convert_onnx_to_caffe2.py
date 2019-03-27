# -*-coding: utf-8 -*-
"""
    @Project: pytorch-learning-tutorials
    @File   : convert_onnx_to_caffe2.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-03-21 10:28:45
"""

from torch.autograd import Variable
import onnx
import caffe2.python.onnx.backend as onnx_caffe2_backend
from caffe2.python.predictor import mobile_exporter
# from onnx_caffe2.backend import Caffe2Backend
from caffe2.python.onnx.backend import Caffe2Backend
import onnx.backend
import torch

def convert_onnx_to_caffe2_v1(onnx_path, init_net_path, predict_net_path):
    '''
    :param onnx_path:
    :param init_net_path:
    :param predict_net_path:
    :return:
    '''
    batch_size=1
    x = Variable(torch.randn(batch_size,3,224,224), requires_grad=True)
    # import onnx_caffe2.backend
    # Load the ONNX ModelProto object. model is a standard Python protobuf object
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)

    # prepare the caffe2 backend for executing the model this converts the ONNX model into a
    # Caffe2 NetDef that can execute it. Other ONNX backends, like one for CNTK will be
    # availiable soon.
    prepared_backend = onnx_caffe2_backend.prepare(model)

    # Construct a map from input names to Tensor data.
    # The graph of the model itself contains inputs for all weight parameters, after the input image.
    # Since the weights are already embedded, we just need to pass the input image.
    # Set the first input.
    # name=model.graph.input[0].name
    W = {model.graph.input[0].name: x.data.numpy()}

    # Run the Caffe2 net:
    c2_out = prepared_backend.run(W)

    # extract the workspace and the model proto from the internal representation
    c2_workspace = prepared_backend.workspace
    c2_model = prepared_backend.predict_net

    # call the Export to get the predict_net, init_net. These nets are needed for running things on mobile
    init_net, predict_net = mobile_exporter.Export(c2_workspace, c2_model, c2_model.external_input)

    # Let's also save the init_net and predict_net to a file that we will later use for running them on mobile
    with open(init_net_path, "wb") as fopen:
        fopen.write(init_net.SerializeToString())
    with open(predict_net_path, "wb") as fopen:
        fopen.write(predict_net.SerializeToString())

def convert_onnx_to_caffe2_v2(onnx_path,init_net_path,predict_net_path):
    '''
    等效命令行: convert-onnx-to-caffe2 path/to/model.onnx --output predict_net.pb --init-net-output init_net.pb
    :param onnx_path
    :param init_net_path:
    :param predict_net_path:
    :return:
    '''
    model = onnx.load(onnx_path)
    # onnx.checker.check_model(model)
    init_net, predict_net = Caffe2Backend.onnx_graph_to_caffe2_net(model)
    with open(init_net_path, "wb") as f:
        f.write(init_net.SerializeToString())
    with open(predict_net_path, "wb") as f:
        f.write(predict_net.SerializeToString())