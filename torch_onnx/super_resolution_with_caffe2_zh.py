# -*-coding: utf-8 -*-
"""
    @Project: pytorch-learning-tutorials
    @File   : super_resolution_with_caffe2_zh.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-03-19 09:58:20
    @url    : https://github.com/apachecn/pytorch-doc-zh/blob/master/docs/1.0/super_resolution_with_caffe2.md
"""


# Some standard imports
import io
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx


######################################################################
'''
超分辨率是一种提高图像，视频分辨率的方法，广泛用于图像处理或视频剪辑。
在本教程中，我们将首先使用带有虚拟输入的小型超分辨率模型。
'''
# 首先，让我们在PyTorch中创建一个SuperResolution模型。这个模型 直接来自PyTorch的例子，没有修改
#  PyTorch中定义的Super Resolution模型
import torch.nn as nn
import torch.nn.init as init


class SuperResolutionNet(nn.Module):
    def __init__(self, upscale_factor, inplace=False):
        super(SuperResolutionNet, self).__init__()

        self.relu = nn.ReLU(inplace=inplace)
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)

# 使用上面模型定义，创建super-resolution模型
torch_model = SuperResolutionNet(upscale_factor=3)


######################################################################
'''
通常，你现在会训练这个模型; 但是，对于本教程我们将下载一些预先训练的权重。
请注意，此模型未经过充分训练来获得良好的准确性，此处仅用于演示目的
'''

# 加载预先训练好的模型权重
model_url = 'https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth'
batch_size = 1    # just a random number

# 使用预训练的权重初始化模型
map_location = lambda storage, loc: storage
if torch.cuda.is_available():
    map_location = None
torch_model.load_state_dict(model_zoo.load_url(model_url, map_location=map_location))

#  将训练模式设置为false, since we will only run the forward pass.
torch_model.train(False)


######################################################################
'''
在PyTorch中导出模型通过跟踪工作。要导出模型，请调用torch.onnx._export（）函数。这将执行模型，记录运算符用于计算输出的轨迹。
因为_export运行模型，我们需要提供输入张量x。这个张量的值并不重要; 它可以是图像或随机张量，只要它是正确的大小。
'''

# Input to the model
x = torch.randn(batch_size, 1, 224, 224, requires_grad=True)

# Export the model
torch_out = torch.onnx._export(torch_model,             # model being run
                               x,                       # model input (or a tuple for multiple inputs)
                               "super_resolution.onnx", # where to save the model (can be a file or file-like object)
                               export_params=True)      # store the trained parameter weights inside the model file


######################################################################
'''
torch_out 是执行模型后的输出。通常您可以忽略此输出，但在这里我们将使用它来验证我们导出的模型在Caffe2中运行时计算相同的值。
'''
'''
现在让我们采用ONNX表示并在Caffe2中使用它。这部分通常可以在一个单独的进程中或在另一台机器上完成，
但我们将继续在同一个进程中，以便我们可以验证Caffe2和PyTorch是否为网络计算相同的值：
'''

import onnx
import caffe2.python.onnx.backend as onnx_caffe2_backend

# Load the ONNX ModelProto object. model is a standard Python protobuf object
model = onnx.load("super_resolution.onnx")

# prepare the caffe2 backend for executing the model this converts the ONNX model into a
# Caffe2 NetDef that can execute it. Other ONNX backends, like one for CNTK will be
# availiable soon.
prepared_backend = onnx_caffe2_backend.prepare(model)

# run the model in Caffe2

# Construct a map from input names to Tensor data.
# The graph of the model itself contains inputs for all weight parameters, after the input image.
# Since the weights are already embedded, we just need to pass the input image.
# Set the first input.
W = {model.graph.input[0].name: x.data.numpy()}

# Run the Caffe2 net:
c2_out = prepared_backend.run(W)[0]

# Verify the numerical correctness upto 3 decimal places
# np.testing.assert_almost_equal(torch_out.data.cpu().numpy(), c2_out, decimal=3)
print("Exported model has been executed on Caffe2 backend, and the result looks good!")

######################################################################
# We should see that the output of PyTorch and Caffe2 runs match
# numerically up to 3 decimal places. As a side-note, if they do not match
# then there is an issue that the operators in Caffe2 and PyTorch are
# implemented differently and please contact us in that case.
#


######################################################################
# Transfering SRResNet using ONNX
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


######################################################################
# Using the same process as above, we also transferred an interesting new
# model "SRResNet" for super-resolution presented in `this
# paper <https://arxiv.org/pdf/1609.04802.pdf>`__ (thanks to the authors
# at Twitter for providing us code and pretrained parameters for the
# purpose of this tutorial). The model definition and a pre-trained model
# can be found
# `here <https://gist.github.com/prigoyal/b245776903efbac00ee89699e001c9bd>`__.
# Below is what SRResNet model input, output looks like. |SRResNet|
#
# .. |SRResNet| image:: /_static/img/SRResNet.png
#


######################################################################
# Running the model on mobile devices
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


######################################################################
# So far we have exported a model from PyTorch and shown how to load it
# and run it in Caffe2. Now that the model is loaded in Caffe2, we can
# convert it into a format suitable for `running on mobile
# devices <https://caffe2.ai/docs/mobile-integration.html>`__.
#
# We will use Caffe2's
# `mobile\_exporter <https://github.com/caffe2/caffe2/blob/master/caffe2/python/predictor/mobile_exporter.py>`__
# to generate the two model protobufs that can run on mobile. The first is
# used to initialize the network with the correct weights, and the second
# actual runs executes the model. We will continue to use the small
# super-resolution model for the rest of this tutorial.
#

# extract the workspace and the model proto from the internal representation
c2_workspace = prepared_backend.workspace
c2_model = prepared_backend.predict_net

# Now import the caffe2 mobile exporter
from caffe2.python.predictor import mobile_exporter

# call the Export to get the predict_net, init_net. These nets are needed for running things on mobile
init_net, predict_net = mobile_exporter.Export(c2_workspace, c2_model, c2_model.external_input)

# 我们还将init_net和predict_net保存到我们稍后将用于在移动设备上运行它们的文件中
with open('init_net.pb', "wb") as fopen:
    fopen.write(init_net.SerializeToString())
with open('predict_net.pb', "wb") as fopen:
    fopen.write(predict_net.SerializeToString())


######################################################################
'''
init_net具有模型参数和嵌入在其中的模型输入，predict_net将用于指导运行时的init_net执行。
 在本教程中，我们将使用上面生成的init_net和predict_net，并在正常的Caffe2后端和移动设备中运行它们，
 并验证两次运行中生成的输出高分辨率猫咪图像是否相同。
'''
#
# For this tutorial, we will use a famous cat image used widely which
# looks like below
#
# .. figure:: /_static/img/cat_224x224.jpg
#    :alt: cat
#

# Some standard imports
from caffe2.proto import caffe2_pb2
from caffe2.python import core, net_drawer, net_printer, visualize, workspace, utils

import numpy as np
import os
import subprocess
from PIL import Image
from matplotlib import pyplot
from skimage import io, transform


######################################################################
# First, let's load the image, pre-process it using standard skimage
# python library. Note that this preprocessing is the standard practice of
# processing data for training/testing neural networks.
#

# load the image
img_in = io.imread("./image/cat.jpg")

# resize the image to dimensions 224x224
img = transform.resize(img_in, [224, 224])

# save this resized image to be used as input to the model
io.imsave("./image/cat_224x224.jpg", img)


######################################################################
# Now, as a next step, let's take the resized cat image and run the
# super-resolution model in Caffe2 backend and save the output image. The
# image processing steps below have been adopted from PyTorch
# implementation of super-resolution model
# `here <https://github.com/pytorch/examples/blob/master/super_resolution/super_resolve.py>`__
#

# load the resized image and convert it to Ybr format
img = Image.open("./image/cat_224x224.jpg")
img_ycbcr = img.convert('YCbCr')
img_y, img_cb, img_cr = img_ycbcr.split()

# Let's run the mobile nets that we generated above so that caffe2 workspace is properly initialized
workspace.RunNetOnce(init_net)
workspace.RunNetOnce(predict_net)

# Caffe2 has a nice net_printer to be able to inspect what the net looks like and identify
# what our input and output blob names are.
print(net_printer.to_string(predict_net))


######################################################################
# From the above output, we can see that input is named "9" and output is
# named "27"(it is a little bit weird that we will have numbers as blob
# names but this is because the tracing JIT produces numbered entries for
# the models)
#

# Now, let's also pass in the resized cat image for processing by the model.
workspace.FeedBlob("9", np.array(img_y)[np.newaxis, np.newaxis, :, :].astype(np.float32))

# run the predict_net to get the model output
workspace.RunNetOnce(predict_net)

# Now let's get the model output blob
# img_out = workspace.FetchBlob("27")
img_out = workspace.FetchBlob("20")


######################################################################
# Now, we'll refer back to the post-processing steps in PyTorch
# implementation of super-resolution model
# `here <https://github.com/pytorch/examples/blob/master/super_resolution/super_resolve.py>`__
# to construct back the final output image and save the image.
#

img_out_y = Image.fromarray(np.uint8((img_out[0, 0]).clip(0, 255)), mode='L')

# get the output image follow post-processing step from PyTorch implementation
final_img = Image.merge(
    "YCbCr", [
        img_out_y,
        img_cb.resize(img_out_y.size, Image.BICUBIC),
        img_cr.resize(img_out_y.size, Image.BICUBIC),
    ]).convert("RGB")

# Save the image, we will compare this with the output image from mobile device
final_img.save("./image/cat_superres.jpg")


######################################################################
# We have finished running our mobile nets in pure Caffe2 backend and now,
# let's execute the model on an Android device and get the model output.
#
# ``NOTE``: for Android development, ``adb`` shell is needed otherwise the
# following section of tutorial will not run.
#
# In our first step of runnig model on mobile, we will push a native speed
# benchmark binary for mobile device to adb. This binary can execute the
# model on mobile and also export the model output that we can retrieve
# later. The binary is available
# `here <https://github.com/caffe2/caffe2/blob/master/caffe2/binaries/speed_benchmark.cc>`__.
# In order to build the binary, execute the ``build_android.sh`` script
# following the instructions
# `here <https://github.com/caffe2/caffe2/blob/master/scripts/build_android.sh>`__.
#
# ``NOTE``: You need to have ``ANDROID_NDK`` installed and set your env
# variable ``ANDROID_NDK=path to ndk root``
#

# let's first push a bunch of stuff to adb, specify the path for the binary
CAFFE2_MOBILE_BINARY = ('caffe2/binaries/speed_benchmark')

# we had saved our init_net and proto_net in steps above, we use them now.
# Push the binary and the model protos
os.system('adb push ' + CAFFE2_MOBILE_BINARY + ' /data/local/tmp/')
os.system('adb push init_net.pb /data/local/tmp')
os.system('adb push predict_net.pb /data/local/tmp')

# Let's serialize the input image blob to a blob proto and then send it to mobile for execution.
with open("input.blobproto", "wb") as fid:
    fid.write(workspace.SerializeBlob("9"))

# push the input image blob to adb
os.system('adb push input.blobproto /data/local/tmp/')

# Now we run the net on mobile, look at the speed_benchmark --help for what various options mean
os.system(
    'adb shell /data/local/tmp/speed_benchmark '                     # binary to execute
    '--init_net=/data/local/tmp/super_resolution_mobile_init.pb '    # mobile init_net
    '--net=/data/local/tmp/super_resolution_mobile_predict.pb '      # mobile predict_net
    '--input=9 '                                                     # name of our input image blob
    '--input_file=/data/local/tmp/input.blobproto '                  # serialized input image
    '--output_folder=/data/local/tmp '                               # destination folder for saving mobile output
    '--output=27,9 '                                                 # output blobs we are interested in
    '--iter=1 '                                                      # number of net iterations to execute
    '--caffe2_log_level=0 '
)

# get the model output from adb and save to a file
os.system('adb pull /data/local/tmp/27 ./output.blobproto')


# We can recover the output content and post-process the model using same steps as we followed earlier
blob_proto = caffe2_pb2.BlobProto()
blob_proto.ParseFromString(open('./output.blobproto').read())
img_out = utils.Caffe2TensorToNumpyArray(blob_proto.tensor)
img_out_y = Image.fromarray(np.uint8((img_out[0,0]).clip(0, 255)), mode='L')
final_img = Image.merge(
    "YCbCr", [
        img_out_y,
        img_cb.resize(img_out_y.size, Image.BICUBIC),
        img_cr.resize(img_out_y.size, Image.BICUBIC),
    ]).convert("RGB")
final_img.save("./image/cat_superres_mobile.jpg")


######################################################################
# Now, you can compare the image ``cat_superres.jpg`` (model output from
# pure caffe2 backend execution) and ``cat_superres_mobile.jpg`` (model
# output from mobile execution) and see that both the images look same. If
# they don't look same, something went wrong with execution on mobile and
# in that case, please contact Caffe2 community. You should expect to see
# the output image to look like following:
#
# .. figure:: /_static/img/cat_output1.png
#    :alt: output\_cat
#
#
# Using the above steps, you can deploy your models on mobile easily.
# Also, for more information on caffe2 mobile backend, checkout
# `caffe2-android-demo <https://caffe2.ai/docs/AI-Camera-demo-android.html>`__.
#
