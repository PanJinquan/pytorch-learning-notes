# -*-coding: utf-8 -*-
"""
    @Project: pytorch-learning-tutorials
    @File   : save_load_model.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-06-14 15:53:08
"""
import torch

num_class = 10

data_file = "./model_data.pth"
data = ["A", "B", "C"]

# 保存数据
data_dict= {}
data_dict["data"] = data
data_dict["input_shape"] = (-1, 3, 224, 224)
data_dict["output_shape"] = (-1, num_class)
# data_dict["state_dict"] ="你训练的模型model"
print("data_dict:{}".format(data_dict))
torch.save(data_dict, data_file)

# 加载模型数据
# model_data = torch.load(data_file, map_location=lambda storage, loc: storage.cuda(0)) #cuda
model_data = torch.load(data_file) #cpu
print("model_data:{}".format(model_data))
