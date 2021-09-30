# -*-coding: utf-8 -*-
"""
    @Project: pytorch-learning-tutorials
    @File   : save_load_model.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-06-14 15:53:08
"""
import torch

num_class=10
# 模型加载
model_file = "path/to/your/model.pth"
oldModel = torch.load(model_file, map_location=lambda storage, loc: storage.cuda(0))

# 训练模型保存
new_model = {}
new_model["arch"] = "model_name.pth"
new_model["input_shape"] = (-1, 3, 224, 224)
new_model["output_shape"] = (-1, num_class)
new_model["state_dict"]=oldModel["state_dict"]
torch.save(new_model,model_file)
