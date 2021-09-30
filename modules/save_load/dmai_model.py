# -*-coding: utf-8 -*-
"""
    @Project: pytorch-learning-tutorials
    @File   : save_load_model.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-06-14 15:53:08
"""
import torch


def convert_dmai_model(model_file, backbone_name, new_model_file):
    '''

    :param model_file:
    :return:
    '''
    num_class = 10
    # 模型加载
    oldModel = torch.load(model_file, map_location=lambda storage, loc: storage.cuda(0))

    # 训练模型保存
    new_model = {}
    new_model["arch"] = "model_name.pth"
    new_model["backbone_name"] = backbone_name
    new_model["input_shape"] = (-1, 3, 112, 112)
    new_model["output_shape"] = (-1, 512)
    # new_model["state_dict"] = oldModel["state_dict"]
    new_model["state_dict"] = oldModel
    torch.save(new_model, new_model_file)


def save_dmai_model(state_dict, model_file, backbone_name, input_size, embedding_size):
    '''
    :param model_file:
    :return:
    '''
    # 训练模型保存
    model = {}
    model["arch"] = "model_name.pth"
    model["backbone_name"] = backbone_name
    model["input_shape"] = (-1, 3, input_size[0], input_size[1])
    model["output_shape"] = (-1, embedding_size)
    model["state_dict"] = state_dict
    torch.save(model, model_file)


def load_dmai_model(model_file):
    model = torch.load(model_file, map_location=lambda storage, loc: storage.cuda(0))
    print(model['arch'])
    print(model['backbone_name'])
    print(model['input_shape'])
    print(model['output_shape'])
    # net.load_state_dict(model["state_dict"])


if __name__ == "__main__":
    model_file = "/media/dm/dm1/project/pytorch-learning-tutorials/modules/save_load/ir_mobilenetv2.pth"
    backbone_name = "IR_MB_V2"
    new_model_file = './IR_MB_V2.pth'
    convert_dmai_model(model_file, backbone_name, new_model_file)
    load_dmai_model(new_model_file)
