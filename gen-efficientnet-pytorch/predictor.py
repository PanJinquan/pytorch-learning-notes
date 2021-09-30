# -*-coding: utf-8 -*-
"""
    @Project: pytorch-learning-tutorials
    @File   : predictor.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-07-06 12:56:43
"""
from model_factory import create_model
import torch
from utils import debug
def predictor():
    pass



if __name__=="__main__":
    model_name="efficientnet_b2" # enum={mobilenetv3_100,mnasnet_100,efficientnet_b2}
    num_classes=1000
    pretrained=True
    checkpoint=None
    model = create_model(
        model_name,
        num_classes=num_classes,
        in_chans=3,
        pretrained=pretrained,
        checkpoint_path=checkpoint).cuda()
    print(model)
    model.eval()
    x =torch.zeros(1,3,640,640).cuda()
    output = model(x)
    output = model(x)

    torch.cuda.synchronize()
    T0=debug.TIME()
    output = model(x)
    torch.cuda.synchronize()
    T1=debug.TIME()
    print("run time:{}ms".format(debug.RUN_TIME(T1-T0)))
    # print(output)
