# -*-coding: utf-8 -*-
"""
    @Project: pytorch-learning-tutorials
    @File   : convert_to_trace_model.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-05-28 18:16:50
"""
import torch
import torchvision

from core import resnet,resNetBatchNorm,resRegularBn



def convert_to_trace_model(model,out_model_path,device):
    example = torch.rand(1, 3, 224, 224).to(device)
    test_input=torch.ones(1, 3, 224, 224).to(device)
    # example = torch.ones(1,3,224,224)
    model = model.eval()

    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(model, example)

    # The traced ScriptModule can now be evaluated identically to a regular PyTorch module:
    output = traced_script_module(test_input)
    traced_script_module.save(out_model_path)
    print(output[0, :5])

if __name__=="__main__":
    '''
    tensor([-0.3027,  3.2513,  6.7108,  4.4508, -4.4523], grad_fn=<SelectBackward>)
    '''
    device = "cpu"
    num_classes=5
    model_path = "../models/model_epoch0_step200.pth"
    out_model_path="../models/trace_model/trace_model.pth"
    model = resRegularBn.nets(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path))

    convert_to_trace_model(model, out_model_path,device)

