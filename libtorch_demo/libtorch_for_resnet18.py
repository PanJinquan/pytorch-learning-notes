# -*-coding: utf-8 -*-
"""
    @Project: pytorch-learning-tutorials
    @File   : resnet18.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-05-28 17:04:31
"""
import torch
# from torchvision.models.mobildnetv2
from torchvision import datasets ,models , transforms
import torchvision

out_model_path="models/model.pt"

model = torchvision.models.resnet18(pretrained=True)
# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224)
# example = torch.ones(1,3,224,224)
model = model.eval()

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)

# The traced ScriptModule can now be evaluated identically to a regular PyTorch module:
output = traced_script_module(torch.ones(1,3,224,224))
traced_script_module.save(out_model_path)
print(output[0, :5])

'''
tensor([-0.0391,  0.1145, -1.7968, -1.2343, -0.8190], grad_fn=<SliceBackward>)
'''