# -*-coding: utf-8 -*-
"""
    @Project: pytorch-learning-tutorials
    @File   : multi_label_example.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-05-22 11:10:47
"""

import torch

batch_size = 2
num_classes = 10

loss_fn = torch.nn.BCELoss()

outputs_before_sigmoid = torch.randn(batch_size, num_classes)
sigmoid_outputs = torch.sigmoid(outputs_before_sigmoid)
target_classes = torch.randint(0, 2, (batch_size, num_classes),dtype=torch.float32)  # randints in [0, 2).

loss = loss_fn(sigmoid_outputs, target_classes)

# alternatively, use BCE with logits, on outputs before sigmoid.
loss_fn_2 = torch.nn.BCEWithLogitsLoss()
loss2 = loss_fn_2(outputs_before_sigmoid, target_classes)


assert loss == loss2