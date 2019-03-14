# -*-coding: utf-8 -*-
"""
    @Project: pytorch-learning-tutorials
    @File   : fashion_mnist_cnn.py.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-03-07 16:46:53
    @url: 《pytorch实现L2和L1正则化的方法》https://panjinquan.blog.csdn.net/article/details/88426648
"""

import torch
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from utils import dataset,image_processing
from core import simpleNet
import numpy as np

def net_val(model, val_loader, loss_func, test_data_nums):
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    with torch.no_grad():
        for batch_image, batch_label in val_loader:
            batch_label = batch_label.squeeze(1)
            batch_x = Variable(batch_image)
            batch_y = Variable(batch_label).long()
            out = model(batch_x)
            loss = loss_func(out, batch_y)
            eval_loss += loss.item()
            pred = torch.max(out, 1)[1]
            eval_acc += (pred == batch_y).sum().item()
    print('------Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / test_data_nums, eval_acc / test_data_nums))

def net_train(train_filename, train_image_dir, test_filename, test_image_dir, inputs_shape, epoch_nums):
    '''
    :param train_filename:
    :param train_image_dir:
    :param test_filename:
    :param test_image_dir:
    :param inputs_shape:
    :param epoch_nums:
    :return:
    '''
    train_log=10       #训练log
    val_interval=500   #测试间隔
    save_interval=1000 #保存模型间隔
    learning_rate=1e-3
    batch_size, image_channel, resize_height, resize_width=inputs_shape
    train_data = dataset.TorchDataset(filename=train_filename, image_dir=train_image_dir,
                                      resize_height=resize_height, resize_width=resize_width,repeat=1)
    test_data = dataset.TorchDataset(filename=test_filename, image_dir=test_image_dir,
                                     resize_height=resize_height, resize_width=resize_width,repeat=1)
    train_loader = dataset.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    val_loader = dataset.DataLoader(dataset=test_data, batch_size=batch_size,shuffle=False)

    train_data_nums=len(train_data)
    test_data_nums=len(test_data)
    print("train_data_nums:{}".format(train_data_nums))
    print("test_data_nums :{}".format(test_data_nums))
    max_iterate=int((train_data_nums+batch_size-1) / batch_size * epoch_nums) #总迭代次数

    model = simpleNet.SimpleNet()
    print("model,info:\n{}".format(model))

    # 优化器
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    '''
    在pytorch中若模型使用CrossEntropyLoss这个loss函数，则不应该在最后一层再使用softmax进行激活。
    '''
    loss_func = torch.nn.CrossEntropyLoss() # CrossEntropyLoss=softmax+cross entropy

    for epoch in range(epoch_nums):
        for step,(batch_image, batch_label) in enumerate(train_loader):
            batch_label =batch_label.squeeze(1)
            batch_x = Variable(batch_image)
            batch_y = Variable(batch_label).long()
            out = model(batch_x)

            # 前向传播计算loss and accuracy
            loss = loss_func(out, batch_y)
            train_loss = loss.item()
            pred = torch.max(out, 1)[1]
            train_acc = (pred == batch_y).sum().item()/np.array(pred.size())

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print log info
            if step % train_log==0:
                print('step/epoch:{}/{},Train Loss: {:.6f}, Acc: {}'.format(step,epoch,train_loss, train_acc))

            # val测试(测试全部val数据)
            if step % val_interval == 0:
                net_val(model, val_loader, loss_func, test_data_nums)

            if step % save_interval==0:
                # 保存和加载整个模型
                # torch.save(model, 'models/model.pkl')
                torch.save(model.state_dict(), "model_epoch{}_step{}.model".format(epoch,step))
                # model = torch.load('model.pkl')#加载模型方法

if __name__=='__main__':
    train_filename="./dataset/fashion_mnist/train.txt"
    test_filename="./dataset/fashion_mnist/test.txt"
    train_image_dir='./dataset/fashion_mnist'
    test_image_dir='./dataset/fashion_mnist'
    epoch_nums=10   #总样本循环次数
    batch_size=32   #训练时的一组数据的大小
    image_channel=3
    resize_height=28
    resize_width=28
    inputs_shape=[batch_size, image_channel, resize_height, resize_width]
    net_train(train_filename,train_image_dir,test_filename,test_image_dir,inputs_shape,epoch_nums)
