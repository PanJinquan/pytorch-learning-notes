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
from torch import nn
from torchvision import transforms
from utils import dataset
import numpy as np
import os
from torchvision import models

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 检查GPU是否可用
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'
print("-----device:{}".format(device))
print("-----Pytorch version:{}".format(torch.__version__))


def val_step(model, val_loader, criterion, test_data_nums):
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_y = batch_y.squeeze(1)

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            out = model(batch_x)
            loss = criterion(out, batch_y)
            eval_loss += loss.item()
            pred = torch.max(out, 1)[1]
            eval_acc += (pred == batch_y).sum().item()
    print('------Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / test_data_nums, eval_acc / test_data_nums))


def net_train(train_filename, train_image_dir, test_filename, test_image_dir, num_classes, inputs_shape, epoch_nums):
    '''
    《pytorch之迁移学习》https://blog.csdn.net/weixin_40123108/article/details/85238355
    :param train_filename:
    :param train_image_dir:
    :param test_filename:
    :param test_image_dir:
    :param inputs_shape:
    :param epoch_nums:
    :return:
    '''
    # pretrained_model = "models/pretrain/resnet18-class5.pth"
    pretrained_model = "models/pretrain/resnet18-class1000.pth"

    train_log = 10  # 训练log
    val_interval = 200  # 测试间隔
    save_interval = 100  # 保存模型间隔
    learning_rate = 0.0001
    batch_size, image_channel, resize_height, resize_width = inputs_shape

    train_transform = transforms.Compose([
        transforms.Resize(size=(resize_height, resize_width)),
        transforms.RandomHorizontalFlip(),  # 随机翻转图像
        transforms.RandomCrop(size=(resize_height, resize_width), padding=4),  # 随机裁剪
        transforms.ToTensor(),  # 吧shape=(H,W,C)->换成shape=(C,H,W),并且归一化到[0.0, 1.0]的torch.FloatTensor类型
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))#给定均值(R,G,B) 方差（R，G，B），将会把Tensor正则化
    ])

    val_transform = transforms.Compose([
        transforms.Resize(size=(resize_height, resize_width)),
        transforms.ToTensor(),  # 吧shape=(H,W,C)->换成shape=(C,H,W),并且归一化到[0.0, 1.0]的torch.FloatTensor类型
    ])

    train_data = dataset.TorchDataset(filename=train_filename,
                                      image_dir=train_image_dir,
                                      resize_height=resize_height,
                                      resize_width=resize_width,
                                      repeat=1,
                                      transform=train_transform)
    test_data = dataset.TorchDataset(filename=test_filename,
                                     image_dir=test_image_dir,
                                     resize_height=resize_height,
                                     resize_width=resize_width,
                                     repeat=1,
                                     transform=val_transform)
    train_loader = dataset.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    val_loader = dataset.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    train_data_nums = len(train_data)
    test_data_nums = len(test_data)
    print("train_data_nums:{}".format(train_data_nums))
    print("test_data_nums :{}".format(test_data_nums))

    # 假设我们已经存在一个1000分类的resnet18的模型：resnet18-class1000.pth
    model = models.resnet18(pretrained=False).to(device)
    checkpoint = torch.load(pretrained_model)
    # state_dict = checkpoint.get("state_dict")
    model.load_state_dict(checkpoint)
    # 打印网络结构：(fc): Linear(in_features=512, out_features=1000, bias=True)
    print("resnet18,info:\n{}".format(model))
    num_features = model.fc.in_features
    '''
    仅改变resnet18最后一层的全连接层：model.fc = nn.Linear(num_features, num_classes)
    1.若想训练时更新全部模型参数，则requires_grad_设置为True(默认是True)
    2.若只想训练最后一层的，将其它层的参数requires_grad设置为 False
    '''
    for param in model.parameters():
        param.requires_grad_(False)
    model.fc = nn.Linear(num_features, num_classes)
    # model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    print("new resnet18,info:\n{}".format(model))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    '''
    在pytorch中若模型使用CrossEntropyLoss这个loss函数，则不应该在最后一层再使用softmax进行激活。
    '''
    criterion = nn.CrossEntropyLoss().to(device)  # CrossEntropyLoss=softmax+cross entropy

    for epoch in range(epoch_nums):
        for step, (batch_x, batch_y) in enumerate(train_loader):
            model.train()
            batch_y = batch_y.squeeze(1)
            batch_x = batch_x.to(device)
            # batch_y = batch_y.to(device)
            batch_y = batch_y.to(device=device, dtype=torch.int64)

            out = model(batch_x)

            # loss and regularization
            loss = criterion(input=out, target=batch_y)
            train_loss = loss.item()

            # accuracy
            pred = torch.max(out, 1)[1]
            train_acc = (pred == batch_y).sum().item() / np.array(pred.size())

            # backprop
            optimizer.zero_grad()  # 清除当前所有的累积梯度
            loss.backward()
            optimizer.step()

            # print log info
            if step % train_log == 0:
                print('step/epoch:{}/{},Train Loss: {:.6f}, Acc: {}'.format(step, epoch, train_loss, train_acc))

            # val测试(测试全部val数据)
            if step % val_interval == 0:
                # val_step(model, val_loader, criterion, test_data_nums)
                pass

            if step % save_interval == 0:
                model_path = "models/model_epoch{}_step{}.pth".format(epoch, step)
                # 保存和加载整个模型
                # torch.save(model, "m)
                # torch.save(model.state_dict(), "models/model_epoch{}_step{}.pth".format(epoch,step))
                # 保存
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss, }, model_path)


if __name__ == '__main__':
    train_filename = "./dataset/images/train.txt"
    test_filename = "./dataset/images/val.txt"
    train_image_dir = './dataset/images/train'
    test_image_dir = './dataset/images/val'
    epoch_nums = 100  # 总样本循环次数
    batch_size = 8  # 训练时的一组数据的大小
    image_channel = 3
    resize_height = 224
    resize_width = 224
    num_classes = 5

    # train_filename="./dataset/fashion_mnist/train.txt"
    # test_filename="./dataset/fashion_mnist/test.txt"
    # train_image_dir='./dataset/fashion_mnist'
    # test_image_dir='./dataset/fashion_mnist'
    # epoch_nums=10   #总样本循环次数
    # batch_size=32   #训练时的一组数据的大小
    # image_channel=3
    # resize_height=224
    # resize_width=224
    # num_classes=10
    inputs_shape = [batch_size, image_channel, resize_height, resize_width]
    net_train(train_filename, train_image_dir, test_filename, test_image_dir, num_classes, inputs_shape, epoch_nums)
