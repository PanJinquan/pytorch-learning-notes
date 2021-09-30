# -*-coding: utf-8 -*-
"""
    @Project: pytorch-learning-tutorials
    @File   : train_mobileNet_V2.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-04-09 13:46:35
"""

import torch
import torch.optim as optim
from torch import  nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from utils import dataset
import core.mobileNet_V2 as nets
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device='cpu'
print("-----device:{}".format(device))
print("-----Pytorch version:{}".format(torch.__version__))


class Regularization(torch.nn.Module):
    def __init__(self,model,weight_decay,p=2):
        '''
        :param model 模型
        :param weight_decay:正则化参数
        :param p: 范数计算中的幂指数值，默认求2范数,
                  当p=0为L2正则化,p=1为L1正则化
        '''
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model=model
        self.weight_decay=weight_decay
        self.p=p
        self.weight_list=self.get_weight(model)
        self.weight_info(self.weight_list)

    def to(self,device):
        '''
        指定运行模式
        :param device: cude or cpu
        :return:
        '''
        self.device=device
        super().to(device)
        return self

    def forward(self, model):
        self.weight_list=self.get_weight(model)#获得最新的权重
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss

    def get_weight(self,model):
        '''
        获得模型的权重列表
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self,weight_list, weight_decay, p=2):
        '''
        计算张量范数
        :param weight_list:
        :param p: 范数计算中的幂指数值，默认求2范数
        :param weight_decay:
        :return:
        '''
        # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
        # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
        # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
        # reg_loss=torch.FloatTensor([0.]).to(self.device)
        reg_loss=0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg

        reg_loss=weight_decay*reg_loss
        return reg_loss

    def weight_info(self,weight_list):
        '''
        打印权重列表信息
        :param weight_list:
        :return:
        '''
        print("---------------regularization weight---------------")
        for name ,w in weight_list:
            print(name)
        print("---------------------------------------------------")


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



def net_train(train_filename, train_image_dir, test_filename, test_image_dir, num_classes,inputs_shape, epoch_nums):
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
    val_interval=200   #测试间隔
    save_interval=1000 #保存模型间隔
    learning_rate=0.0001
    weight_decay=0.0    #正则化参数
    batch_size, image_channel, resize_height, resize_width=inputs_shape
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=(resize_height, resize_width), scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(int(resize_height / 0.875)),
        transforms.CenterCrop(resize_height),
        transforms.ToTensor(),
        normalize,  # 吧shape=(H,W,C)->换成shape=(C,H,W),并且归一化到[0.0, 1.0]的torch.FloatTensor类型
    ])

    train_data = dataset.TorchDataset(filename=train_filename, image_dir=train_image_dir,
                                      resize_height=resize_height, resize_width=resize_width,repeat=1,transform=train_transform)
    test_data = dataset.TorchDataset(filename=test_filename, image_dir=test_image_dir,
                                     resize_height=resize_height, resize_width=resize_width,repeat=1,transform=val_transform)
    train_loader = dataset.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,num_workers=4)
    val_loader = dataset.DataLoader(dataset=test_data, batch_size=batch_size,shuffle=False)

    train_data_nums=len(train_data)
    test_data_nums=len(test_data)
    print("train_data_nums:{}".format(train_data_nums))
    print("test_data_nums :{}".format(test_data_nums))
    max_iterate=int((train_data_nums+batch_size-1) / batch_size * epoch_nums) #总迭代次数

    # model = resnet.ResNet18(num_classes=num_classes).to(device)
    model = nets.MobileNetV2(num_classes=num_classes).to(device)
    # model = resNetBatchNorm.nets(num_classes=num_classes).to(device)
    print("model,info:\n{}".format(model))


    # weight_list, bias_list = get_weight(model)
    if weight_decay>0:
        reg_loss=Regularization(model, weight_decay, p=2).to(device)
    else:
        print("no regularization")
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)

    '''
    在pytorch中若模型使用CrossEntropyLoss这个loss函数，则不应该在最后一层再使用softmax进行激活。
    '''
    criterion= nn.CrossEntropyLoss().to(device) # CrossEntropyLoss=softmax+cross entropy

    for epoch in range(epoch_nums):
        for step,(batch_x, batch_y) in enumerate(train_loader):
            model.train()
            batch_y =batch_y.squeeze(1)
            # batch_x = Variable(batch_image)
            # batch_y = Variable(batch_label).long()
            # print("batch_image.shape:{},batch_label:{}".format(batch_x.shape,batch_y))

            batch_x = batch_x.to(device)
            # batch_y = batch_y.to(device)
            batch_y = batch_y.to(device=device, dtype=torch.int64)

            out = model(batch_x)

            # loss and regularization
            loss = criterion(input=out, target=batch_y)
            if weight_decay > 0:
                loss = loss + reg_loss(model)
            train_loss = loss.item()

            # accuracy
            pred = torch.max(out, 1)[1]
            train_acc = (pred == batch_y).sum().item()/np.array(pred.size())

            # backprop
            optimizer.zero_grad()#清除当前所有的累积梯度
            loss.backward()
            optimizer.step()

            # print log info
            if step % train_log==0:
                print('step/epoch:{}/{},Train Loss: {:.6f}, Acc: {}'.format(step,epoch,train_loss, train_acc))

            # val测试(测试全部val数据)
            if step % val_interval == 0:
                # val_step(model, val_loader, criterion, test_data_nums)
                pass

            if step % save_interval==0:
                # 保存和加载整个模型
                # torch.save(model, 'models/model.pkl')
                torch.save(model.state_dict(), "models/model_epoch{}_step{}.model".format(epoch,step))
                # model = torch.load('model.pkl')#加载模型方法
if __name__=='__main__':
    train_filename="./dataset/images/train.txt"
    test_filename="./dataset/images/val.txt"
    train_image_dir='./dataset/images/train'
    test_image_dir='./dataset/images/val'
    epoch_nums=100   #总样本循环次数
    batch_size=8   #训练时的一组数据的大小
    image_channel=3
    resize_height=224
    resize_width=224
    num_classes=5

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
    inputs_shape=[batch_size, image_channel, resize_height, resize_width]
    net_train(train_filename,train_image_dir,test_filename,test_image_dir,num_classes,inputs_shape,epoch_nums)