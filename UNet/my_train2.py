import sys
import os
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from unet import Loss

import eval
from unet import UNet,unet_dilated
from utils import image_processing,dataset,PairRandomCrop


# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device='cpu'
print("-----device:{}".format(device))
print("-----Pytorch version:{}".format(torch.__version__))

def val_step(model, val_loader, criterion, test_data_nums,device):
    # model.eval()
    # eval_loss = 0.
    # with torch.no_grad():
    #     for batch_x, batch_y in val_loader:
    #         batch_y = batch_y.squeeze(1)
    #
    #         batch_x = batch_x.to(device)
    #         batch_y = batch_y.to(device)
    #         mask_pred = model(batch_x)
    #         loss = criterion(mask_pred, batch_y)
    #         eval_loss += loss.item()
    #         mask_pred = (mask_pred > 0.5).float()
    #         tot += dice_coeff(mask_pred, true_mask).item()
    # print('------Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / test_data_nums, eval_acc / test_data_nums))
    pass

def train_net(orig_dir,masks_dir,train_filename):
    epochs=10000
    batch_size=32
    lr=0.01
    resize_height = 50
    resize_width = 50
    train_log = 10  # 训练log
    finetune=False
    pretrained_model="./checkpoints/model_epoch3800_step113.pth"
    save_model_path="checkpoints"
    val_interval = 200  # 测试间隔
    save_interval = 1000  # 保存模型间隔


    train_transform = transforms.Compose([
        transforms.Resize(size=(resize_height, resize_width)),
        # transforms.RandomHorizontalFlip(),#随机翻转图像
        # transforms.RandomCrop(size=(resize_height, resize_width), padding=4),      # 随机裁剪
        PairRandomCrop.PairRandomCrop(size=(resize_height, resize_width), padding=4),#配对裁剪
        transforms.ToTensor(),  # 吧shape=(H,W,C)->换成shape=(C,H,W),并且归一化到[0.0, 1.0]的torch.FloatTensor类型
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))#给定均值(R,G,B) 方差（R，G，B），将会把Tensor正则化
    ])

    train_data = dataset.TorchDataset(filename=train_filename,
                              orig_dir=orig_dir,
                              dest_dir=masks_dir,
                              resize_height=resize_height,
                              resize_width=resize_width,
                              repeat=1,
                              transform=train_transform)
    # test_data = TorchDataset(test_filename, image_dir,resize_height,resize_width,1,transform=None)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size,shuffle=True)

    # net = UNet(n_channels=3, n_classes=1).to(device)
    net = unet_dilated.UNet(n_channels=3, n_classes=1).to(device)

    if finetune and os.path.exists(pretrained_model):
        net.load_state_dict(torch.load(pretrained_model))
        print("load pretrained model :{}".format(pretrained_model))
    else:
        print("no finetune or pretrained model")
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    # optimizer = optim.SGD(net.parameters(),lr=lr,momentum=0.9,weight_decay=0.0005)
    # optimizer = optim.adagrad(net.parameters(),lr=lr)
    optimizer = optim.Adadelta(net.parameters(),lr=lr)

    criterion = nn.BCELoss().to(device)
    # criterion=Loss.FocalLoss2d().to(device)
    # criterion=nn.BCEWithLogitsLoss().to(device)
    for epoch in range(epochs):
        for step,(batch_orig_image, batch_mask_image) in enumerate(train_loader):
            net.train()
            # show_batch_image("batch_mask_image",batch_mask_image)
            # show_batch_image("batch_orig_image",batch_orig_image)
            batch_orig_image = batch_orig_image.to(device)
            batch_mask_image = batch_mask_image.to(device)
                
            pred_masks = net(batch_orig_image)

            # show_batch_image("masks_pred",masks_pred.detach().cpu().numpy())

            pred_masks_flat = pred_masks.view(-1)
            true_masks_flat = batch_mask_image.view(-1)
            # 计算loss
            loss = criterion(input=pred_masks_flat, target=true_masks_flat)
            # loss = criterion(input=pred_masks, target=batch_mask_image)

            # 计算平均IOU
            iou_mean=eval.get_batch_iou_mean(pred_masks.detach().cpu().numpy(),batch_mask_image.cpu().numpy())
            # 计算AP值
            AP=eval.get_batch_AP(pred_masks.detach().cpu().numpy(),batch_mask_image.cpu().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print log info
            if step % train_log == 0:
                print('step/epoch:{}/{},Train Loss: {:.6f},iou_mean:{:.6f},AP:{:.6f}'.format(epoch,step, loss,iou_mean,AP))

        # val测试(测试全部val数据)
        if epoch % 1 == 0:
            # val_step(model, val_loader, criterion, test_data_nums)
            pass

        if (epoch+1) % 100 == 0:
            # 保存和加载整个模型
            # torch.save(model, 'checkpoints/model.pkl')
            save_model=os.path.join(save_model_path,"model_epoch{}_step{}.pth".format(epoch+1, step+1))
            torch.save(net.state_dict(), save_model)
            # model = torch.load('model.pkl')#加载模型方法




if __name__ == '__main__':
    train_filename="E:/git/dataset/tgs-salt-identification-challenge/train/train.txt"
    orig_dir= 'E:/git/dataset/tgs-salt-identification-challenge/train/images'
    masks_dir= 'E:/git/dataset/tgs-salt-identification-challenge/train/masks'
    train_net(orig_dir,masks_dir,train_filename)

