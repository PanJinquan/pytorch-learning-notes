# -*-coding: utf-8 -*-
"""
    @Project: pytorch-learning-tutorials
    @File   : predict.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-03-11 10:44:16
"""
'''
盐矿区域分割，参赛者需要训练用于从岩层中分离盐矿的机器学习模型。

'''
import torch
import torch.optim as optim
from torch import  nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from utils import dataset,image_processing,file_processing
from PIL import Image
import os,glob
import numpy as np
from unet import UNet
from unet import UNet,unet_dilated
from utils import resize_and_crop, normalize, split_img_into_squares, hwc_to_chw, merge_masks, dense_crf
from utils import plot_img_and_mask

from torchvision import transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device='cpu'
print("-----device:{}".format(device))
print("-----Pytorch version:{}".format(torch.__version__))


def get_rle_encode(im,order="F"):
    '''
    mask编码
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order = order)    # 从上到下，从左到右进行排列-> F
    # pixels = im.flatten()             # 从左到右，从上到下进行排列-> C(默认值)
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    # return ' '.join(str(x) for x in runs)
    return runs


def get_rle_decode(rle_mask,order="F"):
    '''
    mask解码
    rle_mask: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    # s = rle_mask.split()
    s = rle_mask
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(101*101, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(101,101,order=order)

def save_submit(images_list,mask_code_list,out_csv_path):
    '''
    Used for Kaggle submission: predicts and encode all test images
    :param images_list:
    :param mask_code_list:
    :param out_csv_path:
    :return:
    '''
    with open(out_csv_path, 'w') as f:
        f.write('id,rle_mask\n')
        for images_path, mask_code in zip(images_list,mask_code_list):
            base_name=os.path.basename(images_path)
            base_name=base_name[:-len(".png")]
            f.write('{},{}\n'.format(base_name, ' '.join(map(str, mask_code))))
    print("save file successfully:{}".format(out_csv_path))


def predict_img(net,
                full_img,
                scale_factor=0.5,
                out_threshold=0.5,
                use_dense_crf=True,
                use_gpu=False):
    net.eval()
    img_height = full_img.size[1]
    img_width = full_img.size[0]

    img = resize_and_crop(full_img, scale=scale_factor)
    img = normalize(img)

    left_square, right_square = split_img_into_squares(img)

    left_square = hwc_to_chw(left_square)
    right_square = hwc_to_chw(right_square)

    X_left = torch.from_numpy(left_square).unsqueeze(0)
    X_right = torch.from_numpy(right_square).unsqueeze(0)

    if use_gpu:
        X_left = X_left.cuda()
        X_right = X_right.cuda()

    with torch.no_grad():
        output_left = net(X_left)
        output_right = net(X_right)

        left_probs = output_left.squeeze(0)
        right_probs = output_right.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(img_height),
                transforms.ToTensor()
            ]
        )

        left_probs = tf(left_probs.cpu())
        right_probs = tf(right_probs.cpu())

        left_mask_np = left_probs.squeeze().cpu().numpy()
        right_mask_np = right_probs.squeeze().cpu().numpy()

    full_mask = merge_masks(left_mask_np, right_mask_np, img_width)

    if use_dense_crf:
        full_mask = dense_crf(np.array(full_img).astype(np.uint8), full_mask)

    return full_mask > out_threshold


def predict_crf(model,model_path,image_dir,out_csv_path):
    resize_height = 50
    resize_width = 50
    threshold = 0.5
    isShow = False
    # model = UNet(n_channels=3, n_classes=1).to(device)

    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    # image = Image.open(image_path)
    # test_transform = transforms.Compose([
    #     transforms.Resize(size=(resize_height, resize_width)),
    #     transforms.ToTensor(),  # 吧shape=(H,W,C)->换成shape=(C,H,W),并且归一化到[0.0, 1.0]的torch.FloatTensor类型
    # ])
    test_transform = transforms.Compose([
        transforms.Resize(size=(resize_height, resize_width)),
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    normalize_transform = transforms.Compose([
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 给定均值(R,G,B) 方差（R，G，B），将会把Tensor正则化
    ])

    images_list = file_processing.get_images_list(image_dir, ['*.jpg', '*.png'])
    mask_code_list = []
    for image_path in images_list:
        print("processing image:{}".format(image_path))
        src_image = Image.open(image_path).convert('RGB')
        src_image=normalize_transform(src_image)
        re_image=src_image.resize((resize_width, resize_height))
        # re_image=src_image
        src_image=np.asarray(src_image)
        pre_mask = predict_img(net=model,
                           full_img=re_image,
                           scale_factor=1.0,
                           out_threshold=0.5,
                           use_dense_crf=True,
                           use_gpu=True)
        pre_mask=np.asarray(pre_mask+0,dtype=np.float32)
        pre_mask=pre_mask.transpose(1,0)#转换通道顺序
        pre_mask = image_processing.resize_image(pre_mask, 101, 101)
        pre_mask = np.where(pre_mask > threshold, 1, 0)
        mask_code = get_rle_encode(pre_mask)  # 编码
        mask_code_list.append(mask_code)
        if isShow:
            mask_image = get_rle_decode(mask_code)  # 解码显示
            image_processing.show_image("src", src_image)
            image_processing.show_image("mask", pre_mask)
            # image_processing.show_image("mask_image", mask_image)

    save_submit(images_list, mask_code_list, out_csv_path)

def predict(model,model_path,image_dir,out_csv_path):
    resize_height=50
    resize_width=50
    threshold=0.5
    isShow=False
    # model = UNet(n_channels=3, n_classes=1).to(device)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    # image = Image.open(image_path)
    # test_transform = transforms.Compose([
    #     transforms.Resize(size=(resize_height, resize_width)),
    #     transforms.ToTensor(),  # 吧shape=(H,W,C)->换成shape=(C,H,W),并且归一化到[0.0, 1.0]的torch.FloatTensor类型
    # ])
    test_transform = transforms.Compose([
        transforms.Resize(size=(resize_height, resize_width)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 给定均值(R,G,B) 方差（R，G，B），将会把Tensor正则化
    ])

    images_list=file_processing.get_images_list(image_dir,['*.jpg','*.png'])
    mask_code_list=[]
    for image_path in images_list:
        print("processing image:{}".format(image_path))
        image = Image.open(image_path).convert('RGB')
        image_tensor = test_transform(image).float()
        # Add an extra batch dimension since pytorch treats all images as batches
        image_tensor = image_tensor.unsqueeze_(0)
        image_tensor = image_tensor.to(device)
        # Turn the input into a Variable
        input = Variable(image_tensor)

        # Predict the class of the image
        output = model(input)
        output = output.cpu().data.numpy()#gpu:output.data.numpy()
        pre_mask=output[0,:]
        src_image=np.asarray(image)
        pre_mask=np.squeeze(pre_mask)
        pre_mask=pre_mask.transpose(1,0)#转换通道顺序
        pre_mask=image_processing.resize_image(pre_mask,101,101)
        pre_mask=np.where(pre_mask>threshold,1,0)
        mask_code = get_rle_encode(pre_mask)    # 编码
        mask_code_list.append(mask_code)
        if isShow:
            mask_image=get_rle_decode(mask_code)# 解码显示
            image_processing.show_image("src",src_image)
            image_processing.show_image("mask",pre_mask)
            # image_processing.show_image("mask_image",mask_image)

    save_submit(images_list, mask_code_list,out_csv_path)


if __name__=='__main__':
    # image_dir='./images'
    image_dir="E:/git/dataset/tgs-salt-identification-challenge/test/images"
    model_path='./checkpoints/model_epoch1300_step113.pth'
    out_csv_path="./data/submission.csv"
    # net = UNet(n_channels=3, n_classes=1).to(device)
    net = unet_dilated.UNet(n_channels=3, n_classes=1).to(device)

    predict(net,model_path, image_dir,out_csv_path)
    out_csv_path="./data/submission_crf.csv"
    # predict_crf(net,model_path, image_dir,out_csv_path)