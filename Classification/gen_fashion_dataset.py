# -*-coding: utf-8 -*-
"""
    @Project: pytorch-learning-tutorials
    @File   : fashion_mnist_dataset.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-03-07 16:14:33
    @url: 《pytorch: 准备、训练和测试自己的图片数据》https://www.cnblogs.com/denny402/p/7520063.html
"""
import os
from skimage import io
import torchvision.datasets.mnist as mnist

root="./dataset/fashion_mnist"
train_set = (
    mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
    mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
        )
test_set = (
    mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
    mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))
        )
print("training set :",train_set[0].size())
print("test set :",test_set[0].size())

def convert_to_img(train=True):
    if(train):
        f=open(root+'/train.txt','w')
        data_path=root+'/train/'
        if(not os.path.exists(data_path)):
            os.makedirs(data_path)
        for i, (img,label) in enumerate(zip(train_set[0],train_set[1])):
            img_path=data_path+str(i)+'.jpg'
            io.imsave(img_path,img.numpy())
            label=int(label)
            f.write(img_path+' '+str(label)+'\n')
        f.close()
    else:
        f = open(root + '/test.txt', 'w')
        data_path = root + '/test/'
        if (not os.path.exists(data_path)):
            os.makedirs(data_path)
        for i, (img,label) in enumerate(zip(test_set[0],test_set[1])):
            img_path = data_path+ str(i) + '.jpg'
            io.imsave(img_path, img.numpy())
            label=int(label)
            f.write(img_path + ' ' + str(label) + '\n')
        f.close()

convert_to_img(train=True)
convert_to_img(train=False)