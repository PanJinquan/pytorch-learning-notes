# pytorch-learning-tutorials
觉得可以，麻烦给个”Star“
## 1.image_classification 图像分类
>《pytorch实现L2和L1正则化的方法》https://panjinquan.blog.csdn.net/article/details/88426648

## 2.object_detection目标检测
> mobileNet v1 v2 SSD目标检测：该项目是参考《pytorch-ssd》https://github.com/qfgaohao/pytorch-ssd ,修改，主要是方便训练。</br>
<<<<<<< HEAD
> 数据集VOC2007和VOC2012： <br/>
> 训练方法：my_train_ssd.py <br/>
> 修改my_train_ssd.py的参数train_filename和val_filename即可，直接运行训练，例如：<br/>
=======
> 数据集VOC2007和VOC2012：</br>
> 训练方法：my_train_ssd.py</br>
> 修改my_train_ssd.py的参数train_filename和val_filename即可，直接运行训练，例如：</br>
>>>>>>> 166dd1f31f4d3c4ce73f13077cf6619bbff91635
```python
train_filename = 'E:/git/VOC0712_dataset/train.txt' #训练文件
val_filename = 'E:/git/VOC0712_dataset/val.txt'     #测试文件
```
> 测试方法：run_ssd_example.py
```python
net_type = 'mb2-ssd-lite' #模型类型
model_path = 'models/mb2-ssd-lite-Epoch-190-Loss-3.0529016691904802.pth'#模型路径
label_path = 'models/voc-model-labels.txt'#label文件路径
image_path = './dataset/images/6.jpg'#测试图片
```

## 3.DeepLearningTutorials教程
网上收集的Pytorch的学习资料

## 4.caffe2-android
在android上运行caffe2模型实现图像识别的demo

## 5.UNet图像分割
使用UNet模型实现的图像分割
## 一点笔记
<<<<<<< HEAD
> nohup:服务器后台训练,并打印log 
```bash
nohup python train.py --batch_size=4 1>> train.log &
```
> 查看进程并杀死某个进程
```bash
jobs -l
ps aux|grep python
kill 7080
```
=======
>>>>>>> 166dd1f31f4d3c4ce73f13077cf6619bbff91635
> 查看GPU使用情况
```bash
nvidia-smi
```
> 指定使用GPU：
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"#编号从0开始
```
> pytorch版本，检查GPU是否可用
```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device='cuda'
print("-----device:{}".format(device))
print("-----Pytorch version:{}".format(torch.__version__))
```
## 4.相关说明
> Pytorch version:1.0.0 </br>