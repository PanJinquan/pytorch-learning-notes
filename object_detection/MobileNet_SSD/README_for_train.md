# MobileNet-SSD

## 训练方法
- python train_ssd.py  --net mb2-ssd-lite   --batch_size 24 --num_epochs 200 --scheduler cosine --lr 0.01 --t_max 200
- VOC数据集：链接: https://pan.baidu.com/s/1jn1jD514fQbUIKnUOVfBVw 提取码: 4mbv 
```python
train_filename = 'E:/git/VOC0712_dataset/train.txt' #训练文件
val_filename = 'E:/git/VOC0712_dataset/val.txt'     #测试文件
num_classes = 21                                    #样本个数
dataset_type = 'voc'
```

## 测试方法
- run_ssd_example.py
```python
net_type = 'mb2-ssd-lite'
model_path = 'models/mb2-ssd-lite-Epoch-190-Loss-3.0529016691904802.pth'
# model_path = 'models/mb2-ssd-lite-my.pth'
label_path = 'models/voc-model-labels.txt'
image_path = './dataset/images/6.jpg'
```