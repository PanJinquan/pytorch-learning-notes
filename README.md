# pytorch-learning-tutorials
觉得可以，麻烦给个”Star“
## 1.image_classification 图像分类
>《pytorch实现L2和L1正则化的方法》https://panjinquan.blog.csdn.net/article/details/88426648

## 2.DeepLearningTutorials教程
网上收集的Pytorch的学习资料

## 3.caffe2-android
在android上运行caffe2模型实现图像识别的demo
## 4.一点笔记
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