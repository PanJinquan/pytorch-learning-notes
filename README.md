# pytorch-learning-tutorials

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
device='cuda'
print("-----device:{}".format(device))
print("-----Pytorch version:{}".format(torch.__version__))
```
