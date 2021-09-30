# pytorch visualizer tool
## 1. pytorchviz
> https://github.com/szagoruyko/pytorchviz
## pytorchviz example
```python
import torch
from torch import nn
from torchviz import make_dot, make_dot_from_trace
from torch.autograd import Variable

model = nn.Sequential()
model.add_module('W0', nn.Linear(8, 16))
model.add_module('tanh', nn.Tanh())
model.add_module('W1', nn.Linear(16, 1))

# x = Variable(torch.rand(1, 3,640, 640))
x = Variable(torch.rand(1, 8))
dot=make_dot(model(x), params=dict(model.named_parameters()))

```

## save *.gv file
```python
dot.save("torchviz.gv","data") # save *.gv file
```
## convert *.gv to png image
> dot torchviz.gv -Tpng -o torchviz.png


## vision_for_tensorwatch
> jupyter notebook