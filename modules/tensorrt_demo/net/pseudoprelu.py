import torch.nn as nn
import torch
import torch.nn.functional as F

class PseudoPReLu(nn.Module):
    def __init__(self, num_parameters=1, init=0.25, inplace=True):
        self.num_parameters = num_parameters
        super(PseudoPReLu, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_parameters).fill_(init))
        self.inplace = inplace

    def forward(self, input):
        res = F.relu(input)
        min_res = F.relu(-input)
        weight_broadcast = self.weight.reshape(1, int(self.weight.shape[0]), 1, 1)
        return res - weight_broadcast * min_res

    def extra_repr(self):
        return 'num_parameters={}'.format(self.num_parameters)
