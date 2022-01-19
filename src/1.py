# -*- coding: utf-8 -*-
"""
@Time : 2021/11/10 22:47 
@Author : Zhi QIN 
@File : 1.py 
@Software: PyCharm
@Brief : 
"""
import torch
from torch.nn import functional as F

inputs = torch.tensor([[[[4, 7], [6, 3]]]])
weights = torch.tensor([[[[0, 1, 0], [1, 2, 1], [0, 1, 0]]]])

res = torch.conv_transpose2d(inputs, weight=weights, padding=0, stride=2, output_padding=0)

res_1 = F.conv_transpose2d(inputs, weight=weights, padding=0, stride=1)
res_pad = F.pad(inputs, pad=[1, 1, 1, 1])

print(inputs.shape)
print(res)
print(res_1)
print(res_pad)
