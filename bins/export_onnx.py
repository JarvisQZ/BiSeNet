# -*- coding: utf-8 -*-
"""
@Time : 2021/9/23 22:32 
@Author : Zhi QIN 
@File : export_onnx.py 
@Software: PyCharm
@Brief : 将训练好的模型导出成onnx格式
"""

import torch
import torch.nn as nn

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))
import argparse
import torch
import cv2
import time
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2  # ? 不知道为什么
import numpy as np
from models.build_BiSeNet import BiSeNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--path_checkpoint', default=r"D:\PythonCode\BiSeNet\results\09-22_17-15\checkpoint_best.pkl",
                    help="path to your dataset")
parser.add_argument('--path_img', default=r"D:\PythonCode\BiSeNet\data\portrait_2000\training\00004.png",
                    help="path to your dataset")
parser.add_argument('--data_root_dir', default=r"D:\PythonCode\BiSeNet\data\portrait_2000",
                    help="path to your dataset")
args = parser.parse_args()

if __name__ == '__main__':
    model = BiSeNet(num_classes=1, context_path="resnet18")
    checkpoint = torch.load(args.path_checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])

    input = torch.randn(1, 3, 512, 512)

    torch.onnx.export(model, input, r"D:\PythonCode\BiSeNet\results\ONNX\model.onnx", verbose=True)

