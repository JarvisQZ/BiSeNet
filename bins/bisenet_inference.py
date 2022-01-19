# -*- coding: utf-8 -*-
"""
@Time : 2021/9/22 16:44 
@Author : Zhi QIN 
@File : bisenet_inference.py 
@Software: PyCharm
@Brief : bisenet 推理
"""

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
parser.add_argument('--path_checkpoint', default=r"D:\PythonCode\BiSeNet\results\09-27_00-15\checkpoint_best.pkl",
                    help="path to your dataset")
parser.add_argument('--path_img', default=r"D:\PythonCode\BiSeNet\data\portrait_2000\training\00004.png",
                    help="path to your dataset")
parser.add_argument('--data_root_dir', default=r"D:\PythonCode\BiSeNet\data\portrait_2000",
                    help="path to your dataset")
args = parser.parse_args()

if __name__ == '__main__':
    # path_img = args.path_img
    path_img = r"D:\PythonCode\BiSeNet\data\portrait_2000\training\01200.png"
    in_size = 512  # 224， 448 ， 336 ， 1024
    norm_mean = (0.5, 0.5, 0.5)  # 比imagenet的mean效果好
    norm_std = (0.5, 0.5, 0.5)
    # step1: 数据预处理
    transform = A.Compose([
        A.Resize(width=in_size, height=in_size),
        A.Normalize(norm_mean, norm_std),
        ToTensorV2(),
    ])
    img_bgr = cv2.imread(path_img)  # 读取图片
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # 注意 opencv默认bgr，需要转换为rgb
    transformed = transform(image=img_rgb, mask=img_rgb)  # 预处理
    # img_rgb = transformed['image']  # 取出图片
    img_tensor = transformed['image']
    # img_tensor = torch.tensor(np.array(img_rgb), dtype=torch.float).permute(2, 0, 1)  # 转张量，hwc--> chw
    img_tensor.unsqueeze_(0)  # chw --> bchw
    img_tensor = img_tensor.to(device)

    # step2: 模型加载
    model = BiSeNet(num_classes=1, context_path="xception39")
    checkpoint = torch.load(args.path_checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # step3: 推理
    with torch.no_grad():
        # 因cpu/gpu是异步的，因此用cpu对gpu计时，应当等待gpu准备好再即使该操作是等待GPU全部执行结束，CPU才可以读取时间信息。
        torch.cuda.synchronize()
        s = time.time()
        outputs = model(img_tensor)
        torch.cuda.synchronize()  # 该操作是等待GPU全部执行结束，CPU才可以读取时间信息。
        print("{:.4f}s".format(time.time() - s))
        outputs = torch.sigmoid(outputs).squeeze(1)
        pre_label = outputs.data.cpu().numpy()[0]  # 取batch的第一个

    # step4：后处理显示图像
    background = np.zeros_like(img_bgr, dtype=np.uint8)
    background[:] = 255
    alpha_bgr = pre_label
    alpha_bgr = cv2.cvtColor(alpha_bgr, cv2.COLOR_GRAY2BGR)
    h, w, c = img_bgr.shape
    alpha_bgr = cv2.resize(alpha_bgr, (w, h))
    # fusion
    result = np.uint8(img_bgr * alpha_bgr + background * (1 - alpha_bgr))
    out_img = np.concatenate([img_bgr, result], axis=1)
    cv2.imshow("result", out_img)
    cv2.waitKey()
