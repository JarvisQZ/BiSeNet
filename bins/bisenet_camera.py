# -*- coding: utf-8 -*-
"""
@Time : 2021/9/22 23:00 
@Author : Zhi QIN 
@File : bisenet_camera.py 
@Software: PyCharm
@Brief : bisenet 调用摄像头
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))
import argparse
import time
import cv2
import torch
import numpy as np
from models.build_BiSeNet import BiSeNet
import albumentations as A
from tools.predictor import Predictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--path_checkpoint', default=r"D:\PythonCode\BiSeNet\results\09-22_17-15\checkpoint_best.pkl",
                    help="path to your dataset")
args = parser.parse_args()

if __name__ == '__main__':
    in_size = 512
    path_checkpoint = r"D:\PythonCode\BiSeNet\results\09-24_02-00\checkpoint_best.pkl"
    predictor = Predictor(path_checkpoint, device, backbone_name="resnet18", tta=False)  # 边界

    # video_path = r""
    video_path = 0
    vid = cv2.VideoCapture(video_path + cv2.CAP_DSHOW)  # 0表示打开视频，1; cv2.CAP_DSHOW去除黑边

    while True:
        return_value, frame_bgr = vid.read()  # 读取视频每一帧
        if not return_value:
            raise ValueError("No image!")
        # —————————————————————————————————————————————————————————————————— #
        img_t, img_bgr = predictor.preprocess(frame_bgr, in_size=in_size)
        _, pre_label = predictor.predict(img_t)
        result = predictor.postprocess(img_bgr, pre_label, color="w", hide=False)
        # —————————————————————————————————————————————————————————————————— #
        cv2.imshow('result', result)
        # waitKey，参数是1，表示延时1ms；参数为0，如cv2.waitKey(0)只显示当前帧图像，相当于视频暂停
        # ord: 字符串转ASCII 数值
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()               # release()释放摄像头
    cv2.destroyAllWindows()     # 关闭所有图像窗口
