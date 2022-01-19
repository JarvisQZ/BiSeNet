# -*- coding: utf-8 -*-
"""
@Time : 2021/9/22 22:39 
@Author : Zhi QIN 
@File : predictor.py 
@Software: PyCharm
@Brief : 分割模型封装
"""

import os
import time
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from models.build_BiSeNet import BiSeNet
import albumentations as A
import matplotlib.pyplot as plt
import ttach as tta


class Predictor(object):
    def __init__(self, path_checkpoint, device, backbone_name="resnet101", tta=False):
        self.backbone_name = backbone_name
        self.path_checkpoint = path_checkpoint
        self.device = device
        self.tta = tta
        self.model = self.get_model(backbone_name)
        self.transform = A.Compose([A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def get_model(self, backbone_name):
        model = BiSeNet(1, backbone_name)
        checkpoint = torch.load(self.path_checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        if self.tta:
            transforms = tta.Compose(
                [
                    tta.HorizontalFlip(),
                    # tta.Rotate90(angles=[0, 180]),
                    # tta.Scale(scales=[1, 2, 4]),
                    # tta.Multiply(factors=[0.9, 1, 1.1]),
                ]
            )
            model = tta.SegmentationTTAWrapper(model, transforms, merge_mode='mean')

        return model

    def preprocess(self, img_src, in_size=512, keep_ratio=False):
        # 图片预处理定义
        if isinstance(img_src, str):
            assert os.path.exists(img_src), "{} is not exists! "
            img_bgr = self.cv_imread(img_src)
        elif isinstance(img_src, np.ndarray):
            img_bgr = img_src
        else:
            raise ValueError("input must be path or np.ndarray")

        if in_size:
            if keep_ratio:
                h, w, c = img_bgr.shape
                ratio = in_size / min(h, w)
                img_bgr = cv2.resize(img_bgr, (0, 0), fx=ratio, fy=ratio)
            else:
                img_bgr = cv2.resize(img_bgr, (in_size, in_size))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if self.transform:
            transformed = self.transform(image=img_rgb, mask=img_rgb)
            img_rgb = transformed['image']

        img_tensor = torch.tensor(np.array(img_rgb), dtype=torch.float).permute(2, 0, 1)
        img_tensor.unsqueeze_(0)
        img_tensor = img_tensor.to(self.device)
        return img_tensor, img_bgr

    def predict(self, img_tensor):
        with torch.no_grad():
            torch.cuda.synchronize()
            s = time.time()
            outputs = self.model(img_tensor)
            torch.cuda.synchronize()
            print("{:.4f}s {:.1f}FPS".format(time.time() - s, 1 / (time.time() - s)))
            outputs = torch.sigmoid(outputs).squeeze(1)
            pre_label = outputs.data.cpu().numpy()
        return outputs, pre_label[0]

    def postprocess(self, img, pre_label, color="w", hide=False):

        # 背景颜色
        background = np.zeros_like(img, dtype=np.uint8)
        if color == "b":
            background[:, :, 0] = 255
        elif color == "w":
            background[:] = 255
        elif color == "r":
            background[:, :, 2] = 255

        # alpha
        alpha_bgr = pre_label
        alpha_bgr = cv2.cvtColor(alpha_bgr, cv2.COLOR_GRAY2BGR)
        h, w, c = img.shape
        alpha_bgr = cv2.resize(alpha_bgr, (w, h))

        # fusion
        if not hide:
            result = np.uint8(img * alpha_bgr + background * (1 - alpha_bgr))
        else:
            result = np.uint8(background * alpha_bgr + img * (1 - alpha_bgr))
        return result

    @staticmethod
    def cv_imread(file_path):
        cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        return cv_img

    @staticmethod
    def save_img(path_img, img_src):
        base_dir = os.path.dirname(path_img)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        cv2.imwrite(path_img, img_src)


if __name__ == '__main__':
    set_name = "testing"
    in_size = 512

    path_ckpt = r'D:\PythonCode\BiSeNet\results\09-27_00-37\checkpoint_best.pkl'
    root_dir = r'D:\PythonCode\BiSeNet\data\portrait_2000/{}'.format(set_name)
    out_dir = os.path.join(os.path.dirname(path_ckpt), "{}_{}".format(set_name, in_size))

    dir_name = os.path.dirname(path_ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu") # 效果非常差

    names_lst = os.listdir(root_dir)
    names_lst = [n for n in names_lst if not n.endswith("matte.png")]
    path_imgs = [os.path.join(root_dir, n) for n in names_lst]

    # predictor = Predictor(path_ckpt, device=device)
    predictor = Predictor(path_ckpt, device=device, backbone_name='xception39', tta=True)

    for idx, path_img in enumerate(path_imgs):
        # 推理
        img_t, img_bgr = predictor.preprocess(path_img, in_size=in_size)  # 1.预处理
        _, pred_mask = predictor.predict(img_t)  # 2.推理获得mask
        out_img = predictor.postprocess(img_bgr, pred_mask, color="w")  # 后处理，保存图片。 color控制背景颜色
        # 清理缓存
        torch.cuda.empty_cache()
        # 保存
        concat_img = np.concatenate([img_bgr, out_img], axis=1)
        path_out = os.path.join(out_dir, os.path.basename(path_img))
        predictor.save_img(path_out, concat_img)
