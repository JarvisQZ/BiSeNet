# -*- coding: utf-8 -*-
"""
@Time : 2021/9/22 13:30 
@Author : Zhi QIN 
@File : portrait_dataset.py 
@Software: PyCharm
@Brief : 读取Portrait数据集
"""
import os
import random

import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co, ConcatDataset


class PortraitDataset2000(Dataset):
    cls_num = 2
    names = ['bg', 'portrait']

    def __init__(self, root_dir, transform=None, in_size=224):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.in_size = in_size
        self.label_path_list = list()
        # 获取mask的path
        self._get_img_path()

    def __len__(self):
        return len(self.label_path_list)

    def __getitem__(self, index) -> T_co:
        # step1: 读取样本， 得到ndarray的形式
        path_label = self.label_path_list[index]
        path_img = path_label[:-10] + '.png'
        img_bgr = cv2.imread(path_img)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        msk_rgb = cv2.imread(path_label)  # 3个通道值是一样的 # 0-255，边界处是平滑的, 3个通道值是一样的

        # step2: 图片预处理
        if self.transform:
            transformed = self.transform(image=img_rgb, mask=msk_rgb)
            img_rgb = transformed['image']
            msk_rgb = transformed['mask']

        # step3: 处理数据成为需要的形式
        img_rgb = img_rgb.transpose((2, 0, 1))  # hwc --> chw
        img_chw_tensor = torch.from_numpy(img_rgb).float()

        msk_gray = msk_rgb[:, :, 0]  # hwc --> hw
        msk_gray = msk_gray / 255.  # [0,255] scale [0,1] 连续变量
        label_tensor = torch.tensor(msk_gray, dtype=torch.float)  # 标签输出为 0-1之间的连续变量 ，shape=(224, 224)

        return img_chw_tensor, label_tensor

    def _get_img_path(self):
        file_list = os.listdir(self.root_dir)
        file_list = list(filter(lambda x: x.endswith('_matte.png'), file_list))
        path_list = [os.path.join(self.root_dir, name) for name in file_list]
        random.shuffle(path_list)
        if len(path_list) == 0:
            raise Exception("\nroot_dir:{} is  an empty dir!")
        self.label_path_list = path_list


if __name__ == '__main__':
    por_dir = r'../data/portrait_2000/training'
    dir_17 = r'D:\PythonCode\BiSeNet\data\data_aug_17'

    por_set = PortraitDataset2000(por_dir)
    set_17 = PortraitDataset2000(dir_17)
    all_set = ConcatDataset([por_set, dir_17])
    train_loader = DataLoader(por_set, batch_size=1, shuffle=True, num_workers=8)
    print(len(train_loader))

    for i, sample in enumerate(train_loader):
        img, label = sample

        print(img.shape, label.shape)
        print(img)
        print(label)
        break
