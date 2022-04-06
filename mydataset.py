# -*- coding: utf-8 -*-
"""
@Time ： 2022/01/10
@Author ：任博闻
@File ：my dataset
用于制作自己的数据集
使用get_dataset.py作为参考
"""
import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
# torchvision主要是一些数据预处理，数据增强与数据一些变换，如transform，提前预留为之后的程序做准备
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import cv2
import get_feature
import numpy as np
from torchvision.datasets import DatasetFolder


def my_loader(path):
    """
    将图片文件转换为RGB形式，使用PIL内置的Image函数
    :param path: dataset的所在位置
    :return:
    """
    return Image.open(path).convert('RGB')


# 数据预处理与变换
def my_img_tranform():
    """
    数据预处理与变换
    :return: 将图片数据resize并转换成tensor
    """
    my_transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.5),
        # transforms.CenterCrop(224),
        # 使用VGG网络提取特征需要将图片变成（224，224）
        # transforms.Resize((224, 224)),
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # 归一化
    ])
    return my_transform


def my_target_tranform():
    """
    标签转换为one hot向量
    参考：https://blog.csdn.net/chenvvei/article/details/117112938
    :return: 返回标签的one-hot向量
    """
    tarhget_transform = transforms.Compose([
        transforms.Lambda(lambda y: torch.zeros(2, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
    ])
    # tarhget_transform = torch.tensor()
    return tarhget_transform


def find_classes(directory: str):
    """
    得到图片文件夹（数据集）中class的名称和对应的分类关系
    :param directory: dataset地址
    :return: label的名字与索引
    """
    classes = [d.name for d in os.scandir(directory) if d.is_dir()]
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def make_dataset(
        directory: str,
        class_to_idx,
        loader,
        transform,
):
    """
    读取dataset文件夹中的数据
    转化成序列格式的数据
    :arg
    class_to_idx:分类任务对应的指标名称和索引
    loader：读取一张一张的图片
    tranform：图片读取之前的处理手段
    """
    # expanduser 它可以将参数中开头部分的 ~ 或 ~user 替换为当前用户的home目录并返回
    directory = os.path.expanduser(directory)

    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)
    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    # class_to_idx: {'fanyue': 0, 'throw': 1}
    instances = []
    for target_class in sorted(class_to_idx.keys()):
        # target_class: fanyue
        # 对应的class_index：0
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        print(target_dir)
        # target_class表示得到的类别, target_dir表示得到对应的物体类别的文件夹 ./dataset/fanyue
        if not os.path.isdir(target_dir):
            # 判断上面的得到的路径os.path.isdir是否为一目录
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            # fnames 得到对应类别序列文件夹对应的图片（序列）['4.jpg', '5.jpg', '2.jpg', '3.jpg', '1.jpg']
            # sorted可以对所有可迭代的对象进行排序操作
            # root 为序列文件夹的路径 './dataset/fanyue/FANYUE (1)'
            # sequence_length = len(fnames)  # 得到序列图片的长度
            # sequence = torch.zeros(sequence_length, 3, 1024, 1024)  # 用于组合序列图片，组成一个tensor
            # print(root)
            # print(fnames)
            sequence_length = 5  # 指定序列图片的长度为5
            # sequence = torch.zeros(sequence_length, 3, 224, 224)  # 用于组合序列图片，组成一个tensor，对于VGG网络提取到的特征
            sequence = torch.zeros(sequence_length, 3, 1024, 1024)  # 用于组合序列图片，组成一个tensor
            i = 0
            for fname in sorted(fnames):
                if fname == '.DS_Store':
                    # print('.DS_store')
                    continue

                # path为对应图片的路径
                # path: './dataset/fanyue/FANYUE (1)/5.jpg'
                path = os.path.join(root, fname)
                img = transform(loader(path))
                sequence[i] = img
                i = i + 1
                if i == sequence_length:
                    sequence_feature = get_feature.feature_extraction(sequence)
                    item = sequence_feature, class_index
                    instances.append(item)
                    # print('in sequcence')

    return instances


class MyDataset(Dataset):
    """
    继承dataset类构建数据集
    __init__：初始化一些基本参数
    __getitem__：获取单个的数据
    __len__：获取数据的数量
    """

    def __init__(
            self,
            root: str,
            loader,
            transform,
            target_transform,
            # loader: Callable[[str], Any],
            # extensions: Optional[Tuple[str, ...]] = None,
            # transform: Optional[Callable] = None,
            # target_transform: Optional[Callable] = None,
            # is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        self.root = root
        self.transform = transform
        self.loader = loader
        classes, class_to_idx = find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, self.loader, self.transform)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.target_transform = target_transform

    def __getitem__(self, index):
        sample, target = self.samples[index]
        # 使用cross-entropy target需要变成long
        target = torch.tensor(target).long()
        # 只用使用one-hot向量的时候才需要使用，多分类问题不需要one-hot向量
        # target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)


def load_data(directory):
    """
    加载数据集，将上述文件集成
    :param directory: 数据集文件的地址
    :return: 返回整理好的数据集
    """
    train_data = MyDataset(root=directory, loader=my_loader, transform=my_img_tranform(),
                           target_transform=my_target_tranform())
    # train_data = DataLoader(dataset=train, batch_size=5, shuffle=True, num_workers=0, pin_memory=True)
    return train_data

# if __name__ == '__main__':
#     # load_data()
#     directory = './dataset'
#     load_data(directory)
# x = my_loader('./dataset/fanyue/FANYUE (1)/1.jpg')
# img_data = np.array(x)
# print(img_data.shape)
# y = Image.open('./dataset/fanyue/FANYUE (1)/1.jpg')
# y_data = np.array(y)
# print(y.size)
# print(y_data.shape)
# transform = tranform()
# yy = transform(y)
# yyy = transforms.ToTensor()(y_data)
# print(yy)
# loader = my_loader
# transform = tranform()
# available_classes = set()
# # class_to_idx: {'fanyue': 0, 'throw': 1}
# _, class_to_idx = find_classes(directory)
# instances = []
# for target_class in sorted(class_to_idx.keys()):
#     # target_class: fanyue
#     # 对应的class_index：0
#     class_index = class_to_idx[target_class]
#     target_dir = os.path.join(directory, target_class)
#     # target_class表示得到的类别, target_dir表示得到对应的物体类别的文件夹 ./dataset/fanyue
#     if not os.path.isdir(target_dir):
#         # 判断上面的得到的路径os.path.isdir是否为一目录
#         continue
#     for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
#         # sorted可以对所有可迭代的对象进行排序操作
#         # fnames 得到对应类别序列文件夹对应的图片（序列）['4.jpg', '5.jpg', '2.jpg', '3.jpg', '1.jpg']
#         # root 为序列文件夹的路径 './dataset/fanyue/FANYUE (1)'
#         sequence_length = len(fnames)
#         sequence = torch.zeros(sequence_length, 3, 1024, 1024)
#         i = 0
#         for fname in sorted(fnames):
#             if fname == '.DS_Store':
#                 continue
#             # path为对应图片的路径
#             # path: './dataset/fanyue/FANYUE (1)/5.jpg'
#             path = os.path.join(root, fname)
#             img = transform(loader(path))
#             sequence[i] = img
#             i = i+1
#             if i == sequence_length:
#                 sequence_feature = get_feature.feature_extraction(sequence)
#                 item = sequence_feature, class_index
#                 instances.append(item)
#
# print(instances)
# print(instances)
