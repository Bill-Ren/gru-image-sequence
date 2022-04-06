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
def train_tranform1():
    """
    训练集数据预处理与变换
    常规resize与ToTensor变换
    :return: 将图片数据resize并转换成tensor
    """
    my_transform = transforms.Compose([
        # 使用原始的CNN提取到的特征
        # transforms.Resize((1024, 1024)),
        # transforms.ToTensor(),

        # 使用resnet提取特征
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # 进行归一化
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return my_transform


def train_tranform2():
    """
    数据预处理与变换
    进行其他数据增强的手段, 随机垂直和水平旋转
    :return: 将图片数据resize并转换成tensor
    """
    my_transform = transforms.Compose([
         # 使用原始的CNN提取特征
        # transforms.Resize((1024, 1024)),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomHorizontalFlip(),
        # transforms.ToTensor(),

        # 使用resnet提取特征
        transforms.Resize((224, 224)),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    return my_transform

def train_tranform3():
    """
    数据预处理与变换
    进行其他数据增强的手段, 进行色调和饱和度的变化
    :return: 将图片数据resize并转换成tensor
    """
    my_transform = transforms.Compose([
        # # 使用原始的CNN提取特征
        # transforms.Resize((1024, 1024)),
        # transforms.ColorJitter(brightness=(2, 2)),
        # transforms.ColorJitter(contrast=(2, 2)),
        # transforms.ColorJitter(saturation=(2, 2)),
        # transforms.ToTensor(),

        # 使用原始的resnet提取特征
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.5, contrast = 0.5, saturation = 0.5, hue = 0.5),
        # transforms.ColorJitter(brightness=(2, 2)),
        # transforms.ColorJitter(contrast=(2, 2)),
        # transforms.ColorJitter(saturation=(2, 2)),
        transforms.ToTensor(),
        


        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # 归一化
    ])
    return my_transform

def train_tranform4():
    """
    数据预处理与变换
    进行其他数据增强的手段, 进行随机平移
    :return: 将图片数据resize并转换成tensor
    """
    my_transform = transforms.Compose([
        # # 使用原始的CNN提取特征
        # transforms.Resize((1024, 1024)),
        # transforms.RandomAffine(0, (0.1, 0)),
        # # transforms.CenterCrop(224),
        # transforms.ToTensor(),

        # 使用resnet网络提取特征
        transforms.Resize((224, 224)),
        transforms.RandomAffine(0, (0.1, 0)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # 归一化
    ])
    return my_transform




def test_tranform():
    """
    测试集
    不采用任何的数据增强手段,只包括resize和ToTensor的操作
    数据预处理与变换
    :return: 将图片数据resize并转换成tensor
    """
    my_transform = transforms.Compose([
        # # 使用原始的CNN提取特征
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.5),
        # transforms.CenterCrop(224),
        # transforms.Resize((1024, 1024)),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # 归一化

        # 使用resnet网络提取特征
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # 进行归一化
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return my_transform


def my_target_tranform():
    """
    标签转换为one hot向量
    只有使用one-hot向量才会用到，在多分类问题中不需要用到
    参考：https://blog.csdn.net/chenvvei/article/details/117112938
    :return: 返回标签的one-hot向量
    """
    tarhget_transform = transforms.Compose([
        transforms.Lambda(lambda y: torch.zeros(2, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
    ])
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


def make_train_dataset(
        directory: str,
        class_to_idx,
        loader,
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
    # class_to_idx：类别与序号的对应关系，字典 {'fanyue':0, 'normal':1}
    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)
    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    # class_to_idx: {'fanyue': 0, 'throw': 1}
    # instance：存放最后的序列向量，构成所需要的dataset
    instances = []
    # keys()返回字典的键值

    # 使用原始的数据
    for target_class in sorted(class_to_idx.keys()):
        # target_class: 类别，字典的键值（索引）  fanyue
        # 对应的class_index：类别对应的编号，0
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
            # sequence = torch.zeros(sequence_length, 3, 1024, 1024)  # 用于组合序列图片，组成一个tensor
            sequence = torch.zeros(sequence_length, 3, 224, 224)  # resnet网络，用于组合序列图片，组成一个tensor
            # sequence = torch.zeros(sequence_length, 3, 1024, 1024).to('cuda')
            i = 0
            img_transform = train_tranform1()
            for fname in sorted(fnames):
                if fname == '.DS_Store':
                    # print('.DS_store')
                    continue

                # path为对应图片的路径
                # path: './dataset/fanyue/FANYUE (1)/5.jpg'
                path = os.path.join(root, fname)
                # img_transform = train_tranform1()
                img = img_transform(loader(path))
                sequence[i] = img
                i = i + 1
                if i == sequence_length:
                    sequence_feature = get_feature.feature_extraction(sequence)
                    # item: 为tuple:2
                    # 0= tensor(5,16)
                    # 1= 0或1 表示类别
                    item = sequence_feature, class_index
                    # instanece：
                    instances.append(item)
                    # print('in sequcence')
    '''# 使用第一种数据增强的策略
    for target_class in sorted(class_to_idx.keys()):
        # target_class: 类别，字典的键值（索引）  fanyue
        # 对应的class_index：类别对应的编号，0
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
            # sequence = torch.zeros(sequence_length, 3, 1024, 1024)  # 用于组合序列图片，组成一个tensor
            sequence = torch.zeros(sequence_length, 3, 224, 224)  # resnet网络，用于组合序列图片，组成一个tensor
            # sequence = torch.zeros(sequence_length, 3, 1024, 1024).to('cuda')
            i = 0
            img_transform = train_tranform2()
            for fname in sorted(fnames):
                if fname == '.DS_Store':
                    # print('.DS_store')
                    continue

                # path为对应图片的路径
                # path: './dataset/fanyue/FANYUE (1)/5.jpg'
                path = os.path.join(root, fname)
                # img_transform = train_tranform2()
                img = img_transform(loader(path))
                sequence[i] = img
                i = i + 1
                if i == sequence_length:
                    sequence_feature = get_feature.feature_extraction(sequence)
                    # item: 为tuple:2
                    # 0= tensor(5,16)
                    # 1= 0或1 表示类别
                    item = sequence_feature, class_index
                    # instanece：
                    instances.append(item)'''
                    # print('in sequcence')

    # 使用第二种数据增强的策略
    # for target_class in sorted(class_to_idx.keys()):
    #     # target_class: 类别，字典的键值（索引）  fanyue
    #     # 对应的class_index：类别对应的编号，0
    #     class_index = class_to_idx[target_class]
    #     target_dir = os.path.join(directory, target_class)
    #     print(target_dir)
    #     # target_class表示得到的类别, target_dir表示得到对应的物体类别的文件夹 ./dataset/fanyue
    #     if not os.path.isdir(target_dir):
    #         # 判断上面的得到的路径os.path.isdir是否为一目录
    #         continue
    #     for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
    #         # fnames 得到对应类别序列文件夹对应的图片（序列）['4.jpg', '5.jpg', '2.jpg', '3.jpg', '1.jpg']
    #         # sorted可以对所有可迭代的对象进行排序操作
    #         # root 为序列文件夹的路径 './dataset/fanyue/FANYUE (1)'
    #         # sequence_length = len(fnames)  # 得到序列图片的长度
    #         # sequence = torch.zeros(sequence_length, 3, 1024, 1024)  # 用于组合序列图片，组成一个tensor
    #         # print(root)
    #         # print(fnames)
    #         sequence_length = 5  # 指定序列图片的长度为5
    #         # sequence = torch.zeros(sequence_length, 3, 1024, 1024)  # 用于组合序列图片，组成一个tensor
    #         sequence = torch.zeros(sequence_length, 3, 224, 224)  # resnet网络，用于组合序列图片，组成一个tensor
    #         # sequence = torch.zeros(sequence_length, 3, 1024, 1024).to('cuda')
    #         i = 0
    #         img_transform = train_tranform3()
    #         for fname in sorted(fnames):
    #             if fname == '.DS_Store':
    #                 # print('.DS_store')
    #                 continue

    #             # path为对应图片的路径
    #             # path: './dataset/fanyue/FANYUE (1)/5.jpg'
    #             path = os.path.join(root, fname)
    #             # img_transform = train_tranform3()
    #             img = img_transform(loader(path))
    #             sequence[i] = img
    #             i = i + 1
    #             if i == sequence_length:
    #                 sequence_feature = get_feature.feature_extraction(sequence)
    #                 # item: 为tuple:2
    #                 # 0= tensor(5,16)
    #                 # 1= 0或1 表示类别
    #                 item = sequence_feature, class_index
    #                 # instanece：
    #                 instances.append(item)

    return instances


def make_test_dataset(
        directory: str,
        class_to_idx,
        loader,
):
    """
    读取测试集的数据
    与训练集不同的是不需要数据增强
    转化成序列格式的数据
    :arg
    class_to_idx:分类任务对应的指标名称和索引
    loader：读取一张一张的图片
    tranform：图片读取之前的处理手段
    """
    # expanduser 它可以将参数中开头部分的 ~ 或 ~user 替换为当前用户的home目录并返回
    directory = os.path.expanduser(directory)
    # class_to_idx：类别与序号的对应关系，字典 {'fanyue':0, 'normal':1}
    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)
    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    # class_to_idx: {'fanyue': 0, 'throw': 1}
    # instance：存放最后的序列向量，构成所需要的dataset
    instances = []
    # keys()返回字典的键值
    for target_class in sorted(class_to_idx.keys()):
        # target_class: 类别，字典的键值（索引）  fanyue
        # 对应的class_index：类别对应的编号，0
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
            # sequence = torch.zeros(sequence_length, 3, 1024, 1024)  # 用于组合序列图片，组成一个tensor
            sequence = torch.zeros(sequence_length, 3, 224, 224)  # resnet网络，用于组合序列图片，组成一个tensor
            # sequence = torch.zeros(sequence_length, 3, 1024, 1024).to('cuda')
            i = 0
            img_transform = test_tranform()
            for fname in sorted(fnames):
                if fname == '.DS_Store':
                    # print('.DS_store')
                    continue

                # path为对应图片的路径
                # path: './dataset/fanyue/FANYUE (1)/5.jpg'
                path = os.path.join(root, fname)
                # img_transform = test_tranform()
                img = img_transform(loader(path))
                sequence[i] = img
                i = i + 1
                if i == sequence_length:
                    sequence_feature = get_feature.feature_extraction(sequence)
                    # item: 为tuple:2
                    # 0= tensor(5,16)
                    # 1= 0或1 表示类别
                    item = sequence_feature, class_index
                    # instanece：
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
            mode
            # transform,
            # target_transform,
            # loader: Callable[[str], Any],
            # extensions: Optional[Tuple[str, ...]] = None,
            # transform: Optional[Callable] = None,
            # target_transform: Optional[Callable] = None,
            # is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        self.root = root
        # self.transform = transform
        self.loader = loader
        classes, class_to_idx = find_classes(self.root)
        # 训练集和测试集数据增广的方式不同
        if mode == 'train':
            samples = make_train_dataset(self.root, class_to_idx, self.loader)
        else:
            samples = make_test_dataset(self.root, class_to_idx, self.loader)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        # self.target_transform = target_transform

    def __getitem__(self, index):
        sample, target = self.samples[index]
        # 使用cross-entropy target需要变成long
        target = torch.tensor(target).long()
        # 只用使用one-hot向量的时候才需要使用，多分类问题不需要one-hot向量
        # target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)


def load_data(directory, mode):
    """
    加载数据集，将上述文件集成
    :param directory: 数据集文件的地址
    :return: 返回整理好的数据集
    """
    train_data = MyDataset(root=directory, loader=my_loader, mode=mode)
    return train_data


# if __name__ == '__main__':
#     # load_data()
#     train_directory = './dataset-verify/train'
#     test_directory = './dataset-verify/test'
#     train_dataset = load_data(train_directory, mode='train')
#     valid_dataset = load_data(test_directory, mode='test')
#     # 调整batch_size，测试集的batch_size=1
#     train_loader = DataLoader(train_dataset, shuffle=False, drop_last=True, batch_size=1)
#     valid_loader = DataLoader(valid_dataset, shuffle=False, drop_last=True, batch_size=1)
#     # Class labels
#     classes = ('fanyue', 'normal')
#     print('Training set has {} instances'.format(len(train_dataset)))
#     print('Validation set has {} instances'.format(len(valid_dataset)))
#     train_data_size = len(train_dataset)
#     print('训练集数量：%d' % train_data_size)
#     valid_data_size = len(valid_dataset)
#     print('验证集数量：%d' % valid_data_size)

#     # 打印数据集
#     i = 0
#     for image, label in train_loader:
#         i = i + 1
#         print(i)
#         print(image.shape)
#         print(label)
#     i = 0
#     for image, label in valid_loader:
#         i = i + 1
#         print(i)
#         print(image.shape)
#         print(label)
