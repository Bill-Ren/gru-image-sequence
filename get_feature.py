"""
2022.01.05
任博闻
https://blog.csdn.net/fjssharpsword/article/details/104440605
基于pytorch开发CNN提取全连接层作为特征
提取了16维的作为图像特征。训练时，使用5类特征（最后一层）计算softmax损失；并用倒第二全连接层（16维）作为特征
"""
import sys
import os
import math
import random
import heapq
import time
import numpy as np
import pandas as pd
from functools import reduce
from scipy.spatial.distance import pdist
from PIL import Image
import matplotlib.pyplot as plt
import cv2
# import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CnnFclNet(nn.Module):
    """
    定义卷积层
    首先需要super()，给父类的nn.module初始化
    Args:
    """

    def __init__(self, inChannels=3):
        super(CnnFclNet, self).__init__()
        # (channels, Height, Width)
        # layer1: Convolution, (3,1024,1024)->(16,512,512)
        self.conv1 = nn.Conv2d(in_channels=inChannels, out_channels=16, kernel_size=3, padding=1, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        # layer2: max pooling,(16,512,512)->(16,256,256)
        self.maxpool = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(16)
        # layer3: Convolution, (16,256,256)->(8,128,128)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(8)
        self.relu2 = nn.ReLU(inplace=True)
        # layer4: mean pooling, (8,128,128)->(8,64,64)
        self.avgpool1 = nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
        self.bn4 = nn.BatchNorm2d(8)
        # layer5: Convolution, (8,64,64)->(4*32*32)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, padding=1, stride=2)
        self.bn5 = nn.BatchNorm2d(4)
        self.relu3 = nn.ReLU(inplace=True)
        # layer6: mean pooling, (4,32,32)->(4,16,16)
        self.avgpool2 = nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
        self.bn6 = nn.BatchNorm2d(4)
        # layer7: fully connected, 4*16*16->512
        self.fcl1 = nn.Linear(4 * 16 * 16, 512)
        self.relu4 = nn.ReLU(inplace=True)
        # layer8: Hashing layer, 512->16
        self.fcl2 = nn.Linear(512, 16)  #
        self.tanh = nn.Tanh()
        # layer9: fully connected, 16->5
        self.fcl3 = nn.Linear(16, 5)  # type:5

    def forward(self, x):
        """
        真正执行数据的流动, 输入的类型应该是tensor, x作为形参传入forward函数
        使用nn.Module创建的pytorch网络必须定义一个forward方法, 它接受一个一个张量x, 并将其传递给在_init_方法中定义的操作
        需要在forward方法中正确排列操作所需要的顺序

        :param x: input: (batch_size, in_channels, Height, Width)
        :return: output: (batch_size, out_channels, Height, Width)
        """
        # layer1: convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        # layer2: max pooling
        x = self.maxpool(x)
        x = self.bn2(x)
        # layer3: Convolution
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.relu2(x)
        # layer4: mean pooling
        x = self.avgpool1(x)
        x = self.bn4(x)
        # layer5: Convolution
        x = self.conv3(x)
        x = self.bn5(x)
        x = self.relu3(x)
        # layer6: mean pooling
        x = self.avgpool2(x)
        x = self.bn6(x)
        # layer7:fully connected
        x = x.view(x.size(0), -1)  # transfer three dims to one dim
        x = self.fcl1(x)
        x = self.relu4(x)
        # layer8: fully connected
        # 输出为(5,16)
        x = self.fcl2(x)
        x = self.tanh(x)  # [-1,1]
        # layer9: fully connected
        # 输出(5,5)
        out = self.fcl3(x)

        return x, out


class Vgg16_net(nn.Module):
    def __init__(self):
        super(Vgg16_net, self).__init__()
        # (5,3,224,224)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),  # (32-3+2)/1+1=32   32*32*64
            nn.BatchNorm2d(64),
            # inplace-选择是否进行覆盖运算
            # 意思是是否将计算得到的值覆盖之前的值，比如
            nn.ReLU(inplace=True),
            # 意思就是对从上层网络Conv2d中传递下来的tensor直接进行修改，
            # 这样能够节省运算内存，不用多存储其他变量

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            # (32-3+2)/1+1=32    32*32*64
            # Batch Normalization强行将数据拉回到均值为0，方差为1的正太分布上，
            # 一方面使得数据分布一致，另一方面避免梯度消失。
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)  # (32-2)/2+1=16         16*16*64
        )
        # (5,64,112，112)
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            # (16-3+2)/1+1=16  16*16*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            # (16-3+2)/1+1=16   16*16*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)  # (16-2)/2+1=8     8*8*128
        )
        # （5,128,56,56）
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),  # (8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),  # (8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),  # (8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)  # (8-2)/2+1=4      4*4*256
        )
        # （5,256,28,28）
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            # (4-3+2)/1+1=4    4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # (4-3+2)/1+1=4    4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # (4-3+2)/1+1=4    4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)  # (4-2)/2+1=2     2*2*512
        )
        # (5,512,14,14)
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # (2-3+2)/1+1=2    2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # (2-3+2)/1+1=2     2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # (2-3+2)/1+1=2      2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)  # (2-2)/2+1=1      1*1*512
        )
        # (5,512,7,7)
        self.conv = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5
        )

        self.fc = nn.Sequential(
            # y=xA^T+b  x是输入,A是权值,b是偏执,y是输出
            # nn.Liner(in_features,out_features,bias)
            # in_features:输入x的列数  输入数据:[batchsize,in_features]
            # out_freatures:线性变换后输出的y的列数,输出数据的大小是:[batchsize,out_features]
            # bias: bool  默认为True
            # 线性变换不改变输入矩阵x的行数,仅改变列数

            # 采用新的方式
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(in_features=4096, out_features=100, bias=True)
            # nn.Linear(512, 512),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            #
            # nn.Linear(512, 256),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            #
            # nn.Linear(256, 10)
        )

        self.fc_get_feature = nn.Sequential(
            nn.Linear(in_features=25088, out_features=50, bias=True),
        )

    def forward(self, x):
        x = self.conv(x)
        # 这里-1表示一个不确定的数，就是你如果不确定你想要reshape成几行，但是你很肯定要reshape成512列
        # 那不确定的地方就可以写成-1

        # 如果出现x.size(0)表示的是batchsize的值
        x = x.view(x.size(0), -1)
        # x = x.view(-1, 512)
        x = self.fc_get_feature(x)
        return x


def resnet_feature(img, path):
    """
    使用预训练的resnet提取特征
    :param path: 预训练模型存放的地址
    :return: 返回特征
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet34(pretrained=False)
    load_checkpoint = path
    # state_dict = torch.load(load_checkpoint, map_location=lambda storage, loc: storage)
    state_dict = torch.load(load_checkpoint)
    # model = nn.DataParallel(model)
    model.load_state_dict(state_dict)
    # model = model.to(device)

    model.eval()

    # 删掉resnet的倒数前两层
    features = list(model.children())[:-2]
    # print(list(model.children())[:-2])
    # 需要将其从list列表的结构变成sequential的模型结构
    # 通过nn.sequential函数将列表通过非关键字参数的形式传入(列表layers前有一个星号)

    # model用于使用resnet提取特征
    model1 = nn.Sequential(*features,
                           )
    # model2用于使用线性层将特征降维,输出256个特征
    model2 = nn.Sequential(nn.Linear(in_features=25088, out_features=50, bias=True),
                           )
    # img = img.to(device)
    out1 = model1(img)
    out2 = out1.view(out1.size(0), -1)
    fc_out = model2(out2)
    print(fc_out)
    return fc_out

def resnet50_feature(img, path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet50(pretrained = False)
    fc_imputs = model.fc.in_features
    model.fc = nn.Linear(fc_imputs, 2)
    state_dict = torch.load(path)
    model = nn.DataParallel(model)
    model.load_state_dict(state_dict)
    model = model.to(device)
    
    model.eval()

    # 删掉resnet的倒数前两层
    features = list(model.children())[:-2]
    # print(list(model.children())[:-2])
    # 需要将其从list列表的结构变成sequential的模型结构
    # 通过nn.sequential函数将列表通过非关键字参数的形式传入(列表layers前有一个星号)

    # model用于使用resnet提取特征
    model1 = nn.Sequential(*features,
                           )
    # model2用于使用线性层将特征降维,输出256个特征
    model2 = nn.Sequential(nn.Linear(in_features=150528, out_features=50, bias=True),
                           )
    out1 = model1(img)
    out2 = out1.view(out1.size(0), -1)
    # print(out2.shape)
    fc_out = model2(out2)
    # print(fc_out)
    return fc_out




def feature_extraction(sequence):
    # 使用普通CNN网络提取特征
    # model = CnnFclNet()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # if torch.cuda.device_count() > 1:
    #     print("Use", torch.cuda.device_count(), 'gpus')
    #     model = nn.DataParallel(model)
    # model = model.cuda()
    # out_sequence, _ = model(sequence)


    # 使用VGG16网络提取特征
    # 注意大小需要resize成（224,224）
    # model = Vgg16_net()
    # out_sequence = model(sequence)

    # 使用resnet提取特征
    # path = './checkpoint/resnet34.pth'
    # out_sequence = resnet_feature(sequence, path)

    # 使用resnet-50提取特征
    path = './checkpoint/last.pt'
    out_sequence = resnet50_feature(sequence, path)
    return out_sequence

# model = CnnFclNet()
# x = torch.randn(5, 3, 1024, 1024)
# y, _ = model(x)
# print(y.shape)
