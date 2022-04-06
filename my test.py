# -*- coding: utf-8 -*-
"""
用于学习pytorch和神经网络的相关知识
@Time ： 2022/01/12
@Author ：任博闻
@File ：my test
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import os

'''
Variable存放会变化值的地理位置, 里面的值（tensor）发生变换
tensor不能反向传播, variable可以反向传播
Variable计算时，它会逐渐地生成计算图。
这个图就是将所有的计算节点都连接起来，最后进行误差反向传递的时候，
一次性将所有Variable里面的梯度都计算出来，而tensor就没有这个能力。
tensor变成了variable之后才能进行反向传播求梯度
'''

# net = nn.Linear(3, 4)  # 一层的网络，也可以算是一个计算图就构建好了
# input = Variable(torch.randn(2, 3), requires_grad=True)  # 定义一个图的输入变量
# print(input)
# output = net(input)  # 最后的输出
# loss = torch.sum(output)  # 这边加了一个sum() ,因为被backward只能是标量
# loss.backward()  # 到这计算图已经结束，计算图被释放了

# x = Variable(torch.FloatTensor([[1, 2]]), requires_grad=True)  # 定义一个输入变量
# y = Variable(torch.FloatTensor([[3, 4],
#                                 [5, 6]]))
# z = Variable(torch.FloatTensor([1, 2]), requires_grad=True)
# print(x.shape)
# print(z.shape)
# loss = torch.mm(x, y)    # 变量之间的运算
# print(loss)
# loss.backward(torch.FloatTensor([[1, 0]]), retain_graph=True)  # 求梯度，保留图
# print(x.grad.data)   # 求出 x_1 的梯度
# x.grad.data.zero_()  # 最后的梯度会累加到叶节点，所以叶节点清零
# loss.backward(torch.FloatTensor([[0, 1]]))   # 求出 x_2的梯度
# print(x.grad.data)

# for k in range(0, 100):
#     directory_name = './dataset/normal/NORMAL ('+str(k)+')/'
#     if not os.path.exists(directory_name):
#         os.makedirs(directory_name)
'''
top-k准确率
'''
output = torch.tensor([[-5.4783, 0.2298],
                       [-4.2573, -0.4794],
                       [-0.1070, -5.1511],
                       [-0.1785, -4.3339]])
maxk = max((1,))  # 取top1准确率，若取top1和top5准确率改为max((1,5))
_, pred = output.topk(maxk, 1, True, True)
print(_)
print('predict:', pred)
topk = (1,)
print('topk:', topk)

pred = pred.t()
print(pred.shape)
print('pred:', pred)
target = torch.tensor([[1, 0],
                       [0, 1],
                       [1, 0],
                       [1, 0]])
print(target.shape)
# label = torch.zeros(1, 4)
print(target[1])
for one_label in target:
    print(one_label)
    print(one_label.tolist().index(1))

label = torch.tensor([one_label.tolist().index(1) for one_label in target] )  # 找到下标是1的位置
print('label:', label)
correct = pred.eq(label.view(1, -1).expand_as(pred))
print(correct)
class_to = pred[0].cpu().numpy()
print('class_to', class_to)
res = []
for k in topk:
    print('correct[:k]:', correct[:k])
    print('correct[:k].view(-1):', correct[:k].view(-1).float())
    correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
    print('correct_k', correct_k)
    res.append(correct_k.mul_(100.0 / 1))
print(res)
