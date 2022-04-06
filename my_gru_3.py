# -*- coding: utf-8 -*-
"""
@Time ： 2022/01/11
@Author ：任博闻
@File ：my gru
使用GRU网络训练自己的数据
参考代码：time-sequcence-gru.py

训练集和测试集没有分开,在同一个文件夹下，没有分开。
"""
import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
import mydataset
import shutil
import matplotlib.pyplot as plt


class GRUnet(nn.Module):
    """
    自定义GRU模块
     Args:
    input_dim = 16
    output_dim = 2
    n_layers = 1
    hidden_dim = 100
    batch_size = 1
    """

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(GRUnet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        # 将最后一个rnn的输出使用全连接得到最后的分类结果
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    # 前向传播的一过程
    def forward(self, input):
        batch_size = input.size(0)
        # 使用默认的隐藏状态
        # output, h = self.gru(input)

        h = self.init_hidden(batch_size)
        output, h = self.gru(input, h)
        # h[-1]表示最后一行数据值
        # fc_output = self.fc(self.relu(h[-1]))
        fc_output = self.fc(h[-1])
        return fc_output, h 

    def init_hidden(self, batch_size):
        # weight = next(self.parameters()).data
        # hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
        return hidden


def accuracy(output, target, topk=(1,)):
    """
        计算topk的准确率
        output：是预测值
        target：是标签
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        # 使用cross-entropy
        target = target.to(device)
        correct = pred.eq(target.view(1, -1).expand_as(pred)).to(device)
         # 使用one-hot编码
        # label = torch.tensor([one_label.tolist().index(1) for one_label in target]).to(device)
        # correct = pred.eq(label.view(1, -1).expand_as(pred)).to(device)
        # class_to为预测的类别
        class_to = pred[0].cpu().detach().numpy()

        res = []
        for k in topk:
            # view(-1), X里面的所有维度数据转化成一维的，并且按先后顺序排列。
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res, class_to


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(train_loader, model, criterion, optimizer, epoch):
    # 计算准确率的一系列参数
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()  

    model.train()

    end = time.time()
    i = 0
    for (x, label) in train_loader:
        # measure data loading time
        data_time.update(time.time() - end)

        # print(x.shape)
        # print(label.shape)
        # print(label)

        # compute output
        out, _ = model(x.to(device).float())
        # 使用MSE loss，但对应的是回归问题
        # loss = criterion(out, label.to(device).float())
        # 使用cross entroy loss对应分类问题
        loss = criterion(out, label.to(device).long())

        # measure accuracy and record loss
        [prec1], class_to = accuracy(out, label, topk=(1,))
        # 更新loss与accuracy
        losses.update(loss.item(), x.size(0))
        top1.update(prec1[0], x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        i = i + 1
        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))
    print(' * {} Prec@1 {top1.avg:.3f} '
          .format('train', top1=top1))
    return losses.avg, top1.avg


def validate(val_loader, model, criterion, epoch, phase="VAL"):
    """
        验证代码
        参数：
            val_loader - 验证集的 DataLoader
            model - 模型
            criterion - 损失函数
            epoch - 进行第几个 epoch
            writer - 用于写 tensorboardX
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    i = 0
    with torch.no_grad():
        end = time.time()
        for (input, target) in val_loader:
            # input = input.cuda()
            # target = target.cuda()
            output, _ = model(input.to(device).float())
            # 使用MSE loss，但对应的是回归问题
            # loss = criterion(output, target.to(device).float())
            # 使用cross entroy loss对应分类问题
            loss = criterion(output, target.to(device).long())
            

            # measure accuracy and record loss
            [prec1], class_to = accuracy(output, target, topk=(1,))
            # 更新loss与accuracy
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            i = i + 1
            if i % 10 == 0:
                print('Test-{0}: [{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                    phase, i, len(val_loader),
                    batch_time=batch_time,
                    loss=losses,
                    top1=top1))

        print(' * {} Prec@1 {top1.avg:.3f} '
              .format(phase, top1=top1))
    return losses.avg, top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
        根据 is_best 存模型，一般保存 valid acc 最好的模型
    """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './result/model_best_checkpoint_gru.pth.tar')


'''
主程序
参数设定
'''
"""数据类型、缺失情况(df.isna().sum())等，可绘图查看"""
if __name__ == '__main__':

    # -------------------step 1: 加载数据---------------------------#
    train_directory = './dataset-process'

    full_dataset = mydataset.load_data(train_directory)
    print('总数据集大小：%d' % len(full_dataset))
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(40))
    train_loader = DataLoader(train_dataset, shuffle=False, drop_last=True, batch_size=2)
    valid_loader = DataLoader(valid_dataset, shuffle=False, drop_last=True, batch_size=1)
    # Class labels
    classes = ('fanyue', 'normal')
    print('Training set has {} instances'.format(len(train_dataset)))
    print('Validation set has {} instances'.format(len(valid_dataset)))
    train_data_size = len(train_dataset)
    print('训练集数量：%d' % train_data_size)
    valid_data_size = len(valid_dataset)
    print('验证集数量：%d' % valid_data_size)

    # # 打印数据集
    # i = 0
    # for image, label in train_loader:
    #     i = i + 1
    #     print(i)
    #     print(image.shape)
    #     print(label)
    # for image, label in valid_loader:
    #     i = i + 1
    #     print(i)
    #     print(image.shape)
    #     print(label)

    # ---------------------step 2: 定义网络-----------------------
    # 使用普通的CNN网络提取特征，提取的特征维度为16维
    input_dim = 16
    output_dim = 2
    n_layers = 1
    hidden_dim = 50
    batch_size = 2

    # 使用VGG网络提取特征，提取的特征维度为50维
    # input_dim = 50
    # output_dim = 2
    # n_layers = 1
    # hidden_dim = 50
    # batch_size = 1

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'device is {device}')
    # 使用GRU模型
    model = GRUnet(input_dim, hidden_dim, output_dim, n_layers)
    # 使用多个GPU并行计算
    # if torch.cuda.device_count() > 1:
    #     print("Use", torch.cuda.device_count(), 'gpus')
    #     model = nn.DataParallel(model)
    
    model.to(device)
    # model.to(device)
    # model = model.cuda()

    # ---------------------step 3: 定义损失函数与优化器-----------------------
    learn_rate = 0.001
    # criterion = nn.MSELoss()
    # 尝试采用不同的损失函数
    criterion = torch.nn.CrossEntropyLoss().to(device)
    # criterion = torch.nn.BCELoss().to(device)
    # criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    # ---------------------step 4: 训练-----------------------
    # 绘制损失和准确率的图像
    Train_Loss_list = []
    Train_Accuracy_list = []
    Test_Loss_list = []
    Test_Accuracy_list = []

    epochs = 50
    best_prec1 = 0

    for epoch in range(epochs):
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch)
        # 在验证集上测试效果
        valid_loss, valid_prec1 = validate(valid_loader, model, criterion, epoch, phase="VAL")

        # 将loss和accuracy数据保存
        Train_Loss_list.append(train_loss)
        Train_Accuracy_list.append(train_acc)
        Test_Loss_list.append(valid_loss)
        Test_Accuracy_list.append(valid_prec1)
        # print(Test_Accuracy_list)

        is_best = valid_prec1 > best_prec1
        best_prec1 = max(valid_prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'gru',
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best,
            filename='./result/checkpoint_gru.pth.tar')

    # 绘制图像

    plt.switch_backend('agg')
    epochs_plot = range(len(Test_Accuracy_list))
    plt.plot(epochs_plot, Train_Accuracy_list, '--r', label='Training-accuracy')  # bo为画蓝色圆点，不连线
    plt.plot(epochs_plot, Test_Accuracy_list, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()  # 绘制图例，默认在右上角
    plt.savefig('./result/accuracy.jpg')

    plt.figure()
    plt.plot(epochs_plot, Train_Loss_list, '--r', label='Training loss')
    plt.plot(epochs_plot, Test_Loss_list, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig('./result/loss.jpg')

    plt.show()

