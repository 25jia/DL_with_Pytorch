import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt


class LogicNet(nn.Module):
    def __init__(self, inputdim, hiddendim, outputdim):
        super(LogicNet, self).__init__()  # 初始化一部分父类属性
        self.Linear1 = nn.Linear(in_features=inputdim, out_features=hiddendim, bias=True)  # 定义全连接层1
        self.Linear2 = nn.Linear(in_features=hiddendim, out_features=outputdim, bias=True)  # 定义全连接层2
        self.criterion = nn.CrossEntropyLoss()  # 定义交叉熵函数

    def forword(self, x):
        x = self.Linear1(x)
        x = torch.tanh(X)
        x = self.Linear2(x)

    def predict(self, x):
        pred = torch.softmax(self.forword(x), dim=1)
        return torch.argmax(pred, dim=1) # 返回每一组预测中最大值的索引，dim=1相当于组数

    def calculate_loss(self,x,y):
        y_pred = self.forword(x)
        loss = self.criterion(y_pred,y)
        return loss

