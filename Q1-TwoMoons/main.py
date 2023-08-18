import sklearn.datasets  # 数据集
from sklearn.datasets import make_moons  # 数据集、
import torch
import numpy as np
import matplotlib.pyplot as plt
import model
import utils
import torch.nn as nn
from sklearn.metrics import accuracy_score
import os

print(torch.cuda.is_available())
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


"""数据导入"""
np.random.seed(10)  # 设置随机种子
X, Y = make_moons(n_samples=200, noise=0.2)
utils.show_moons(X, Y) # 双月图像展示
"转化为张量"
# 张量可以理解为多维矩阵
xt = torch.from_numpy(X).type(torch.FloatTensor)
yt = torch.from_numpy(Y).type(torch.LongTensor)


"""模型准备"""
my_model = model.LogicNet(inputdim=2, hiddendim=3, outputdim=2)  # 实例化模型
criterion = nn.CrossEntropyLoss()  # 定义交叉熵函数
optimizer = torch.optim.Adam(my_model.parameters(), lr=1e-2)  # 定义优化器


"""训练"""
epochs = 5000
losses = []
for i in range(epochs):

    y_pred = my_model(xt)
    loss = criterion(y_pred,yt)
    losses.append(loss.item()) # item() 取出单元素张量元素值并返回，保持该元素类型不变

    optimizer.zero_grad() # 清空之前梯度
    loss.backward() # 反向传播损失值
    optimizer.step() # 更新参数

    # 这里思路不好，因为不能跟随最小loss更新模型，所以，会偏掉

"""预测与结果评价"""
print(accuracy_score(my_model.predict(xt),yt))

"""可视化训练结果与模型"""
utils.plot_loss(losses)
utils.plot_decision_boundary(my_model, X, Y)


a = 1
