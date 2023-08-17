import sklearn.datasets  # 数据集
from sklearn.datasets import make_moons  # 数据集、
# from sklearn.datasets import make_ # 数据集
import torch
import numpy as np
import matplotlib.pyplot as plt
import model

"""数据导入"""
np.random.seed(10)  # 设置随机种子
X, Y = make_moons(n_samples=200, noise=0.2)

"展示"
arg0 = np.squeeze(np.argwhere(Y == 0), axis=1) # axis=0是最外面一层
arg1 = np.squeeze(np.argwhere(Y == 1), axis=1) # axis=0是最外面一层
plt.title("Moons Data")
plt.scatter(X[arg0,0],X[arg0,1], s=100, c='b',marker='+',label='class0')
plt.scatter(X[arg1,0],X[arg1,1], s=40, c='r',marker='o',label='class1')
plt.legend()
plt.grid()
plt.show()

"""模型实例化"""
mymodel=model.LogicNet(inputdim=2,hiddendim=3,outputdim=2) # 实例化模型

optimizer=torch.optim.Adam(mymodel.parameters(),lr=0.01) # 定义优化器

a = 1
