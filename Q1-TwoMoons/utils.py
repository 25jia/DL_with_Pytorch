import numpy as np
import matplotlib.pyplot as plt
import torch


def show_moons(X, Y):
    arg0 = np.squeeze(np.argwhere(Y == 0), axis=1)  # axis=0是最外面一层
    arg1 = np.squeeze(np.argwhere(Y == 1), axis=1)  # axis=0是最外面一层
    plt.title("Moons Data")
    plt.scatter(X[arg0, 0], X[arg0, 1], s=100, c='b', marker='+', label='class0')
    plt.scatter(X[arg1, 0], X[arg1, 1], s=40, c='r', marker='o', label='class1')
    plt.legend()
    plt.grid()
    plt.show()


def moving_average(a, w=10):
    """
    对a以w步长取平均
    :param a: 数字队列
    :param w: 整型步长
    :return: 结果序列
    """
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[idx - w:idx]) / w for idx, val in enumerate(a)]


def plot_loss(losses):
    avgloss = moving_average(losses)
    plt.figure(1)
    plt.subplot(211)
    plt.plot(range(len(avgloss)), avgloss, 'b--')
    plt.xlabel('Step number')
    plt.ylabel('Training loss')
    plt.title('Step number vs. Training loss')
    plt.show()


def plot_decision_boundary(model, X, Y):

    # 计算取值范围
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    step = 0.01

    # 生成网格矩阵
    mesh_x, mesh_y = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    mesh_X = np.c_[mesh_x.ravel(), mesh_y.ravel()]

    # 对网格进行预测
    mesh_Xt = torch.from_numpy(mesh_X).type(torch.FloatTensor)
    mesh_Yt = model.predict(mesh_Xt) # 输入和输出是tensor形式
    mesh_Y = mesh_Yt.numpy()
    mesh_Y = mesh_Y.reshape(mesh_x.shape)

    # 画图
    plt.contourf(mesh_x,mesh_y,mesh_Y,cmap=plt.cm.Spectral)
    plt.title("Linear predict")
    arg0 = np.squeeze(np.argwhere(Y == 0), axis=1)  # axis=0是最外面一层
    arg1 = np.squeeze(np.argwhere(Y == 1), axis=1)  # axis=0是最外面一层
    plt.scatter(X[arg0, 0], X[arg0, 1], s=100, c='b', marker='+', label='class0')
    plt.scatter(X[arg1, 0], X[arg1, 1], s=40, c='r', marker='o', label='class1')
    plt.legend()
    plt.grid()
    plt.show()

