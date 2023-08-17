import numpy as np
import matplotlib.pyplot as plt


def show_moons(X, Y):
    arg0 = np.squeeze(np.argwhere(Y == 0), axis=1)  # axis=0是最外面一层
    arg1 = np.squeeze(np.argwhere(Y == 1), axis=1)  # axis=0是最外面一层
    plt.title("Moons Data")
    plt.scatter(X[arg0, 0], X[arg0, 1], s=100, c='b', marker='+', label='class0')
    plt.scatter(X[arg1, 0], X[arg1, 1], s=40, c='r', marker='o', label='class1')
    plt.legend()
    plt.grid()
    plt.show()
