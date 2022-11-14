# Re-Inplement RANSAC Algorithm

import math
import numpy as np
import os
import random


def ransac(num_arrys, sigma=0.25, P=0.99, max_iters=1000000,
           eps=1e-5):
    """
    Args:
        sigma:      数据和模型之间可接受的差值 --> 在这个差值内, 认为是内点
        max_iters:  最大迭代次数, 每次得到更好的估计会优化iters数值
        P:          期望得到 正确模型的概率

    """
    print("Using sigma:{}, P: {} to fit the line".format(sigma, P))

    # y = ax + b
    best_a = 0
    best_b = 0
    pretotal = 0
    iters = max_iters

    length = len(num_arrys)

    for i in range(max_iters):
        # 随机在数据中选择两个点去求解
        sample_index  = random.sample(range(SIZE * 2), 2)
        x_1, y_1 = num_arrys[sample_index[0]]
        x_2, y_2 = num_arrys[sample_index[1]]

        a = (y_2 - y_1) / (x_2 - x_1 + eps)
        b = y_1 - a * x_1

        # 计算内点数目
        total_inlier = 0
        for idx in range(length):
            y_estimate = a * num_arrys[idx][0] + b
            if abs(y_estimate - num_arrys[idx][1]) < sigma:
                total_inlier += 1
        
        # 判断当前模型 是否 优于此前模型?
        if total_inlier > pretotal:
            # 更新 max_iters, 根据估算的内点比例
            # 每次只选了两个点, 所以在分母的log中, 是平方
            iters = math.ceil(math.log(1 - P) / math.log(1 - pow(total_inlier / length, 2)))
            pretotal = total_inlier

            best_a = a
            best_b = b
        
        if (i + 1) % 100000 == 0:
            print("At {} iters, y = {}x + {}".format(i+1, best_a, best_b))
        
        if total_inlier > length // 2 or i > iters:
            print("i: {},  iters: {}".format(i, iters))
            break
    
    print("After {} iters, find the estimation!".format(i + 1))
    print("Current inliers ratio: {}".format(total_inlier / length))
    
    return best_a, best_b


class MeanStd_Calculator:
    def __init__(self) -> None:
        self.n = 0
        self.avg = 0.0
        self.std = 0.0

    
    def incre_with_batch(self, new_list):
        m = len(new_list)
    

    def incre_with_single(self, val):
        avg_new = (self.avg * self.n + val) / (self.n + 1)
        std_new = math.sqrt(
            (self.n * (self.std ** 2) + (avg_new - self.avg) ** 2 + (avg_new - val) ** 2) / (self.n + 1)
        )
        self.avg = avg_new
        self.std = std_new
        self.n += 1


def incremental_mean_std():
    pass


def pnp():
    pass


def gaussian_newton_2order():
    pass


if __name__ == '__main__':
    SIZE = 100
    X = np.linspace(0, 10, SIZE)
    Y = 3 * X + 10

    rand_x, rand_y = [], []
    for i in range(SIZE):
        rand_x.append(X[i] + random.uniform(-0.5, 0.5))
        rand_y.append(Y[i] + random.uniform(-0.5, 0.5))
    
    for i in range(SIZE):
        rand_x.append(X[i] + random.uniform(0, 10))
        rand_y.append(Y[i] + random.uniform(10, 40))
    
    RANDOM_X = np.array(rand_x).reshape(-1, 1)
    RANDOM_Y = np.array(rand_y).reshape(-1, 1)
    
    num_arrays = np.hstack([RANDOM_X, RANDOM_Y])
    # a, b = ransac(num_arrays, sigma=0.25)
    # print("y = {}x + {}".format(a, b))

    the_mean = np.mean(RANDOM_X)
    the_std = np.std(RANDOM_X)
    print("Calculated mean, std by Numpy is : {}, {}".format(the_mean, the_std))

    calculator = MeanStd_Calculator()
    for i in range(RANDOM_X.shape[0]):
        calculator.incre_with_single(RANDOM_X[i])
    print("Incremental mean, std: ", calculator.avg, calculator.std)



