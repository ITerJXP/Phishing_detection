# -*- coding:utf-8 -*-

#########################################################################
# 1. 缺失值填补: sed -i "" "s/oldstring/newstring/g" filename
# 2. MaxAbsScaler, training data lies within the range [-1, 1]
#    by dividing the largest maximum value in each feature
#########################################################################

import sys
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from conf import PHISH_TRAIN, PHISH_TEST, BENIGN_TRAIN, TRAIN_SET


def to_matrix(data_file):
    """
    - 特征文件 ——> 特征矩阵
    :param data_file: 特征文件路径
    :return: 特征矩阵
    """
    hole_list = []
    for line in open(data_file).readlines():
        line_list = []
        feature = line.strip().split()[1:]
        for i in feature:
            line_list.append(float(i))
        hole_list.append(line_list)
    matrix = np.array(hole_list)
    return matrix


def standardize(train, test):
    """
    - 在train集上做标准化后，用同样的标准化器去标准化test集
    params: 训练集，测试集
    return: 标准化后的的矩阵
    """
    scaler = StandardScaler().fit(train)
    standard_train = scaler.transform(train)
    standard_test = scaler.transform(test)
    return standard_train, standard_test


def poly_features(init_matrix, degree, start_index, end_index):
    """
    - 线性特征——>非线性特征；
    - 特征下标范围：1 ~ 34
    :param data_matrix:
    :return: 多项式特征 or 交叉特征
    """
    # 根据特征区间，选择交叉特征范围
    matrix = init_matrix[:, start_index:end_index]
    # 特征交叉
    poly = PolynomialFeatures(degree=degree, interaction_only=True)
    cross_matrix = poly.transform(matrix)
    # 特征矩阵合并
    final_matrix = np.hstack((init_matrix, cross_matrix))
    return final_matrix


def label():
    train_y = []
    test_y = []
    for i in range(16664):
        train_y.append(0)
    for j in range(16801):
        train_y.append(1)
    for k in range(3040):
        test_y.append(1)
    return train_y, test_y


if __name__ == '__main__':
    train_matrix = to_matrix(TRAIN_SET)
    test_matrix = to_matrix(PHISH_TEST)
    train_ = poly_features(init_matrix=train_matrix, degree=2, start_index=1, end_index=4)
    train_X, test_X = standardize(train_, test_matrix)