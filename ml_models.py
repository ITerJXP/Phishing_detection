# -*- coding:utf-8 -*-

###################################
#         模型构建流程              #
# 1.数据预处理：数据清洗；归一化      #
# 2.训练模型                       #
# 3.交叉验证                       #
# 4.模型测试                       #
###################################

from conf import TRAIN_SET, PHISH_TEST
from data_preprocess import to_matrix, poly_features, standardize, label


def get_matrix():
    """
    return: 训练集 & 测试集
    """
    train_matrix = to_matrix(TRAIN_SET)
    test_matrix = to_matrix(PHISH_TEST)
    train_ = poly_features(init_matrix=train_matrix, degree=2, start_index=1, end_index=4)
    train_x, test_x = standardize(train_, test_matrix)
    train_y, test_y = label()
    return train_x, test_x, train_y, test_y


def logistic_Rregression():
    pass


def decision_tree():
    pass


def k_means():
    pass


def xgboost():
    pass


def mixed_model():
    pass


def nenual_nets():
    pass


if __name__ == '__main__':
    pass