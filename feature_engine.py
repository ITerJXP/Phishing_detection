# -*- coding:utf-8 -*-

##########################################################
#                       特征工程
#  1. 初始特征重要度排序
#  2. 最优类特征集合排序
#  3. 特征扩大（构建联合特征）
##########################################################

from ml_models import matrix
from conf import feature_name, Init_features_sorted
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame



def dt_feature_sorted():
    """
    过滤型特征排序：信息熵
    feature_sorted = [(0.3899, 'digit_letter_ratio_path'), (0.1409, 'rank_value'), (0.0532, 'has_target_path'),
    (0.0477, 'num_non_alpha_url'), (0.0367, 'is_time'), (0.0315, 'num_dots_url'), (0.0314, 'longest_token_url'),
    (0.0307, 'average_token_url'), (0.0294, 'num_slashes_url'), (0.0241, 'length_url'), (0.021, 'longest_token_path'),
    (0.0206, 'ave_token_path'), (0.018, 'length_domain'), (0.0171, 'longest_token_domain'), (0.017, 'ave_token_domain'),
     (0.0163, 'Is_input_pwd'), (0.0158, 'Num_input'), (0.0158, 'Is_favicon'), (0.0102, 'trigram_domain'),
      (0.0095, 'sensitive_term_url'), (0.0067, 't_scheme'), (0.0038, 'is_hyphen_url'), (0.0033, 'longest_query'),
      (0.0025, 'num_token_domain'), (0.0018, 'length_query'), (0.0015, 'subdomain'), (0.0015, 'num_query'),
      (0.0011, 'num_hyphen_domain'), (0.0006, 'is_at_url'), (0.0002, 'num_token_path'), (0.0001, 'contains_ip_url'),
      (0.0, 'same_target_url'), (0.0, 'is_slash_redir_url'), (0.0, 'bigram_domain')]
    """
    #######################################################################
    # 构建决策树
    from sklearn import tree
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    train_x, test_x, train_y, test_y = matrix(is_poly=False)
    clf.fit(train_x, train_y)
    feature_sorted = sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_), feature_name), reverse=True)
    # print feature_sorted
    # for item in feature_sorted:
    #     print '{},{}'.format(item[0], item[1])

    #######################################################################
    # 画图
    sns.set(style="whitegrid")
    f, ax = plt.subplots(figsize=(15, 10))
    crashes = sns.load_dataset("init_features_sorted").sort_values("info_gain", ascending=False)
    # Plot the total crashes
    sns.set_color_codes("pastel")
    sns.barplot(x="sub_class", y="feature_attr", data=crashes,
                label="Class", color="b")
    # Plot the crashes where alcohol was involved
    sns.set_color_codes("muted")
    sns.barplot(x="info_gain", y="feature_attr", data=crashes,
                label="Info Gain", color="b")
    # Add a legend and informative axis label
    ax.legend(ncol=2, loc="lower right", frameon=True)
    ax.set(xlim=(0, .5), ylabel="",
           xlabel="Features Sorted")
    sns.despine(left=True, bottom=True)
    plt.show()
    return feature_sorted


def rfe_feature_sorted():
    """
    包裹型特征排序：逻辑回归
    feature_sorted = [(1.0, 'num_slashes_url'), (2.0, 'digit_letter_ratio_path'), (3.0, 'has_target_path'),
    (4.0, 'rank_value'), (5.0, 'is_time'), (6.0, 'average_token_url'), (7.0, 'num_query'), (8.0, 'length_query'),
    (9.0, 'subdomain'), (10.0, 'num_token_domain'), (11.0, 'Is_input_pwd'), (12.0, 'ave_token_path'),
    (13.0, 'longest_token_url'), (14.0, 'longest_query'), (15.0, 'length_url'), (16.0, 'longest_token_path'),
    (17.0, 'length_domain'), (18.0, 'Is_favicon'), (19.0, 'trigram_domain'), (20.0, 'ave_token_domain'),
    (21.0, 'sensitive_term_url'), (22.0, 'num_token_path'), (23.0, 'is_hyphen_url'), (24.0, 'is_at_url'),
    (25.0, 'longest_token_domain'), (26.0, 'Num_input'), (27.0, 'num_dots_url'), (28.0, 'num_hyphen_domain'),
    (29.0, 'contains_ip_url'), (30.0, 'num_non_alpha_url'), (31.0, 't_scheme'), (32.0, 'is_slash_redir_url'),
    (33.0, 'bigram_domain'),  (34.0, 'same_target_url')]
    """
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LinearRegression
    train_x, test_x, train_y, test_y = matrix(is_poly=False)
    lr = LinearRegression()
    rfe = RFE(lr, n_features_to_select=1)
    rfe.fit(train_x, train_y)
    feature_sorted = sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), feature_name))
    # print feature_sorted
    return feature_sorted


if __name__ == '__main__':
    # dt_feature_sorted()
    rfe_feature_sorted()