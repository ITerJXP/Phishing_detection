# -*- coding:utf-8 -*-

################################################################
#                      cache_data 数据集文件
################################################################
# phish源数据（csv表）
PHISH_CSV = '/Users/JasonJay/programming/workspace/Python_anaconda2.4/Phishing_Detction/cache_data/initial_phish_info.csv'
# 正常url数据
NORMAL_URL = '/Users/JasonJay/programming/workspace/Python_anaconda2.4/Phishing_Detction/cache_data/benign_url_train.txt'
# 钓鱼url数据
PHISH_URL = '/Users/JasonJay/programming/workspace/Python_anaconda2.4/Phishing_Detction/cache_data/phish_url_train.txt'
# 带标签的所有url
ALL_LABEL_URL = '/Users/JasonJay/programming/workspace/Python_anaconda2.4/Phishing_Detction/cache_data/sum_url_labeled.txt'
# 钓鱼特征数据
PHISH_FEATURES = '/Users/JasonJay/programming/workspace/Python_anaconda2.4/Phishing_Detction/cache_data/phish_features.txt'
# 临时存储文件
CACHE_FILE = '/Users/JasonJay/programming/workspace/Python_anaconda2.4/Phishing_Detction/1489_train.txt'



#################################################################
#                       data_sets 特征文件
#################################################################
# 训练集
TRAIN_SET = '/Users/JasonJay/programming/workspace/Python_anaconda2.4/Phishing_Detction/data_sets/train_set.txt'
# 测试集
TEST_SET = '/Users/JasonJay/programming/workspace/Python_anaconda2.4/Phishing_Detction/data_sets/test_set.txt'
# phishing 训练集
PHISH_TRAIN = '/Users/JasonJay/programming/workspace/Python_anaconda2.4/Phishing_Detction/data_sets/phish_train_featureMatrix.txt'
# phishing 测试集
PHISH_TEST = '/Users/JasonJay/programming/workspace/Python_anaconda2.4/Phishing_Detction/data_sets/phish_test_featureMatrix.txt'
# benign 训练集
BENIGN_TRAIN = '/Users/JasonJay/programming/workspace/Python_anaconda2.4/Phishing_Detction/data_sets/' \
               'benign_train_featureMatrix.txt'
# 初始特征排序
Init_features_sorted = '/Users/JasonJay/seaborn-data/init_features_sorted.csv'

#######################################################
#                          常量
#######################################################
feature_name = ['length_url', 'num_dots_url', 'num_slashes_url', 'is_at_url', 'is_hyphen_url', 'is_slash_redir_url',
                'contains_ip_url', 'sensitive_term_url', 'num_non_alpha_url', 'longest_token_url', 'average_token_url',
                'same_target_url', 't_scheme', 'length_domain', 'num_token_domain', 'longest_token_domain',
                'ave_token_domain', 'bigram_domain', 'trigram_domain', 'num_hyphen_domain', 'subdomain',
                'num_token_path', 'longest_token_path', 'digit_letter_ratio_path', 'has_target_path',
                'length_query', 'num_query', 'longest_query', 'is_time', 'rank_value', 'num_input', 'Is_input_pwd',
                'Is_favicon', 'hyphen_domain', 'at_domain', 'dots_domain', 'len_path', 'ave_token_path', 'hyphen_query',
                'at_query', 'dots_query']