#########################   Phishing_Detction    ##############################
#########################                        ##############################
# #                                                                          ##
# #   It's a tool can detect phishing website online. 一个在线钓鱼网页检测工具  ##
# #                                                                          ##
###############################################################################
###############################################################################



######################## 目录说明 ###########################
# cache_data   目录
    # initial_phish_info.csv    原始钓鱼网页数据
    # benign_url.txt            15598行
    # sum_url_labeled.txt       39282行
    # phish_features.txt        phish特征文件
    # bengin_url_train.txt      正常网页（训练数据+交叉验证）:16670
    # phish_url_train.txt       钓鱼网页（训练数据+交叉验证）:20000
    # benign_url_test.txt       正常网页（测试集）：待建
    # phish_url_test.txt        钓鱼网页（测试集）：4739
    # cache.txt                 临时文件

# data_sets    目录
    # benign_trainset.txt       正常网页特征集合
    # phish_testset.txt         钓鱼网页测试特征集合  *
    # phish_trainset.txt        钓鱼网页训练特征集合
    # trainset.txt              正常&钓鱼 训练集合   *


####################### .py文件说明 #########################

# conf.py                  配置文件
# csvToData.py             从phishTank的csv文件提取钓鱼url
# rank.py                  实验 - alexa rank

# n_gram.py                接口 - 词袋模型
# get_features.py          接口 - 特征提取
# get_web_content.py       接口 - 网页内容类
# get_whois.py             接口 - 网页whois信息

# data_preprocess          接口 - 数据预处理（特征归一化等）
# ml_models                接口 - 机器学习算法
# feature_engine           接口 - 特征工程

