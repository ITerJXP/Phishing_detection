# -*- coding:utf-8 -*-

###################################
#         模型构建流程              #
# 1.数据预处理：数据清洗；归一化      #
# 2.训练模型                       #
# 3.交叉验证                       #
# 4.模型测试                       #
###################################

from conf import TRAIN_SET, TEST_SET
from data_preprocess import to_matrix, poly_features, standardize, label
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cross_validation import cross_val_score


def matrix(is_poly=True):
    """
    加入poly后的特征矩阵
    return: 训练集 & 测试集
    """
    train_matrix = to_matrix(TRAIN_SET)
    test_matrix = to_matrix(TEST_SET)
    train_y, test_y = label()
    if is_poly:
        train_ = poly_features(init_matrix=train_matrix, degree=2, start_index=1, end_index=4)
        train_x, test_x = standardize(train_, test_matrix)
        return train_x, test_x, train_y, test_y
    else:
        train_x, test_x = standardize(train_matrix, test_matrix)
        return train_x, test_x, train_y, test_y


def logistic_regression():
    '''
    最优参数组合：C:0.1; max_iter:50; penalty:'l2'
    准确率： 0.882913276444
    召回率： 0.861184210526
    F1   ： 0.886
    :return: cv=5 轮交叉验证的平均值
    '''
    from sklearn.linear_model.logistic import LogisticRegression

    train_x, test_x, train_y, test_y = matrix(is_poly=False)
    classifier = LogisticRegression()
    classifier.fit(train_x, train_y)

    # score = cross_val_score(classifier, train_x, train_y, cv=5)
    # print u'准确率：', np.mean(score), score
    # precisions = cross_val_score(classifier, train_x, train_y, cv=5, scoring='precision')
    # print u'精确率：', np.mean(precisions), precisions
    # recalls = cross_val_score(classifier, train_x, train_y, cv=5, scoring='recall')
    # print u'召回率：', np.mean(recalls), recalls
    # f1 = cross_val_score(classifier, train_x, train_y, cv=5, scoring='f1')
    # print u'f1   ：', np.mean(f1), f1

    def plot():
        predictions = classifier.predict_proba(test_x)  # 每一类的概率
        false_positive_rate, recall, thresholds = roc_curve(test_y, predictions[:, 1])
        roc_auc = auc(false_positive_rate, recall)
        plt.title('logistic regression: AUC')
        plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.ylabel('Recall')
        plt.xlabel('precision')
        plt.show()
    plot()

    # 网格计算最优参数
    def best_params():
        # pipeline = Pipeline([('clf', LogisticRegression())])
        parameters = {
            'penalty': ('l1', 'l2'),
            'C': (0.01, 0.1, 1, 10),
            'max_iter': (50, 100, 200)
        }
        scores = ['precision', 'recall', 'f1']

        for score in scores:
            grid_search = GridSearchCV(classifier, parameters, scoring=score, cv=5)
            grid_search.fit(train_x, train_y)
            print '最佳 %s 得分：%0.3f' % (score, grid_search.best_score_)
            print '最优参数组合：'
            best_parameters = grid_search.best_estimator_.get_params()
            for param_name in sorted(parameters.keys()):
                print '\t%s:%r' % (param_name, best_parameters[param_name])

            for params, mean_score, scores in grid_search.grid_scores_:
                print '%s:\t%0.3f for %r' % (score, mean_score, params)
    best_params()


def random_forest():
    from sklearn.ensemble import RandomForestClassifier
    train_x, test_x, train_y, test_y = matrix(is_poly=False)
    parameters = {'max_features':range(10, 45, 5)}
    grid_search = GridSearchCV(estimator=RandomForestClassifier(n_estimators=60, max_depth=30, min_samples_split=20,
                                                             min_samples_leaf=5, oob_score=True, random_state=10),
                            param_grid=parameters, scoring='roc_auc', iid=False, cv=5)
    grid_search.fit(train_x, train_y)
    for params, mean_score, scores in grid_search.grid_scores_:
        print '%0.5f for %r' % (mean_score, params)


def adaboost():
    from sklearn.ensemble import AdaBoostClassifier
    train_x, test_x, train_y, test_y = matrix(is_poly=False)
    parameters = {'n_estimators': range(75, 100, 5),
                  'learning_rate': np.arange(0.2, 1.8, 0.2)
                  }
    grid_search = GridSearchCV(estimator=AdaBoostClassifier(),
                            param_grid=parameters, scoring='roc_auc', iid=False, cv=5)
    grid_search.fit(train_x, train_y)
    for params, mean_score, scores in grid_search.grid_scores_:
        print '%0.5f for %r' % (mean_score, params)


def neural_nets():
    from sklearn.neural_network import MLPClassifier
    train_x, test_x, train_y, test_y = matrix(is_poly=False)
    parameters = {'learning_rate_init': []}
    classifier = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto', beta_1=0.9, beta_2=0.999,
                               early_stopping=False, epsilon=1e-08,hidden_layer_sizes=(20,20), learning_rate='constant',
                               learning_rate_init=0.001, max_iter=200, momentum=0.9, nesterovs_momentum=True,
                               power_t=0.5, random_state=1, shuffle=True, solver='adam', tol=0.0001,
                               validation_fraction=0.1, verbose=False, warm_start=False)
    grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='roc_auc')
    grid_search.fit(train_x, train_y)
    for params, mean_score, scores in grid_search.grid_scores_:
        print '%0.5f for %r' % (mean_score, params)


def stacking():
    pass


def roc_auc():
    from sklearn.linear_model.logistic import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.neural_network import MLPClassifier

    train_x, test_x, train_y, test_y = matrix(is_poly=False)

    lr = LogisticRegression(penalty='l2', C=0.1, max_iter=50)
    lr.fit(train_x, train_y)
    predictions = lr.predict_proba(test_x)  # 每一类的概率
    lr_false_positive_rate, lr_recall, lr_thresholds = roc_curve(test_y, predictions[:, 1])
    lr_roc_auc = auc(lr_false_positive_rate, lr_recall)
    lr_pre = lr.predict(test_x)

    rf = RandomForestClassifier(n_estimators=50, max_depth=30, min_samples_split=2, min_samples_leaf=5, max_features=15)
    rf.fit(train_x, train_y)
    predictions = rf.predict_proba(test_x)  # 每一类的概率
    rf_false_positive_rate, rf_recall, rf_thresholds = roc_curve(test_y, predictions[:, 1])
    rf_roc_auc = auc(rf_false_positive_rate, rf_recall)
    rf_pre = rf.predict(test_x)

    ada = AdaBoostClassifier(n_estimators=95, learning_rate=1.0)
    ada.fit(train_x, train_y)
    predictions = ada.predict_proba(test_x)  # 每一类的概率
    ada_false_positive_rate, ada_recall, ada_thresholds = roc_curve(test_y, predictions[:, 1])
    ada_roc_auc = auc(ada_false_positive_rate, ada_recall)
    ada_pre = ada.predict(test_x)

    nn = MLPClassifier(hidden_layer_sizes=(20,20))
    nn.fit(train_x, train_y)
    predictions = nn.predict_proba(test_x)  # 每一类的概率
    nn_false_positive_rate, nn_recall, nn_thresholds = roc_curve(test_y, predictions[:, 1])
    nn_roc_auc = auc(nn_false_positive_rate, nn_recall)
    nn_pre = nn.predict(test_x)

    # print 'lr: %.4f' % f1_score(test_y, lr_pre)    # 0.8987
    # print 'rf: %.4f' % f1_score(test_y, rf_pre)     # 0.9500
    # print 'ada: %.4f' % f1_score(test_y, ada_pre)       # 0.9137
    # print 'nn: %.4f' % f1_score(test_y, nn_pre)     # 0.9288

    def plot():
        plt.plot(lr_false_positive_rate, lr_recall, 'b-', label='logistic regression AUC: %0.3f' % lr_roc_auc)
        plt.plot(rf_false_positive_rate, rf_recall, 'g-', label='random forest AUC: %0.3f' % rf_roc_auc)
        plt.plot(ada_false_positive_rate, ada_recall, 'r-', label='adaboost AUC: %0.3f' % ada_roc_auc)
        plt.plot(nn_false_positive_rate, nn_recall, 'c-', label='neural network AUC: %0.3f' % nn_roc_auc)
        # plt.title('ROC & AUC')
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'm--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.ylabel('Recall')
        plt.xlabel('False positive rate')
        plt.show()
    plot()


def joint_feature_ratio():
    '''
    联合特征率：四种模型的F1
    :return: None
    '''
    ratio = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    lr = [0.8987, 0.8993, 0.9001, 0.8989, 0.9012, 0.9010, 0.9004, 0.9027, 0.9012, 0.9001, 0.9020]
    rf = [0.9500, 0.9521, 0.9588, 0.9602, 0.9628, 0.9645, 0.9643, 0.9648, 0.9642, 0.9645, 0.9650]
    ada = [0.9137, 0.9192, 0.9274, 0.9292, 0.9287, 0.9289, 0.9287, 0.9292, 0.9285, 0.9291, 0.9288]
    nn = [0.9288, 0.9301, 0.9312, 0.9311, 0.9354, 0.9372, 0.9418, 0.9439, 0.9501, 0.9567, 0.9589]

    plt.plot(ratio, lr, 'bo-', label='logistic regression F1')
    plt.plot(ratio, rf, 'g^-', label='andom forest F1')
    plt.plot(ratio, ada, 'r*-', label='adaboost F1')
    plt.plot(ratio, nn, 'co--', label='neural network F1')
    # plt.title('joint_feature_ratio')
    plt.legend(loc='lower right')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.86, 0.98])
    plt.ylabel('F1 score')
    plt.xlabel('joint_feature_ratio')
    plt.show()


def test():
    n = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]
    r = [0.9655,0.9641,0.9643,0.9652,0.9656,0.9660,0.9645,0.9682,0.9664,0.9685,0.9656,0.9666,0.9641,0.9636,0.9631]
    plt.plot(n, r, 'ko-')
    plt.ylim([0.961, 0.970])
    plt.ylabel('ROC_AUC')
    plt.xlabel('the number of cells in hidden layer 2')
    plt.show()

if __name__ == '__main__':
    # logistic_regression()
    # random_forest()
    # adaboost()
    # neural_nets()
    # stacking()
    # roc_auc()
    joint_feature_ratio()
    # test()