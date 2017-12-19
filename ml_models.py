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
    return false_positive_rate, recall, roc_auc


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
    clf = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto', beta_1=0.9,
                        beta_2=0.999, early_stopping=False, epsilon=1e-08,
                        hidden_layer_sizes=(5, 2), learning_rate='constant',
                        learning_rate_init=0.001, max_iter=200, momentum=0.9,
                        nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
                        solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
                        warm_start=False)


def stacking():
    pass


def roc_auc():
    pass
    # plt.title('ROC & AUC')
    # plt.plot(lr_false_positive_rate, lr_recall, 'b', label='AUC = %0.2f' % lr_roc_auc)
    # plt.plot(rf_false_positive_rate, rf_recall, 'b', label='AUC = %0.2f' % rf_roc_auc)
    # plt.plot(ada_false_positive_rate, ada_recall, 'b', label='AUC = %0.2f' % ada_roc_auc)
    # plt.plot(nn_false_positive_rate, nn_recall, 'b', label='AUC = %0.2f' % nn_roc_auc)
    #
    # plt.legend(loc='lower right')
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.0])
    # plt.ylabel('Recall')
    # plt.xlabel('precision')
    # plt.show()

if __name__ == '__main__':
    # logistic_regression()
    # random_forest()
    adaboost()
    # neural_nets()
    # stacking()
    # roc_auc()