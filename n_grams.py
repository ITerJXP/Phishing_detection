# -*- coding:utf-8 -*-

import sys
reload(sys)
from nltk.util import ngrams
import collections


def make_str(url_part_list):
    _tab_ = ','
    _str_ = ''
    for p in url_part_list:
        _str_ = _str_ + _tab_ + p
    return _str_

def train_grams(_str_, n):
    """
    计算n-gram集合
    :param _str_:   url集合
    :param n:       n-gram
    :return:        计数集合

    # e.g.:
    # url = 'www.paypal.com'
    # ∆=[('w',), ('w',), ('w',), ('.',), ('p',), ('a',), ('y',), ('p',), ('a',), ('l',), ('.',), ('c',), ('o',), ('m',)]
    # Counter({('w',): 3, ('.',): 2, ('a',): 2, ('p',): 2, ('c',): 1, ('m',): 1, ('l',): 1, ('y',): 1, ('o',): 1})
    # theta = []
    # for d in delta:
    #     theta.append(delta_collect[d])        # ø = [3, 3, 3, 2, 2, 2, 1, 2, 2, 1, 2, 1, 1, 1]
    """
    delta = list(ngrams(_str_, n))  # ∆
    delta_clt = collections.Counter(delta)
    return delta_clt


def search(item, pre_delta_clt):
    count = 0
    for i in pre_delta_clt:
        if item == i:
            count = pre_delta_clt[i]
    return count


def unigram(url, this_delta_clt):
    P = []
    sum_p = 0
    _sum = sum(this_delta_clt.values())
    for item in this_delta_clt:
        p = float(this_delta_clt[item] / _sum)
        P.append(p)

    for i in P:
        sum_p = sum_p + i
    sim = float(sum_p/len(url))
    return sim


def n_grams(url, this_delta_clt, pre_delta_clt, n_gram):
    """
    计算url某部分的similarity
    :param _str_:               某个url的部分
    :param self_delta_clt:      训练过的n_gram计数集合
    :param pre_delta_clt:       训练过的(n-1)_gram计数集合
    :return: similarity
    """
    P = []
    sum_p = 0
    length = len(url) - n_gram + 1

    for item in this_delta_clt:
        # 查找前一状态相同项的值
        count = search(item[0:-1], pre_delta_clt)
        p = float(this_delta_clt[item]/count)
        P.append(p)

    for i in P:
        sum_p = sum_p + i
    try:
        sim = float(sum_p)/length
    except Exception, e:
        sim = 0
    return sim


if __name__ == '__main__':
    trainset = ['aaaa', 'aaabbb', 'aaaabbbbbcccc']
    _str_ = make_str(trainset)
    uni_delta_clt = train_grams(_str_, 1)
    bi_delta_clt = train_grams(_str_, 2)
    tri_delta_clt = train_grams(_str_, 3)
    qua_delta_clt = train_grams(_str_, 4)

    # print sum(uni_delta_clt.values())
    # print bi_delta_clt

    url = ''
    Sim = n_grams(url, this_delta_clt=bi_delta_clt, pre_delta_clt=uni_delta_clt, n_gram=2)
    # print Sim