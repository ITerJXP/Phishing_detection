# -*- coding: utf-8 -*-

import sys
import re
from urlparse import urlparse
from n_grams import make_str, train_grams, unigram, n_grams
from get_web_content import get_content
from get_whois import _whois
from conf import NORMAL_URL, PHISH_URL, CACHE_FILE, PHISH_FEATURES
import urlparse
import Levenshtein
from tld import get_tld
import urllib2
from multiprocessing import Pool
import multiprocessing
from decimal import getcontext

#############################################################################
#                           特征提取流程解释                                  #
#   1. 获取 URL 各部分内容，包括六大部分                                        #
#       （Scheme, Hostname, Domain, Path, Param, Query）                    #
#   2. 创建 URL 类，实现各种特征方法                                           #
#   3. 所有bool类型的特征：有为1；无为0                                        #
#############################################################################

# 全局变量
# sys.path.append('/Users/JasonJay/programming/workspace/Python_anaconda2.4/Phishing_Detction')
target_list = ['adobe', 'alibaba', 'amazon', 'apple', 'baidu', 'dropbox', 'ebay',
               'tencent', 'google', 'qq', 'microsoft', 'outlook', 'facebook', 'paypal', 'yahoo',
               'webscr', 'secure', 'banking', 'ebayisapi', 'account', 'confirm','login', 'signin',
               'free', 'lucky', 'bonus index', 'includes', 'content', 'images', 'admin', 'file doc',
               'account', 'update', 'confirm', 'verify', 'secur', 'notif', 'log', 'click', 'inconvenien',
               'urgent', 'alert']
# getcontext().prec = 3


def get_url_parts(url):
    '''
        提取url各部分
    :param url: 完整url
    :return: 六大部分（Scheme, Hostname, Domain, Path, Param, Query）
    '''
    result = urlparse.urlparse(url)
    scheme = result.scheme
    hostname = result.netloc
    try:
        domain = get_tld(url)
    except Exception, e:
        domain = hostname
    path = result.path
    param = result.params
    query = result.query
    return scheme, hostname, domain, path, param, query


def train_n_grams(trainset):
    '''
        训练词袋模型
    :param trainset: 指定结构的url集合
    :return: 四种n-gram
    '''
    print >> sys.stderr, 'training is start...'
    _str_ = make_str(trainset)
    uni_delta_clt = train_grams(_str_, 1)
    bi_delta_clt = train_grams(_str_, 2)
    tri_delta_clt = train_grams(_str_, 3)
    # qua_delta_clt = train_grams(_str_, 4)
    return uni_delta_clt, bi_delta_clt, tri_delta_clt


class URL:
    def __init__(self, url):
        self.url = url
        self.contains_dots = '.'
        self.contains_at = '@'
        self.hyphen = '-'
        self.slashes_url = '/'
        self.slash_reidr = '//'
        self.token = '[?_.=&-/]'

    def url_self_features(self):
        '''
            url基本特征
        :return: URL总长度，'.'数，'/'数，是否有'@'，是否有'-'，是否有'//'
        '''
        length_url = len(self.url)  # url长度
        num_dots_url = self.url.count(self.contains_dots)   # 包括'.'数
        num_slashes_url = self.url.count(self.slashes_url)  # 包括'/'数
        contains_at = self.url.find(self.contains_at) != -1 # 是否包括'@''
        if contains_at == True:is_at = 1
        else:is_at = 0
        hyphen_url = self.url.find(self.hyphen) != -1   # 是否包括 '-'
        if hyphen_url == True:is_hyphen = 1
        else:is_hyphen = 0
        contains_slash_redir = self.url.find(self.slash_reidr) != -1  # 是否包括'//'
        if contains_slash_redir == True:is_slash_redir = 1
        else:is_slash_redir = 0
        return length_url, num_dots_url, num_slashes_url, is_at, is_hyphen, is_slash_redir

    def bag_of_word(self, this_delta, pre_delta, N):
        """
            bag_of_word模型
        :param this_delta: 当前状态为 M 时的deltaz值
        :param pre_delta: 当前状态为 m-1 时的deltaz值
        :param N: n-gram
        :return: 相似值
        """
        sim = n_grams(self.url, this_delta_clt=this_delta, pre_delta_clt=pre_delta, n_gram=N)
        return sim

    def ip_exist_ip(self):      # 是否包括ip地址
        compile_rule = re.compile(r'\d+[\\.]\d+[\\.]\d+[\\.]\d+')
        match_ip = re.findall(compile_rule, self.url)       
        if match_ip:
            return 1
        else:
            return 0

    def has_sensitive_terms(self):      # 是否包括敏感词汇
        bool_list = [i in self.url for i in target_list]
        if True in bool_list:
            return 1
        else:
            return 0

    def num_non_alpha_url(self):        # 非字母个数
        num_non_alpha = 0
        for s in self.url:
            if not s.isalpha():
                num_non_alpha += 1
        return num_non_alpha

    def tokens(self):    # URL的token相关计算
        '''
        :return: 最长token, 平均token
        '''
        url_token = [x for x in re.split(self.token, self.url)]
        num_url_tokens = len(url_token)  # url_token数
        longest_url_tokens = max(len(i) for i in url_token)  # domain token最大长度
        length_token = 0
        for i in url_token:
            length_token += len(i)
        average_url_tokens = float(length_token) / num_url_tokens  # domain tokens平均长度

        def same_target():
            for token in url_token:
                for target in target_list:
                    if 0 <= Levenshtein.distance(token, target) < 3:
                        return 1
            return 0
        has_sametarget = same_target()
        return longest_url_tokens, average_url_tokens, has_sametarget


def type_scheme(scheme):
    '''
        https or http
    :param scheme
    :return: https(1) or http(0)
    '''
    if scheme == 'https':
        return 1
    else:
        return 0


class Hostname:
    def __init__(self, hostname):
        self.hostname = hostname
        self.token = '[?_.=&-/]'

    def length(self):
        length_hn = len(self.hostname)  # url长度
        return length_hn

    def tokens(self):
        '''
        :return: 最长token, 平均token
        '''
        hn_token = [x for x in re.split(self.token, self.hostname)]
        num_url_tokens = len(hn_token)  # url_token数
        longest_url_tokens = max(len(i) for i in hn_token)  # domain token最大长度
        length_token = 0
        for i in hn_token:
            length_token += len(i)
        average_url_tokens = float(length_token) / num_url_tokens  # domain tokens平均长度
        return longest_url_tokens, average_url_tokens

    def bag_of_word(self, this_delta, pre_delta, N):
        if this_delta == pre_delta:
            sim = unigram(self.hostname, this_delta)
        else:
            sim = n_grams(self.hostname, this_delta_clt=this_delta, pre_delta_clt=pre_delta, n_gram=N)
        return sim

    def num_chars_after_target(self):   # target后的字符数
        num_chars = -1
        for target in target_list:
            if target in self.hostname:
                after = self.hostname.split(target)[-1]
                num_chars = num(after)
                break
        return num_chars


class Domain:
    def __init__(self, domain):
        self.domain = domain
        self.contains_dots = '.'
        self.contains_at = '@'
        self.contains_hyphen = '-'

    def length(self):
        length_domain = len(self.domain)  # url长度
        return length_domain

    def tokens(self):  # token数
        token_list = self.domain.split('.')
        num_token = len(token_list)
        longest_token = max(len(i) for i in token_list)
        length_token = 0
        for i in token_list:
            length_token += len(i)
        ave_token = float(length_token) / len(token_list)  # domain tokens平均长度
        return num_token, longest_token, ave_token

    def bag_of_word(self, this_delta, pre_delta, N):
        sim = n_grams(self.domain, this_delta_clt=this_delta, pre_delta_clt=pre_delta, n_gram=N)
        return sim

    def domain_characters(self):
        hyphen_domain = self.domain.find(self.contains_hyphen) != -1
        if hyphen_domain == True:hyphen_domain = 1
        else:hyphen_domain = 0
        at_domain = self.domain.find(self.contains_at) != -1
        if at_domain == True:at_domain = 1
        else:at_domain = 0
        dots_domain = self.domain.find(self.contains_dots) != -1
        if dots_domain == True:dots_domain = 1
        else: dots_domain = 0

        num_hyphen = 0  # '-'数
        for i in self.domain:
            if i == '-':
                num_hyphen += 1
        return hyphen_domain, num_hyphen, at_domain, dots_domain

    def subdomain(self):
        if len(self.domain.strip().split('.')) > 2:
            return 1
        else:
            return 0


class Path:
    def __init__(self, path):
        self.path = path

    def tokens(self):
        len_path = len(self.path)
        token_list = self.path.split('/')
        token_count = len(target_list) - 1
        token_length = 0
        for token in token_list[1:]:
            token_length += len(token)
        ave_token = float(token_length)/len(token_list)
        longest_token = max(len(i) for i in token_list)
        return len_path, token_count, ave_token, longest_token

    def bag_of_word(self, this_delta, pre_delta, N):
        if this_delta == pre_delta:
            sim = unigram(self.path, this_delta)
        else:
            sim = n_grams(self.path, this_delta_clt=this_delta, pre_delta_clt=pre_delta, n_gram=N)
        return sim

    def digit_letter_ratio(self):
        num_digit = 0
        num_letter = 0
        for i in self.path:
            if i.isdigit():
                num_digit += 1
            elif i.isalpha():
                num_letter += 1
        if num_letter == 0:
            ratio = -1
        else:
            ratio = float(num_digit)/num_letter
        return ratio

    def has_target(self):
        for target in target_list:
            if target in self.path:
                return 1
        return 0


class Query:
    # 包括params
    def __init__(self, query):
        self.query = query
        self.contains_dots = '.'
        self.contains_at = '@'
        self.contains_hyphen = '-'

    def self_query(self):
        length = len(self.query)
        token_list = self.query.split('&')
        num_params = len(token_list)
        query_list = []
        for item in token_list:
            if '=' in item:
                query_value = item.split('=')[1]
                query_list.append(query_value)
            else:
                query_list.append(item)
        longest_query = max(len(i) for i in query_list)
        return length, num_params, longest_query

    def bag_of_word(self, this_delta, pre_delta, N):
        sim = n_grams(self.query, this_delta_clt=this_delta, pre_delta_clt=pre_delta, n_gram=N)
        return sim

    def query_characters(self):
        hyphen_query = self.query.find(self.contains_hyphen) != -1
        if hyphen_query == True:hyphen_query = 1
        else:hyphen_query = 0
        at_query = self.query.find(self.contains_at) != -1
        if at_query == True:at_query = 1
        else:at_query = 0
        dots_query = self.query.find(self.contains_dots) != -1
        if dots_query == True:dots_query = 1
        else: dots_query = 0
        return hyphen_query, at_query, dots_query


def get_alexa_rank(url):
    try:
        data = urllib2.urlopen('http://data.alexa.com/data?cli=10&dat=snbamz&url=%s' % (url)).read()
        reach_rank = re.findall("REACH[^\d]*(\d+)", data)
        if reach_rank:
            reach_rank = reach_rank[0]
        else:
            reach_rank = -1
        popularity_rank = re.findall("POPULARITY[^\d]*(\d+)", data)
        if popularity_rank:
            popularity_rank = popularity_rank[0]
        else:
            popularity_rank = -1
        return int(popularity_rank), int(reach_rank)
    except (KeyboardInterrupt, SystemExit):
        raise
    except:
        return None


def load_data():
    # 载入数据
    benign_list = []
    f = open(NORMAL_URL)
    for line in f.readlines():
        scheme, hostname, domain, path, param, query = get_url_parts(line.strip())
        benign_list.append(domain)
    f.close()
    return benign_list


#############################################################################
    #   特征提取函数
#############################################################################
def get_features_url(_url):  # 解析 URl
    url = URL(_url)
    length_url, num_dots_url, num_slashes_url, is_at_url, is_hyphen_url, is_slash_redir_url = url.url_self_features()
    # bigram_url = url.bag_of_word(bi_delta_clt, uni_delta_clt, 2)
    # trigram_url = url.bag_of_word(tri_delta_clt, bi_delta_clt, 3)
    # quadgram_url = url.bag_of_word(qua_delta_clt, tri_delta_clt, 4)
    contains_ip_url = url.ip_exist_ip()
    sensitive_term_url = url.has_sensitive_terms()
    num_non_alpha_url = url.num_non_alpha_url()
    longest_token_url, average_token_url, same_target_url = url.tokens()
    return length_url, num_dots_url, num_slashes_url, is_at_url, is_hyphen_url, is_slash_redir_url, \
           contains_ip_url, sensitive_term_url, num_non_alpha_url, \
           longest_token_url, average_token_url, same_target_url


def get_features_scheme(_Scheme):  # 解析 Scheme
    t_scheme = type_scheme(_Scheme)
    return t_scheme

'''
def get_featrues_hn():  # 解析 Hostname
    hostname = Hostname(_Hostname)
    length_hn = hostname.length()
    longest_url_hn, average_url_hn = hostname.tokens()
    unigram_hn = hostname.bag_of_word(uni_delta_clt, uni_delta_clt)
    bigram_hn = hostname.bag_of_word(bi_delta_clt, uni_delta_clt, 2)
    trigram_hn = hostname.bag_of_word(tri_delta_clt, bi_delta_clt, 3)
    quadgram_hn = hostname.bag_of_word(qua_delta_clt, tri_delta_clt, 4)
    num_chars_afterT_hn = hostname.num_chars_after_target()
    return length_hn, longest_url_hn, average_url_hn, unigram_hn, bigram_hn, trigram_hn, quadgram_hn, num_chars_afterT_hn
'''


def get_features_domain(_Domain, uni_delta_clt, bi_delta_clt, tri_delta_clt):  # 解析 Domain
    domain = Domain(_Domain)
    length_domain = domain.length()
    num_token_domain, longest_token_domain, ave_token_domain = domain.tokens()
    bigram_domain = domain.bag_of_word(bi_delta_clt, uni_delta_clt, 2)
    trigram_domain = domain.bag_of_word(tri_delta_clt, bi_delta_clt, 3)
    hyphen_domain, num_hyphen, at_domain, dots_domain = domain.domain_characters()
    subdomain = domain.subdomain()
    # return length_domain, num_token_domain, longest_token_domain, ave_token_domain, bigram_domain, \
    #        trigram_domain, subdomain, hyphen_domain, num_hyphen, at_domain, dots_domain
    return hyphen_domain, at_domain, dots_domain


def get_features_path(_Path):  # 解析 Path
    path = Path(_Path)
    if _Path == '':
        len_path, num_token_path, ave_token_path, longest_token_path, digit_letter_ratio_path, has_target_path = 0, 0, 0, 0, 0, 0
    else:
        len_path, num_token_path, ave_token_path, longest_token_path = path.tokens()
        # unigram_path = path.bag_of_word(uni_delta_clt, uni_delta_clt, 1)
        # bigram_path = path.bag_of_word(bi_delta_clt, uni_delta_clt, 2)
        # trigram_path = path.bag_of_word(tri_delta_clt, bi_delta_clt, 3)
        # quadgram_path = path.bag_of_word(qua_delta_clt, tri_delta_clt, 4)
        digit_letter_ratio_path = path.digit_letter_ratio()
        has_target_path = path.has_target()
    return len_path, ave_token_path #, num_token_path, ave_token_path, longest_token_path, digit_letter_ratio_path, has_target_path


def get_features_query(_Query):  # 解析 Query
    query = Query(_Query)
    if _Query == '':
        len_query, num_querys, longest_query, hyphen_query, at_query, dots_query = 0, 0, 0, 0, 0, 0
    else:
        len_query, num_querys, longest_query = query.self_query()
        hyphen_query, at_query, dots_query = query.query_characters()
        # trigram_query = query.bag_of_word(tri_delta_clt, bi_delta_clt, 3)
        # quadgram_query = query.bag_of_word(qua_delta_clt, tri_delta_clt, 4)
    # return len_query, num_querys, longest_query, hyphen_query, at_query, dots_query
    return hyphen_query, at_query, dots_query

def feature_extract(url, uni_delta_clt, bi_delta_clt, tri_delta_clt):
    print >> sys.stderr, 'Features extraction is starting...'
    # print >> sys.stderr, 'Process {} start ...'.format(multiprocessing.current_process())

    #############################################################################
    #   特征提取过程
    #############################################################################

    url = url.strip()
    _url = url.lower()
    _Scheme, _Hostname, _Domain, _Path, _Params, _Query = get_url_parts(_url)

    # length_url, num_dots_url, num_slashes_url, is_at_url, is_hyphen_url, is_slash_redir_url, \
    # contains_ip_url, sensitive_term_url, num_non_alpha_url, \
    # longest_token_url, average_token_url, same_target_url = get_features_url(url)

    # t_scheme = get_features_scheme(_Scheme)

    # length_domain, num_token_domain, longest_token_domain, ave_token_domain, bigram_domain, trigram_domain, \
    # subdomain, hyphen_domain, num_hyphen_domain, at_domain, dots_domain = get_features_domain(_Domain, uni_delta_clt, bi_delta_clt, tri_delta_clt)
    hyphen_domain, at_domain, dots_domain = get_features_domain(_Domain, uni_delta_clt, bi_delta_clt, tri_delta_clt)

    # len_path, num_token_path, ave_token_path, longest_token_path, digit_letter_ratio_path, has_target_path = get_features_path(_Path)
    len_path, ave_token_path = get_features_path(_Path)

    # length_query, num_querys, longest_query, hyphen_query, at_query, dots_query = get_features_query(_Query)
    hyphen_query, at_query, dots_query = get_features_query(_Query)

    # is_time = _whois(url)
    #
    # # URL's rank
    # data = get_alexa_rank(url)
    # if data:
    #     popularity_rank, reach_rank = data
    # else:
    #     popularity_rank, reach_rank = 0, 0
    # gap = max(popularity_rank, reach_rank) - min(popularity_rank, reach_rank)   # 权值转换
    # if popularity_rank != 0 and reach_rank != 0:
    #     if gap == 0:rank_value = 100001
    #     elif gap < 100000:rank_value = float(100000 / gap)
    #     else:rank_value = -1
    # else:rank_value = 0
    #
    # # webpage content
    # num_input, is_input_pwd, is_favicon = get_content(url)

    #############################################################################
    #  写入文件
    #############################################################################
    # print '{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}' \
    #     ' {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}'\
    #     .format(url, length_url, num_dots_url, num_slashes_url, is_at_url, is_hyphen_url, is_slash_redir_url,
    #                 contains_ip_url, sensitive_term_url, num_non_alpha_url,
    #                 longest_token_url, average_token_url, same_target_url,
    #                 t_scheme,
    #                 length_domain, num_token_domain, longest_token_domain, ave_token_domain,
    #                 bigram_domain, trigram_domain,
    #                 subdomain,
    #                 hyphen_domain, num_hyphen_domain, at_domain, dots_domain,
    #                 len_path, num_token_path, ave_token_path, longest_token_path,
    #                 digit_letter_ratio_path, has_target_path,
    #                 length_query, num_querys, longest_query,
    #                 hyphen_query, at_query, dots_query,
    #                 is_time, rank_value,
    #                 num_input, is_input_pwd, is_favicon)

    print '{} {} {} {} {} {} {} {}'.format(hyphen_domain, at_domain, dots_domain, len_path, ave_token_path, hyphen_query, at_query, dots_query)


def multi_feature_extract():
    # 训练bag_of_words
    uni_delta_clt, bi_delta_clt, tri_delta_clt = train_n_grams(load_data())
    print >> sys.stderr, 'training has finished!'
    print >> sys.stderr, 'Multi process is starting ...'
    p = Pool(processes=5)
    for url in sys.stdin:
        p.apply_async(feature_extract, args=(url, uni_delta_clt, bi_delta_clt, tri_delta_clt, ))
    p.close()
    p.join()
    print >> sys.stderr, 'Done，main thread quit ...'


def single_func():  # 单进程
    import signal

    def handler(signum, frame):
        raise AssertionError

    # 训练bag_of_words
    uni_delta_clt, bi_delta_clt, tri_delta_clt = train_n_grams(load_data())
    print >> sys.stderr, 'training has finished!'
    count = 0
    for url in sys.stdin:
        count += 1
        try:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(20)
            feature_extract(url, uni_delta_clt, bi_delta_clt, tri_delta_clt)  # 特征提取，写入特征文件
            print >> sys.stderr, 'no.%d: %s    has been extracted all features...' % (count, url)
        except AssertionError:
            print >> sys.stderr, 'no.%d: %s    has a problem...' % (count, url)
            continue


if __name__ == '__main__':
    single_func()
    # multi_feature_extract()
