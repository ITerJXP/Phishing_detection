# -*- coding:utf-8 -*-
import sys
reload(sys)
from conf import PHISH_CSV, NORMAL_URL, PHISH_URL
import pandas as pd

# 提取原始phish_csv的url]
def get_phish_url():
    p_data = pd.read_csv(PHISH_CSV)
    count = 0
    for url in p_data['url']:
        print url
        count += 1
    print count


# 将data目录中normal_url和phish_url两个文件合并，同时标记（钓鱼为'1'，正常为'0'）
def combine_url():
    for p_url in open(PHISH_URL).readlines():
        p = p_url.strip()
        print '{}\t{}'.format(p, '1')
    for n_url in open(NORMAL_URL).readlines():
        n = n_url.strip()
        print '{}\t{}'.format(n, '0')

if __name__ == '__main__':
    # get_phish_url()
    combine_url()

