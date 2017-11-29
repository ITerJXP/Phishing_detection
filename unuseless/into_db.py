# !usr/bin/env python2
# -*- coding:utf-8 -*-
import sys
reload(sys)
import urllib2
import whois
from tld import get_tld
import json
import datetime
import socket
socket.setdefaulttimeout(20.0)
# import logging
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)


DB_FILE = '/Users/JasonJay/programming/workspace/Python_anaconda2.4/Phishing_Detction/checked_url/labeled_phishSet.txt'
FALSE_URL = '/Users/JasonJay/programming/workspace/Python_anaconda2.4/Phishing_Detction/checked_url/false_url.txt'


'''
重写构造json类，遇到日期特殊处理
'''
class DateEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, date):
            return obj.strftime("%Y-%m-%d")
        else:
            return json.JSONEncoder.default(self, obj)


def batch_into():
    url_info_dic = {}
    i = 0
    with open(DB_FILE, 'wb') as f:
        for line in sys.stdin:
            url_info_dic['URL'] = line  # 添加url
            try:
                html = urllib2.urlopen(line)
                i += 1
                print "No.%d : Got it" % i
            except (urllib2.URLError, urllib2.HTTPError), e:
                i += 1
                print "No.%d" %i + " : There was an error: %r" % e
                with open(FALSE_URL, 'a') as fu:
                    fu.write(line)
                continue
            url_info_dic['HTML'] = html.readlines()  # 添加html
            if url_info_dic['HTML'] == None or len(url_info_dic['HTML']) == 0:
                url_info_dic['STATUS'] = 'No'
            else:
                url_info_dic['STATUS'] = 'YES'
            whois_info = whois.whois(get_tld(line))  # 添加whois
            url_info_dic['whois'] = whois_info
            all_data = json.dumps(url_info_dic, cls=DateEncoder, ensure_ascii=False)  # 写成json格式
            f.write(all_data)



def single_into(url):
    url_info_dic = {}
    url_info_dic['URL'] = url  # 添加url
    with open(DB_FILE, 'a') as f:
        try:
            html = urllib2.urlopen(url)
        # except (urllib2.URLError, urllib2.HTTPError), e:
        except Exception, e:
            raise Exception(e)
        url_info_dic['HTML'] = html.readlines()  # 添加html
        if url_info_dic['HTML'] == None or len(url_info_dic['HTML']) == 0:
            url_info_dic['STATUS'] = 'no'
        else:
            url_info_dic['STATUS'] = 'yes'
        whois_info = whois.whois(get_tld(url))  # 添加whois
        url_info_dic['WHOIS'] = whois_info
        url_info_dic['Label'] = 'phish'
        url_info_dic['Imitation'] = 'paypal'
        all_data = json.dumps(url_info_dic, cls=DateEncoder, ensure_ascii=False)  # 写成json格式
        f.write(all_data + '\n')


def main():
    if len(sys.argv) == 1:       # 批量添加
        batch_into()
    elif len(sys.argv) == 2:      # 单个url添加
        url = sys.argv[1]
        single_into(url)

if __name__ == "__main__":
    main()

