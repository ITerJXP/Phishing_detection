#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding("utf-8")
from tld import get_tld
from urlparse import urlparse
import whois

class URL:
    def __init__(self, url):
        self.url = url
        self.contains_dots = '.'
        self.contains_at = '@'
        self.sensitive_term_url = ['webscr', 'secure', 'banking', 'ebayisapi', 'account', 'confirm', \
                                    'login', 'signin', 'paypal', 'free', 'lucky', 'bonus index', 'includes', \
                                     'content', 'images', 'admin', 'file doc', 'account', 'update', 'confirm', \
                                    'verify', 'secur', 'notif', 'log', 'click', 'inconvenien', 'urgent', 'alert']
        self.hyphen = '-'
        self.slashes_url = '/'
        self.slash_reidr = '//'
        self.token = '?_.=&-'

    
    '''
        URL & Lexical Features
    '''    

    def url_self_features():
        # url长度
        length_url = len(self.url)  
        # 包括.数
        num_dots_url = self.url.count(self.contains_dots)
        # 包括/数
        num_slashes_url = self.url.count(self.slashes_url)
        # 是否包括@
        contains_at = self.url.find(self.contains_at)!=-1
        # 是否包括-
        hyphen_url = self.url.find(self.hyphen)!=-1
        return length_url, num_dots_ulr, num_slashes_url, contains_at, hyphen_url 
        
    
    def ip_exist_one():      # 是否包括ip地址
        compile_rule = re.compile(r'\d+[\.]\d+[\.]\d+[\.]\d+')   
        match_ip = re.findall(compile_rule, self.url)       
        if match_ip:
            return True
        else:
            return False


    def domain_token():
        domain = get_tld(self.url)
        domain_token = [x for x in re.split(self.token), domain) if x]
        num_domain_tokens = len(domain_token)       # domain_token数
        longest_domain_tokens = max(len(i) for i in domain_token)      # domain token最大长度
        
        num_token = 0
        for i in domain_token:  
            num_token += len(i)
        average_domain_tokens = float(num_token) / num_domain_tokens       # domain tokens平均长度
        return num_domain_tokens, longest_domain_tokens, average_domain_tokens
    
    
    def path():
        num_path_tokens = 0
        longest_path_tokens = 0
        average_domain_tokens = 0.0
        num_subdirectory = 0

        url = urlparse(self.url)
        path = url.path.replace('/','')
        num_subdirectory = len(path.split('/'))     # path数
    
        if len(path)!=0:
            path_token = [x for x in re.split(self.token), path) if x]
            num_path_tokens = len([path_token)       # path_token数
            longest_path_tokens = max(len(i) for i in path_token)      # path token最大长度
            num_token = 0
            for i in path_token: 
                num_token += len(i)
            average_path_tokens = float(num_token) / num_path_tokens       # domain tokens平均长度
        return num_subdirectory, num_path_tokens, longest_path_tokens,  average_domain_tokens

    
    def num_non_alpha_url():        # 非字母个数
        num_non_alpha = 0
        for s in self.url:
            if not s.isalpha():
                num_non_alpha += 1
        return num_non_alpha 
        

    def has_sensitive_terms():      # 是否包括敏感词汇
        bool_list = [i in self.url for i in self.sensitive_term_url]
        if True in bool_list:           
            return True
        else:
            return False


    def has_port():     # 获取url端口
        url = urlparse.urlparse(self.url)
        return url.port  
    
    
    def query():
        url = urlparse.urlparse(self.url)
        length_querystr = len(url.query)        # query长度  
        num_params = len(url.query.split('&'))      # query个数
        return length_querystr, num_params

                                   
class CONTENT:
    def __init__(self, url):
        self.url = url
 
 
class WHOIS:
    def __init__(self, url, whois_info):
        self.url = url
        self.whois = whois_info
    
    def age_of_domain():
        creation_date = self.whois['creation_date']


if __name__ == '__main__':
    for url in sys.stdin:
        url = url.lower()
        whois_info = whois.whois(url)
        # 实例化
        ob_url = ULR(url)
        ob_content = CONTENT(url)
        ob_whois = WHOIS(url, whois_info)
        # 生成特征

