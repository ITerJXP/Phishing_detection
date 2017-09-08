# -*- coding: utf-8 -*-

import sys
reload(sys)
from tld import get_tld
from urlparse import urlparse
import whois
import re
import datetime
from get_web_content import get_html, main
sys.path.append('/Users/JasonJay/programming/workspace/Python_anaconda2.4/Phishing_Detction');
from get_web_content import HtmlDom;
import urllib2

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
      	self.token = '[?_.=&-]'

    '''
        URL & Lexical Features
    '''    

    def url_self_features(self):
        # url长度
        length_url = len(self.url)  
        # 包括.数
        num_dots_url = self.url.count(self.contains_dots)
        # 包括/数
        num_slashes_url = self.url.count(self.slashes_url)
        # 是否包括@
        contains_at = self.url.find(self.contains_at) != -1
        if contains_at == True:at = 1
        else:at = 0
        # 是否包括-
        hyphen_url = self.url.find(self.hyphen) != -1
        if hyphen_url == True:url = 1
        else:url = 0
        return length_url, num_dots_url, num_slashes_url, at, url

    def ip_exist_one(self):      # 是否包括ip地址
        compile_rule = re.compile(r'\d+[\\.]\d+[\\.]\d+[\\.]\d+')
        match_ip = re.findall(compile_rule, self.url)       
        if match_ip:
            return 1
        else:
            return 0

    def domain_token(self):
        import tld
        try:
            domain = get_tld(self.url)
        except (tld.exceptions.TldBadUrl,tld.exceptions.TldDomainNotFound),e:
            #raise tld.exceptions.TldBadUrl(self.url)
            return 0,0,0
        else:
            domain_token = [x for x in re.split(self.token, domain)]
            num_domain_tokens = len(domain_token)       # domain_token数
            longest_domain_tokens = max(len(i) for i in domain_token)      # domain token最大长度
        
            num_token = 0
            for i in domain_token:  
                num_token += len(i)
            average_domain_tokens = float(num_token) / num_domain_tokens       # domain tokens平均长度
            return num_domain_tokens, longest_domain_tokens, average_domain_tokens

    def path(self):
        num_path_tokens = 0
        longest_path_tokens = 0
        average_path_tokens = 0.0

        url = urlparse(self.url)
        path = url.path.replace('/', '')
        num_subdirectory = len(path.split('/'))     # path数
    
        if len(path) != 0:
            path_token = [x for x in re.split(self.token, path)]
            num_path_tokens = len(path_token)       # path_token数
            longest_path_tokens = max(len(i) for i in path_token)      # path token最大长度
            num_token = 0
            for i in path_token: 
                num_token += len(i)
            average_path_tokens = float(num_token) / num_path_tokens       # domain tokens平均长度
        return num_subdirectory, num_path_tokens, longest_path_tokens,  average_path_tokens

    def num_non_alpha_url(self):        # 非字母个数
        num_non_alpha = 0
        for s in self.url:
            if not s.isalpha():
                num_non_alpha += 1
        return num_non_alpha 

    def has_sensitive_terms(self):      # 是否包括敏感词汇
        bool_list = [i in self.url for i in self.sensitive_term_url]
        if True in bool_list:           
            return 1
        else:
            return 0

    '''
    def has_port(self):     # 获取url端口
        url = urlparse(self.url)
        return url.port  
    '''

    def query(self):
        url = urlparse(self.url)
        length_querystr = len(url.query)        # query长度  
        num_params = len(url.query.split('&'))      # query个数
        return length_querystr, num_params


class CONTENT:
    def __init__(self, url):
        self.url = url
        self.dom = main(url) 

    def get_input(self):
        n_input, n_text, n_password = self.dom.get_input()
        return n_input, n_text, n_password      # 输入框数； 文本输入框数； 密码输入框数

    def get_href(self):
        same_num = 0
        href_list, href_num = self.dom.get_href()
        for href in href_list:
            if href == self.url:
                same_num += 1
        return href_num, same_num       # 一个网页中包括总链接数； 网页中链接数据和url相同数


class WHOIS:
    def __init__(self, url, whois_dic):
        self.url = url
        self.whois_dic = whois_info
    
    def age_of_domain(self):        # 判断domain age是否小于60天
        creation_date = self.whois_dic['creation_date']
        if creation_data != None:
            cur = datetime.datetime.now()
            if cur - datetime.timedelta(days=60) > creation_date:
                return 1
            else:
                return 0
        else:
	    	return 0



if __name__ == '__main__':
    for url in sys.stdin:
        url = url.lower()
        
		# url
        ob_url = URL(url)
        length_url, num_dots_url, num_slashes_url, contains_at, hyphen_url = ob_url.url_self_features()
        ip_status = ob_url.ip_exist_one()
        num_domain_tokens, longest_domain_tokens, average_domain_tokens = ob_url.domain_token()
        num_subdirectory, num_path_tokens, longest_path_tokens, average_path_tokens = ob_url.path()
        num_non_alpha = ob_url.num_non_alpha_url()
        sens_status = ob_url.has_sensitive_terms()
        length_querystr, num_params = ob_url.query()
        print '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(url.strip(), length_url, num_dots_url, num_slashes_url, contains_at, hyphen_url, ip_status, num_domain_tokens, longest_domain_tokens, average_domain_tokens,num_subdirectory, num_path_tokens, longest_path_tokens, average_path_tokens,num_non_alpha, sens_status, length_querystr, num_params)
       	'''
        # content
        import socket
        import httplib
        try:
            ob_content = CONTENT(url)
        except (urllib2.HTTPError, urllib2.URLError, httplib.BadStatusLine, socket.error),e:
            print '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(url.strip(), length_url, num_dots_url, num_slashes_url, contains_at, hyphen_url, ip_status, num_domain_tokens, longest_domain_tokens, average_domain_tokens,num_subdirectory, num_path_tokens, longest_path_tokens, average_path_tokens,num_non_alpha, sens_status, length_querystr, num_params, 0, 0, 0, 0, 0)
        else:
            _input, n_text, n_password = ob_content.get_input()
            href_num, same_num = ob_content.get_href()
            print '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(url.strip(), length_url, num_dots_url, num_slashes_url, contains_at, hyphen_url,ip_status, num_domain_tokens, longest_domain_tokens, average_domain_tokens, num_subdirectory, num_path_tokens, longest_path_tokens, average_path_tokens,num_non_alpha, sens_status, length_querystr, num_params, _input, n_text, n_password, href_num, same_num)
        '''
        # whois
        '''
        try:
        	whois_info = whois.query(get_tld(url))
	    	ob_whois = WHOIS(url, whois_info.__dict__)
        	age_status = ob_whois.age_of_domain()

        	print '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}' \
              '\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'\
            .format(url, length_url, num_dots_url, num_slashes_url, contains_at, hyphen_url,
                    ip_status, num_domain_tokens, longest_domain_tokens, average_domain_tokens,
                    num_subdirectory, num_path_tokens, longest_path_tokens, average_path_tokens,
                    num_non_alpha, sens_status, length_querystr, num_params,_input, n_text, n_password,
                    href_num, same_num, age_status)
        except Exception, e:
        	pass    	             
        '''



