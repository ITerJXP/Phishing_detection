#!/usr/bin/python  
# -*- coding: utf-8 -*-  

import sys
reload(sys)
sys.setdefaultencoding("utf-8") 
import whois

def get_whois_info(url):
    whois_info = whois.whois(url)
    print 'url包括几个字段：%d'%len(whois_info)
    print '----------------------------------------'
    for k,v in whois_info.items():
        print '{}:{}'.format(k,v)

def get_whois_s_info(url, key):
    whois_info = whois.whois(url)
    print type(whois_info[key]), whois_info[key]

        
        
if __name__ == '__main__':
    url = sys.argv[1]
    key = sys.argv[2]
   
    if len(sys.argv) == 2:
        get_whois_info(url)

    if key!=None and len(sys.argv) == 3:
       get_whois_s_info(url, key)
