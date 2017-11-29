# -*- coding: utf-8 -*-

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import json

def analysis():
    json_path = '/Users/JasonJay/programming/workspace/Python_anaconda2.4/Phishing_Detction/verified_online.json'
    with open(json_path) as f:
        phish_list = json.load(f)
    return phish_list

def main():
    phish_list = analysis()
    i = 0
    for l in phish_list:
        valid = l['verified']
        online = l['online']
        if valid == 'yes' and online == 'yes':
            print l['url']
            i += 1
        if i == 40000:
            break

if __name__ == '__main__':
    main()
