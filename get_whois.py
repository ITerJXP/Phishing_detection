# -*- coding: utf-8 -*-

import datetime
import whois
import sys
import socket
socket.setdefaulttimeout(20)


def _whois(url):
    """
    :param: url
    :return: URL注册时间是否合理
    """
    time_request = 730  # 网页注册时间限制
    try:
        _info = whois.whois(url)
        creat_date = _info['creation_date']

        if _info is not None:
            current = datetime.datetime.now()  # 当前时间
            if type(creat_date) == datetime.datetime:
                if (current - creat_date).days > time_request:
                    return 1
                else:
                    return 0
            elif type(creat_date) == list:
                _date = creat_date[0]
                if (current - _date).days > time_request:
                    return 1
                else:
                    return 0
        else:
            return 0
    except Exception, e:
        return 0

if __name__ == '__main__':
    url = 'http://www.wingsee.com/ghibli/'
    _whois(url)