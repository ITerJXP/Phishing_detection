# -*- coding: utf-8 -*- 

from bs4 import BeautifulSoup
import urllib2
import sys

def get_html(url):
    if 'https' in url:
        url.replace('https', 'http')
    try:
        html = urllib2.urlopen(url, timeout=5).read()
        soup = BeautifulSoup(html, "html.parser")
        return soup
    except Exception, e:
        return None


class HtmlDom:
    #################################
    ## HtmlDom类包括方法：
    ## 1. get_soup() 
    ## 2. get_title()
    ## 3. get_input()
    ## 4. get_href()
    #################################

    def __init__(self, soup):
        self.soup = soup

    '''
    def get_soup(self):     # 返回格式化soup对象
        info = self.soup.prettify('utf-8')
        return info
    
    def get_title(self):    # 返回soup标题
        # print self.soup.title
        return self.soup.title

    '''
    def get_href(self):     # 获取 link的href属性内容
        href_list = []
        for link in self.soup.find_all('a'):
            href = link.get('href')
            if 'http' or 'https' in href:
                href_list.append(link)
        return href_list

    def get_input(self):        # 返回input标签
        """
        return: input标签总数, 是否包括password
        """
        is_pwd = 0
        n_input, n_text, n_password = 0, 0, 0
        for _input in self.soup.find_all('input'):
            _type = _input.get('type') 
            if _type == 'password':
                is_pwd = 1
                n_password = 1
            elif _type != 'hidden':
                n_text += 1
        num_input = n_text + n_password
        return num_input, is_pwd

    def favicon(self):      # 是否包括 shortcut icon
        for _link in self.soup.find_all('link'):
            rel_value = _link.get('rel')
            if rel_value is not None:
                if 'icon' in rel_value:
                    is_favicon = 1
                    return is_favicon
                else:
                    is_favicon = -1
                    return is_favicon
            else:
                is_favicon = 0
                return is_favicon


def text_featrues(url):
    """
    :param :url
    :return:(1) 包括input数
            (2) 是否包括密码输入
    """
    _soup = get_html(url)
    if _soup is not None:
        dom = HtmlDom(_soup)     # 新建对象
        num_input, is_input_pwd = dom.get_input()
    else:
        num_input, is_input_pwd = 0, 0
    return num_input, is_input_pwd


def icon_features(url):
    _soup = get_html(url)
    if _soup is not None:
        dom = HtmlDom(_soup)
        is_favicon = dom.favicon()
    else:
        is_favicon = 0
    return is_favicon


def get_content(url):
    num_input, is_input_pwd = text_featrues(url)
    is_favicon = icon_features(url)
    return num_input, is_input_pwd, is_favicon


if __name__ == '__main__':
    URL = 'https://www.paypal.com/signin?country.x=C2&locale.x=zh_C2'
    get_content(URL)
