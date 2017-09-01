# -*- coding: utf-8 -*- 
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from bs4 import BeautifulSoup
import urllib2


def get_html(url):
    if 'https' in url:
        url.replace('https', 'http')
    html = urllib2.urlopen(url).read()
    soup = BeautifulSoup(html, "html.parser")
    return soup


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
    
    def get_soup(self):     # 返回格式化soup对象
        info = self.soup.prettify('utf-8')
        return info
    
    def get_title(self):    # 返回soup标题
        print self.soup.title
        return self.soup.title        

    def get_input(self):    # 返回input标签
        '''
        return: input标签总数，text的input数，password的input数
        '''
        n_input, n_text, n_password = 0
        for _input in self.soup.find_all('input'):
            _type = _input.get('type') 
            if _type == 'password':
                n_password += 1
            if _type == 'text':
                n_text += 1
        n_input = n_text + n_password
        return n_input, n_text, n_password
        
    def get_href(self):     #获取 link的href属性内容
        for link in self.soup.find_all('a'):
            href = link.get('href')
            if 'http' or 'https' in href:
                print href



def main(url):
    soup = get_html(url)
    dom = HtmlDom(soup)     # 新建对象
    '''
    获取信息方法
    '''
    #dom.get_soup()
    #dom.get_title()    
    #dom.get_input()
    #dom.get_href()



if __name__ == '__main__':
    # 单例测试
    url = sys.argv[1]
    if len(sys.argv) == 2:
        main(url)
    
    # 批量
    for url in sys.stdin:
        main(url)

