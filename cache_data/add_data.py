# -*- coding:utf-8 -*-
import sys

if __name__ == '__main__':
    i = 0
    for line in sys.stdin:
        url = line.strip().split(',')[0]
        label = line.strip().split(',')[1]
        if label == 'yes':
            print url
