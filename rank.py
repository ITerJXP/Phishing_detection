import re
import urllib2
from decimal import getcontext
getcontext().prec = 3

def get_alexa_rank(url):
    try:
        data = urllib2.urlopen('http://data.alexa.com/data?cli=10&dat=snbamz&url=%s' % (url)).read()
        #print data
        reach_rank = re.findall("REACH[^\d]*(\d+)", data)
        if reach_rank: reach_rank = reach_rank[0]
        else: reach_rank = 0

        popularity_rank = re.findall("POPULARITY[^\d]*(\d+)", data)
        if popularity_rank: popularity_rank = popularity_rank[0]
        else: popularity_rank = 0

        return int(popularity_rank), int(reach_rank)

    except (KeyboardInterrupt, SystemExit):
        raise
    except:
        return None

if __name__ == '__main__':
    pass
    '''
    url = 'http://indiatravelmatters.co.in/'
    data = get_alexa_rank(url)

    if data:
        popularity_rank, reach_rank = data
    else:
        popularity_rank, reach_rank = 0, 0

    print 'popularity rank = %d\nreach_rank = %d' % (popularity_rank, reach_rank)
    gap = max(popularity_rank, reach_rank) - min(popularity_rank, reach_rank)
    if popularity_rank != 0 and reach_rank != 0:
        if gap < 100000:
            value = float(100000 / gap)
        else:
            value = -1
    else:
        value = 0
    print value
    '''