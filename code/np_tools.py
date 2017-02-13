#! /usr/bin/python
# -*- coding: utf-8 -*-

from math import log
from operator import itemgetter
from pickle import load, dump
from os.path import exists
from datetime import date, datetime, timedelta
import operator
from itertools import *
from collections import defaultdict
from random import *
from bisect import *
import time
import re
import gc

RE_URL_HEAD = re.compile("^/([^/]+)/")
RE_FM_URL = re.compile('^/fm')
RE_ID = re.compile("\S+/([0-9]+)")


def get_url_head(url):
    url_head_match = RE_URL_HEAD.match(url)
    return url_head_match.group(1) if url_head_match else ('homepage' if url == '/' else 'unknown')


def is_fm_url(url):
    m = RE_FM_URL.match(url)
    return m is not None


def avg(l):
    s, n = 0.0, 0
    for i in l:
        s += i
        n += 1
    return s / n


def median(li):
    n = len(li)
    sli = sorted(li)
    return sli[n / 2]


def retry(func, retries=3, delay=3, IgnoreException=Exception):
    """retry decorator to emulator ruby's retry"""

    def dec(*args, **kwargs):
        for i in range(retries - 1):
            try:
                return func(*args, **kwargs)
            except IgnoreException:
                time.sleep(delay)
        func(*args, **kwargs)

    return dec


def curry(x, argc=None):
    if argc is None:
        argc = x.__code__.co_argcount

    def p(*a):
        if len(a) == argc:
            return x(*a)

        def q(*b):
            return x(*(a + b))

        return curry(q, argc - len(a))

    return p




def get_digit_id_from_url(url):
    url_match = RE_ID.match(url)
    return int(url_match.group(1)) if url_match else 0


@curry
def get_id_from_url(url_re, url):
    url_head_match = url_re.match(url)
    return int(url_head_match.group(1)) if url_head_match else 0


def get_id_from_url_factory(url_head):
    head_re = re.compile("^/%s/([0-9]+)/" % url_head)
    return get_id_from_url(head_re)


get_people_id = get_id_from_url_factory("people")
get_group_id = get_id_from_url_factory("group")
get_topic_id = get_id_from_url_factory('group/topic')
get_online_id = get_id_from_url_factory('online')
get_tribe_id = get_id_from_url_factory('tribe')


def double_iter(iter):
    for x in iter:
        yield x
        yield x


def choose_iter(elements, n):
    """iter choose n in elements"""
    for i in range(len(elements)):
        if n == 1:
            yield (elements[i],)
        else:
            for next in choose_iter(elements[i + 1:len(elements)], n - 1):
                yield (elements[i],) + next


def choose(l, k):
    """generating all combinations of k elements from L"""
    return list(choose_iter(l, k))


def double_dict():
    return defaultdict(lambda: {})


def double_dict1():
    return defaultdict(lambda: defaultdict(lambda: 0))


def double_dict2():
    return defaultdict(lambda: defaultdict(lambda: double_dict1()))


def topk_defaultdict(dic, k):
    allitems = sorted(dic.items(), key=itemgetter(1), reverse=True)
    b = defaultdict(int)
    b.update(dict(allitems[:k]))
    return b


def get_next_month(mydate):
    year, month = divmod(mydate.month + 1, 12)
    if month == 0:
        month = 12
        year = year - 1
    return date(mydate.year + year, month, 1)


def auto_cache(func):
    caches = {}
    cache_time = {}

    def _cache(*args):
        t = datetime.now()
        if (args not in caches) or (t - cache_time[args]).seconds > 3600:
            caches[args] = func(*args)
            cache_time[args] = t
        return caches[args]

    return _cache


class Printer_and_Writer(object):
    def __init__(self, f_path):
        self.f = open(f_path, 'w')
        self.un_closed = True

    def __del__(self):
        if self.un_closed: self.close()

    def __call__(self, *args):
        self.output(args[0])

    def output(self, line):
        self.f.write("%s\n" % line)
        print(line)

    def close(self):
        self.f.flush()
        self.f.close()
        self.un_closed = False


@curry
def write_and_print(f, line):
    f.write("%s\n" % line)
    print(line)


def sublist(li, sub):
    """if list contains sub, return True or False"""
    n = len(sub)
    return any((sub == li[i:i + n]) for i in range(len(li) - n + 1))


def read_text(txt_file):
    with open(txt_file) as f:
        content = f.read().splitlines()
        return "".join(content)


def read_lines(txt_file):
    with open(txt_file) as f:
        for line in f.readlines():
            yield line


def load_from(fname, path='./data/', init=None):
    if not is_exist(fname, path): return init
    f = open(path + fname)
    a = load(f)
    f.close()
    return a


def get_timestamp(y, m, d):
    day = date(y, m, d)
    t = time.mktime(day.timetuple())
    return int(t * 1000)


def dump_to(a, fname, path='./data/'):
    f = open(path + fname, 'w')
    dump(a, f)
    f.close()


def is_exist(fname, path='./data/'):
    return exists(path + fname)


def all_right(x):
    return True


def tree():
    return defaultdict(tree)


def interval(x, left, right):
    return max(min(x, right), left)


def max_min(li):
    a = li.next()
    ma, mi = a, a
    for i in li:
        if i > ma:
            ma = i
        elif i < mi:
            mi = i
    return ma, mi


def max_min_avg(li):
    li1, li2 = tee(li)
    ma, mi = max_min(li1)
    av = avg(li2)
    return ma, mi, av


def spliter(v, group_size=100):
    x = []
    for a in v:
        x.append(a)
        if len(x) == group_size:
            y = list(x)
            x = []
            yield y
    yield x


def get_date_in_month(y, m):
    d = date(y, m, 1)
    enddate = date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)
    delta = timedelta(days=1)
    while d < enddate:
        yield d
        d += delta


def get_months_in_year(start_year, start_month, num=12):
    y, m = start_year, start_month
    t = 0
    while t < num:
        d = date(y, m, 1)
        yield d
        if m < 12:
            m += 1
        else:
            y, m = y + 1, 1
        t += 1


def datespan(startDate, endDate, delta=timedelta(days=1), include=False):
    """usage: for day in datespan(date(2010,1,1), date(2010,7,7)): """
    currentDate = startDate
    if include: endDate += timedelta(days=1)
    while currentDate < endDate:
        yield currentDate
        currentDate += delta


def get_seconds_span(t1, t2, order=1):
    if t1 > t2:
        t1, t2 = t2, t1
        order = -1
    s = ((t2.hour - t1.hour) * 60 + t2.minute - t1.minute) * 60 + t2.second - t1.second
    return order * s


def get_seconds_span_old(t1, t2):
    d = datetime.today()
    d1 = datetime.combine(d, t1)
    d2 = datetime.combine(d, t2)
    return (d2 - d1).seconds


def until_today(howmuch_days_before=0):
    """
    howmuch_days_before is a positive int number
    """
    start_date = get_date_before(d=howmuch_days_before)
    today = date.today()
    delta = timedelta(days=1)
    current_date = start_date
    while current_date < today:
        yield current_date
        current_date += delta


def do_until_today(job_func, days=0):
    for day in until_today(days):
        job_func(day)


def gen_dates_until(target_date, base_date, delta):
    c = base_date
    yield c
    while c < target_date:
        c += delta
        yield c
    yield target_date


def day_by_day(check_by_day, startday_str, endday_str):
    startdate = str2date(startday_str)
    enddate = str2date(endday_str)
    for day in datespan(startdate, enddate):
        check_by_day(day)


def str2date(s):
    y, m, d = tuple([int(x) for x in s[:10].split("-")])
    return date(y, m, d)


def sql_in(farm_execute, pre_sql, li):
    st = pre_sql + " (%s)" % (",".join(['%s'] * len(li)))
    return farm_execute(st, tuple(li))


def sql_regexp_num(farm_execute, pre_sql, num):
    st = pre_sql + " '%s'" % (gen_num_regexp(num))
    return farm_execute(st)


def sql_regexp_nums(farm_execute, pre_sql, nums):
    st = pre_sql + " '%s'" % (gen_nums_regexp(nums))
    return farm_execute(st)


def gen_nums_regexp(numlist):
    """regexp for a list of nums in string"""
    str = "|".join([gen_num_regexp(n) for n in numlist])
    return str


def gen_num_regexp(num):
    """regexp for num in string"""
    return "[^0-9]%s[^0-9]|^%s$|^%s[^0-9]|[^0-9]%s$" % (num, num, num, num)


def str2list(str, type=None):
    if str:
        a = str.split('|')
        return type and [type(x) for x in a if x] or a
    else:
        return []


def liststr2intlist(str):
    a = str[1:-1].split(",")
    results = []
    for x in a:
        if x[-1] == "L":
            x = x[:-1]
        results.append(int(x))
    return results


def list_trans(li):
    """turn aaaaabbbbccc to [(5,a),(4,b),(3,c)]"""
    ch = ''
    n = 0
    new_li = []
    for x in li:
        if x != ch:
            if n: new_li.append((n, ch))
            ch = x
            n = 1
        else:
            n += 1
    if ch: new_li.append((n, ch))
    return new_li


def max_of_dict(dic):
    """return the (key, max_value) of dict"""
    li = sorted([a for a in dic.items()], key=itemgetter(1), reverse=True)
    return li[0]


def str2set(s, type=None):
    """turn aaa|bbb|cccc to a int set([aaa,bbb,cccc])"""
    return set(str2list(s, type))


def str2set_int(s):
    return str2set(s, type=int)


def set2str(s):
    return "|".join((str(x) for x in s))


def utf8tuple2str(tu):
    pass


def list2str_(li, sort=True, split="|", prefix="", postfix=""):
    """str rep for database"""
    if sort: li.sort()
    s = split.join([str(x) for x in li])
    return prefix + s + postfix


def list2str(li):
    """return str as (111,222,333)"""
    return list2str_(li, sort=False, split=',', prefix='(', postfix=')')


def strli2str(li):
    """make str li to ('111','222','333')"""
    return list2str_(li, sort=False, split="','", prefix="('", postfix="')")


def results2str(results, col=0, split=',', prefix='(', postfix=')'):
    s = split.join((str(x[col]) for x in results))
    return prefix + s + postfix if s else ""


def dict2str(dc, split1='|', split2=":"):
    token = "%s" + split2 + "%s"
    s = split1.join([token % (k, v) for k, v in dc.items()])
    return s


def dict2str_f(dc, split1='|', split2=":", f=lambda x: x):
    token = "%s" + split2 + "%s"
    s = split1.join([token % (k, f(v)) for k, v in dc.items()])
    return s


def dict_values2str(dc, split=','):
    return split.join([str(v) for k, v in sorted(dc.items())])


def str2dict(st, ktype=int, vtype=int, split1="|", split2=":", filter_v=0):
    c = (x.split(split2) for x in st.split(split1) if x)
    if filter_v:
        cc = [[ktype(x[0]), vtype(x[1])] for x in c if x]
        cc = filter(lambda x: x[1] > filter_v, cc)
    else:
        cc = [[ktype(x[0]), vtype(x[1])] for x in c if x]
    return dict(cc)


def str2dict_x(str):
    """return str2dict(str,ktype=int,vtype=float)"""
    return str2dict(str, vtype=float)


def str2defaultdict_x(str):
    c = (x.split(':') for x in str.split('|') if x)
    cc = ((int(a), float(b)) for a, b in c)
    d = defaultdict(int)
    d.update(dict(cc))
    return d


def str2dict_xf(str):
    return str2dict(str, vtype=float, filter_v=0.25)


def filter_dict(d, value_cond=lambda x: x):
    for k, v in d.items():
        if value_cond(v):
            yield k


def get_filted_dict(d, value_cond):
    new_dict = {}
    for k, v in d.items():
        if value_cond(v):
            new_dict[k] = v
    return new_dict


def get_filted_dict_2(dic, key_cond):
    new_dic = {}
    for k, li in dic.items():
        if key_cond(k):
            new_dic[k] = li
    return new_dic


def del_from_dict(dic, del_condition):
    to_del = []
    for id in dic:
        if del_condition(dic[id]): to_del.append(id)
    for id in to_del: del dic[id]


def double_dict_2str(dc):
    def r(x): return "%.4f" % x

    def dict2str_x(dic): return dict2str_f(dic, f=r)

    return dict2str_f(dc, split1=";", split2="_", f=dict2str_x)


def str2_double_dict(str, firstkey=int):
    return str2dict(str, ktype=firstkey, vtype=str2dict_x, split1=";", split2="_")


def invert_dict(dic):
    return dict(zip(dic.values(), dic.keys()))


def invert_dict_2_keyset(dic):
    re_dic = {}
    for k, v in dic.items():
        re_dic.setdefault(v, set()).add(k)
    return re_dic


def invert_keyset(dic):
    re_dic = defaultdict(set)
    for k, li in dic.items():
        for i in li:
            re_dic[i].add(k)
    return re_dic


def keyli_2_keyset(kl):
    ks = defaultdict(set)
    for k, li in kl.items():
        ks[k] = set(li)
    return ks


def keyset_2_keyli(ks):
    kl = defaultdict(list)
    for k, se in ks.items():
        kl[k] = list(se)
    return kl


def dict2_default_dict(dic, default_value):
    a = defaultdict(lambda: default_value)
    for k in dic:
        a[k] = dic[k]
    return a


def rank_dict(unsorted_dict):
    return sorted(unsorted_dict.items(), key=itemgetter(1), reverse=True)


def col2list(fetch_results, column=0, f=None):
    """"""
    return f and [f(row[column]) for row in fetch_results] or [row[column] for row in fetch_results]


def col2filter(fetch_results, f, filter=1):
    return [row for row in fetch_results if f(row[filter])]


def col2set(fetch_results, column=0, f=None):
    """get a column from fetchall() results"""
    return set(col2list(fetch_results, column, f))


def col2dict(rows, ktype=None, vtype=None):
    """rows:[k,v]->{k:v}"""
    return dict(((ktype(x), vtype(y)) for x, y in rows) if (ktype and vtype) else rows)


def col2dict1(rows, default_func, ktype=int, vtype=int):
    """rows:[k,v]->defaultdict{k:v}"""
    a = defaultdict(default_func)
    for k, v in rows: a[ktype(k)] = vtype(v)
    return a


@curry
def _sql2key_list(kf, vf, dbq, sql):
    a = {}
    for k, v in dbq(sql):
        a.setdefault(kf(k), []).append(vf(v))
    return a


sql2key_list_int_int = _sql2key_list(int, int)


def col2dict2(rows, ktype=int, vtype=int, return_defaultdict=True):
    """rows:[(k,v)]-> {k:[v1,v2],...}"""
    a = defaultdict(list)
    for k, v in rows:
        a[ktype(k)].append(vtype(v))
    return a if return_defaultdict else dict(a)


def col2dict3(rows, ktype=int, vtype=int, return_defaultdict=True):
    """rows:[(k,v)]->{K:set([v1,v2]),..}"""
    d = defaultdict(set)
    for k, v in rows:
        d[ktype(k)].add(vtype(v))
    return d if return_defaultdict else dict(d)


def col2dict3_v(rows, ktype=int, vtype=int, return_defaultdict=True):
    """rows:[(v,k)]->{K:set([v1,v2]),..}"""
    d = defaultdict(set)
    for v, k in rows:
        d[ktype(k)].add(vtype(v))
    return d if return_defaultdict else dict(d)


def col2dict4(rows, ktype=int, vtype=list):
    """row: x1,x2,...xn -> x1:(x2,...xn)"""
    v = [(ktype(x[0]), vtype(x[1:])) for x in rows]
    return dict(v)


def tuple2str(tu):
    stu = map(str, tu)
    s = ",".join(stu)
    return "(" + s + ")"


def dict_add(d1, d2, func=operator.add):
    for k in d2:
        d1[k] = func(d1[k], d2[k]) if k in d1 else d2[k]


def gen_batches_(start, end, step):
    """smallest step is 0.01, from start DOWNTO end"""

    def r(x): return round(x * 100) / 100.0

    _max, _min = max(start, end), min(start, end)
    return [(r(_max - (i + 1) * step), r(_max - i * step)) for i in range(int((_max - _min) / step))]


def gen_batches(start, end, step):
    def r(x): return round(x * 100) / 100.0

    _max, _min = max(start, end), min(start, end)
    for i in range(int(_max - _min) / step):
        yield r(_max - (i + 1) * step), r(_max - i * step)


def print_distribution(dicts, min=0, max=2, step=0.1):
    ba = gen_batches_(min, max, step)
    for min, max in ba:
        print(min, max, ":", len([x for x in dicts if max > dicts[x] >= min]))


def timer(test_f):
    '''a decorator to show func time consumed.'''

    def _timer(*args, **kw):
        t1 = datetime.now()
        r = test_f(*args, **kw)
        t2 = datetime.now()
        delta = t2 - t1
        print("%s cost %s sec, %s ms" % (test_f.__name__, delta.seconds, delta.microseconds))
        return r

    return _timer


def now_str(hide_microseconds=True, hide_date=True):
    t = str(datetime.now())
    t = t[:19] if hide_microseconds else t
    return t[11:] if hide_date else t


def shift(li, target, delta):
    '''for list li, move target in li to position: index(target)+delta'''
    index = li.index(target)
    new_index = (index + delta) % len(li)
    move(li, index, new_index)


def move(li, index1, index2):
    """for list `li`: move li[index1] to li[index2]"""
    if index1 < 0:
        index1 += len(li)
    if index2 < 0:
        index2 += len(li)
    if index1 < index2:
        li.insert(index2 + 1, li[index1])
        del li[index1]
    elif index1 > index2:
        li.insert(index2, li[index1])
        del li[index1 + 1]


def exchange(li, index1, index2):
    """in list `li`: exchange li[index1] and li[index2]"""
    if index1 < 0:
        index1 += len(li)
    if index2 < 0:
        index2 += len(li)
    _min, _max = min(index1, index2), max(index1, index2)

    if _min < _max:
        x1 = li[_min]
        x2 = li[_max]
        li.insert(_min, x2)
        li.insert(_max + 1, x1)
        del li[_min + 1]
        del li[_max + 1]


def position_min(li):
    return li.index(min(li))


def position_max(li):
    return li.index(max(li))


def position_avg(li):
    a = avg(li)
    d, delta = 0, li[0]
    for x in li:
        d0 = abs(x - a)
        if d0 < delta:
            d, delta = li.index(x), d0
    return d


class Groupby(dict):
    def __init__(self, seq, key=lambda x: x, **kwargs):
        for value in seq:
            k = key(value)
            self.setdefault(k, []).append(value)

    __iter__ = dict.items


class GroupCount(dict):
    def __init__(self, seq=None, **kwargs):
        if seq:
            self.update_seq(seq)

    def update_seq(self, seq, filter_set=None):
        if filter_set is None:
            for x in seq: self[x] = self.setdefault(x, 0) + 1
        else:
            for x in seq:
                if x in filter_set: self[x] = self.setdefault(x, 0) + 1

    def update_thres(self, dc, threshold=0):
        for x, v in dc.items():
            if v > threshold:
                self[x] = self.setdefault(x, 0) + v

    def update_dict(self, dc, filter_set=None):
        if filter_set is None:
            for x, v in dc.items(): self[x] = self.setdefault(x, 0) + v
        else:
            for x, v in dc.items():
                if x in filter_set: self[x] = self.setdefault(x, 0) + v

    def add(self, a):
        self[a] = self.setdefault(a, 0) + 1

    def iter_validkey(self, threshold):
        for x in self.keys():
            if self[x] > threshold: yield x

    def get_validkeys(self, threshold):
        return [x for x in self.iter_validkey(threshold)]


class CDF_Grouper(dict):
    def __init__(self, seq, key=lambda x: x, **kwargs):
        a = {}
        for value in seq:
            k = key(value)
            a[k] = a.setdefault(k, 0) + 1
        self.total = sum(a.values())
        b = 0
        for k in sorted(a.keys()):
            b += a[k]
            self[k] = 1.0 * b / self.total

    __iter__ = dict.items


class Discrete_cdf(object):
    def __init__(self, data):
        self._data = sorted(data)  # must be sorted
        self._data_len = float(len(data))

    def __call__(self, point):
        return (len(self._data[:bisect_right(self._data, point)]) /
                self._data_len)


def list2set(seq, threshold):
    gcount = GroupCount(seq)
    return set([x for x, v in gcount.items() if v > threshold])


def most_of(li):
    maxc, mk = 0, None
    for k, g in groupby(li):
        gg = list(g)
        if len(gg) > maxc:
            maxc, mk = len(gg), k
    return mk, maxc


def most_sub_of(li):
    gb = [list(g) for k, g in groupby(li)]
    gb.sort(cmp=lambda x, y: len(y) - len(x))
    return gb[0]


def index_of(li):
    return dict([(x, y) for y, x in enumerate(li)])


def sub_dict(somedict, somekeys, default=None):
    return dict([(k, somedict.get(k, default)) for k in somekeys])


def normalize(feature, col_avg_std):
    col_n = len(feature)
    row = [0] * col_n
    for i in range(col_n):
        avg, std = col_avg_std[i]
        row[i] = (feature[i] - avg) / std
    return row


def anti_normalize(data, col_avg_std):
    row_n = len(data)
    col_n = len(col_avg_std)

    new_data = [[0] * col_n for _ in range(row_n)]
    for i in range(col_n):
        avg_i, std_i = col_avg_std[i]
        for j in range(row_n):
            new_data[j][i] = data[j][i] * std_i + avg_i

    return new_data


def grouper_no_fill(iterable, n):
    """
    把iterable切割成n个一份逐个传出
    :param iterable: iterable
    :param n: 每份的大小
    """
    result = []
    for x in iterable:
        result.append(x)
        if len(result) >= n:
            yield result
            del result[:]
    if result:
        yield result


def grouper_insert(store, insert_sql_pre, format, li):
    glist = grouper_no_fill(li, 50000)
    for g in glist:
        str_g = ",".join((format % tuple(item) for item in g))
        store.farm.execute(insert_sql_pre + str_g)
    store.commit()


def bisection_index_sorted(sorted_tuplelist, target_value, tuple_key=0, start_i=0, end_i=None):
    # presumption: tuplelist sorted and reverse=False.
    ei = len(sorted_tuplelist) if (end_i is None) else end_i
    if start_i >= ei: return -1
    i = (start_i + ei) / 2
    t = sorted_tuplelist[i][tuple_key]
    if t == target_value:
        return i
    elif t > target_value:
        return bisection_index_sorted(sorted_tuplelist, target_value, tuple_key, start_i, i)
    elif t < target_value:
        return bisection_index_sorted(sorted_tuplelist, target_value, tuple_key, i + 1, end_i)
    else:
        return -1


def bindex(tuplelist, target_value, key=0, not_sorted=True):
    if not_sorted: tuplelist.sort(key=itemgetter(key))
    i = bisection(tuplelist, target_value, key)
    return tuplelist[i] if i >= 0 else None


def bisection(sorted_tuplelist, target_value, key=0, start_i=0, end_i=None):
    if end_i is None: end_i = len(sorted_tuplelist)
    while start_i < end_i:
        mi = (start_i + end_i) / 2
        t = sorted_tuplelist[mi][key]
        if t < target_value:
            start_i = mi + 1
        elif t > target_value:
            end_i = mi
        else:
            return mi
    return -1


def merge_key_listers(kl1, kl2):
    for k, li in kl1.items():
        kl2[k].extend(li)
    return kl2


def ranker(li, with_weight=True):
    """li: [(id, w, t), ...]
    id: object id.
    w: int，from 1 to n to measure importance
    t: int. from 0 to n, how much time it last.
    return a sorted list as [id1, id2, id3,... ]
    """
    li_with_rank = ((id, 1.0 * w / ((t + 2) ** 1.5)) for id, w, t in li)
    li_sorted = sorted(li_with_rank, key=itemgetter(1), reverse=True)
    return li_sorted if with_weight else [id for id, r in li_sorted]


def hours(time_delta):
    return time_delta.days * 24 + time_delta.seconds / 3600


def datetime_ranker(li):
    """
    li: [(id, w, t)...] id: obj , w: from 1 to n , t: datetime
    """
    now = datetime.now()
    nli = ((id, w, hours(now - t)) for id, w, t in li)
    return ranker(nli)


# def rank(source_set, target_li, target_N, ):
#    "keep target_N and iterate sources"
#    mixed = source_set.intersection(target_li)
#    n = len(mixed)
#    B = target_N * sum_N
#    if B==0: print "zero base:", target_N, sum_N
#    return 1.0 * (n * n) / B if B else 0

def int_rank(n, sum_n):
    return int(1000.0 * n * n / sum_n)


#    return int(100*rank(target_li, target_N, sum_N))



def bhh_ranker(li):
    """
    li: [(id, score, date), (id, score, date),...]
    id: obj id
    t1: score:
    t2: date.
    """
    bhh_time = datetime(year=1989, month=9, day=7, hour=9, minute=12)

    def bhh_seconds(date):
        td = date - bhh_time
        return td.days * 86400 + td.seconds

    def hot(score, date):
        order = log(max(score, 1), 10)
        seconds = bhh_seconds(date) - 667323000  # bhh_seconds(datetime(2010,10,31))
        return round(order + seconds / 700000.0, 7)

    li_ranked = [(id, hot(s, d)) for id, s, d in li]
    li_sorted = sorted(li_ranked, key=itemgetter(1), reverse=True)
    return [id for id, r in li_sorted]


def calculate_hot_score(n_digg, n_comment, date):
    bhh = datetime(year=1989, month=9, day=7, hour=9, minute=12)

    def bhh_seconds(dt):
        td = dt - bhh
        return td.days * 86400 + td.seconds

    score = max(0.5 * n_comment + n_digg, 1)
    order = log(score, 10)
    seconds = bhh_seconds(date) - 667323000
    return round(order + seconds / 70000.0, 7)


def hot_score(n_vote, date):
    bhh = datetime(year=1989, month=9, day=7, hour=9, minute=12)

    def bhh_seconds(dt):
        td = dt - bhh
        return td.days * 86400 + td.seconds

    order = log(n_vote + 1, 10)
    seconds = bhh_seconds(date) - 667323000
    return order + seconds / 70000.0


def chain_iterable(iterable_li):
    for it in iterable_li:
        for i in it: yield i


def really(probability):
    # condition is a 0-1 real number.
    return random() < probability


def yesterday():
    return get_date_before()


def yesterday_datetime():
    d = yesterday()
    return datetime(d.year, d.month, d.day)


def get_date_before(d=1):
    return date.today() - timedelta(days=d)


def get_date_after(day, d=1):
    return day + timedelta(days=d)


def renumerate(iterable):
    return ((j, k) for (k, j) in enumerate(iterable))


def listerizer(iterable):
    return ([p] for p in iterable)


def tuplerizer(iterable):
    return ((p,) for p in iterable)


def return_zero(): return 0


def return_one(): return 1


def return_defaultdict_dict():
    return defaultdict(dict)


def return_defaultdict_int():  # for pickle to omit using lambda
    return defaultdict(int)


def return_defaultdict_list():
    return defaultdict(list)


def return_defaultdict_set():
    return defaultdict(set)


def iterdoubledict(d):
    for i in d:
        for j in d[i]:
            yield d[i][j]


def weighted_choice(w_li):
    """
    :param w_li:[(item: weight), ...]
    """
    indexes = []
    _sum = 0
    for i, w in w_li:
        _sum += w
        indexes.append(_sum)
    r = random() * _sum
    for i, index in enumerate(indexes):
        if r < index:
            return w_li[i]


def gen_weighted_list(li, weight_fun):
    return ((x, weight_fun(x)) for x in li)


def weighted_sample_li(li, n, w_fun):
    w_li = gen_weighted_list(li, w_fun)
    return weighted_sample(w_li, n)


def weighted_sample_dict(di, n):
    w_li = di.items()
    return weighted_sample(w_li, n)


def weighted_sample(li, n):
    if n >= len(li): return li
    results = []
    nli = li[:]
    for _ in range(n):
        i = weighted_choice(nli)
        results.append(i)
        del nli[i]
    return results


def wsample(wlist, with_weight=False):
    """sample one item from wlist by weight"""
    if isinstance(wlist, dict):
        wlist = wlist.items()
    lst = []
    s = 0
    for val, w in wlist:
        s += w
        lst.append(s)
    r = random() * s
    idx = bisect_left(lst, r)
    idx = min(idx, len(lst) - 1)  # in case not found
    if with_weight:
        return wlist[idx]
    return wlist[idx][0]


def _isupper_(x):
    return x.isupper()


def get_upper(_str):
    return filter(_isupper_, _str)


class V(object):
    def __init__(self, v=0):
        self.value = v

    def add(self, other_value=1):
        self.value += other_value
        return self.value


class Counter_in_iteration(object):
    def __init__(self, interval=1000000, inter_f=gc.collect):
        self.ct = 0
        self.it = interval
        self.f = inter_f

    def count(self):
        self.ct += 1
        if self.ct % self.it == 0:
            self.f()
            print(self.ct, now_str())

    def end_count(self):
        self.f()
        print(self.ct, now_str())

    def __repr__(self):
        return "%s" % self.ct


class QueueSet(object):
    def __init__(self, iter):
        self.queue = list(iter)
        self.aset = set(self.queue)
        self.index = 0
        self.size = len(self.queue)

    def add(self, a):
        if a not in self.aset:
            self.aset.add(a)
            self.queue.append(a)
            self.size += 1

    def pop(self):
        if self.index >= self.size: return None
        self.index += 1
        return self.queue[self.index - 1]

    def add_all(self, li):
        for a in li:
            self.add(a)


def most_common(dd, n):
    ss = sorted(dd.items(), key=itemgetter(1), reverse=True)
    return ss[:n]


def update_seq(dd, seq):
    for item in seq: dd[item] += 1


def update_dic(dd, dic):
    for item, v in dic.items():
        dd[item] += v


def get_similarity_by_set(all_sets, min_score=0.1):
    scores = defaultdict(return_defaultdict_int)
    max_score = 0

    for a in all_sets:
        set_a = all_sets[a]
        N = len(set_a)
        for b in all_sets:
            if b == a: continue
            set_b = all_sets[b]
            M = len(set_b)
            n = len(set_b.intersection(set_a))
            if n > 0:
                score = 1.0 * (n * n * n) / (N * M)
                if score > max_score: max_score = score
                if score > min_score: scores[a][b] = score
    for a in scores:
        for b in scores[a]:
            scores[a][b] /= max_score

    return scores


def get_friend_graph(include_fromid):
    import gc

    gc.disable()
    import os

    friend_graph = {}
    PATH = '/mfs/alg/dbsync/contact/'

    res = os.popen('ls %s' % PATH)
    files = [r.strip() for r in res.readlines()]
    for f in files:
        count = 0
        fp = open(PATH + f, 'r')
        for line in fp:
            count += 1
            from_id, to_id, st = line.split('\t')
            if from_id in include_fromid:
                friend_graph.setdefault(from_id, []).append(to_id)
        fp.close()
    gc.enable()
    return friend_graph


def cal_GI(cate_word_DF, cate_docs, min_word_docs=0):
    """
    calculate Gain of Information
    @param cate_word_DF: cate_word_DF = cate-> (words-> DF)
    @param cate_docs: cate_docs = cate -> (docs in cat)
    """
    total_doc = sum(cate_docs.values())
    cate_prob = dict((c, 1.0 * d / total_doc) for c, d in cate_docs.items())  # 每一类的基准概率。
    E_doc = -1.0 * sum(cate_prob[c] * log(cate_prob[c], 2) for c in cate_prob)
    total_words = set()
    for w in cate_word_DF.values():
        total_words.update(w.keys())

    word_GI = []
    for w in total_words:
        wDF = dict((c, cate_word_DF.setdefault(c, {}).setdefault(w, 0)) for c in cate_word_DF.keys())
        w_docs = sum(wDF.values())
        if w_docs < min_word_docs: continue  # wf太少的词忽略。
        p_word = 1.0 * w_docs / total_doc
        np_word = 1.0 - p_word
        E_p = -1.0 * sum(p * log(p + 0.00001, 2) for p in (1.0 * wDF[c] / cate_docs[c] for c in wDF))
        E_np = -1.0 * sum(
            p * log(p + 0.00001, 2) for p in (1.0 * (cate_docs[c] - wDF[c]) / (total_doc - w_docs) for c in wDF))
        GI = E_doc - (p_word * E_p + np_word * E_np)
        word_GI.append((w, GI))
    word_GI.sort(key=itemgetter(1), reverse=True)
    return word_GI


if __name__ == "__main__":
    print("hello")

