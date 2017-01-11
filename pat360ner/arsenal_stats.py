# -*- coding: utf-8 -*-
from __future__ import (unicode_literals, print_function, division)

from collections import OrderedDict
from datetime import date, datetime
from itertools import islice
from json import loads
from operator import itemgetter

import numpy as np
import pandas as pd
from enum import Enum
from pandas import ExcelWriter
from functools import reduce
from configparser import ConfigParser


########################################################################################################################


# File Reading and Writing


def pd2file(pandas_df, out_file):
    """
    write a pandas DataFrame to a file
    """
    oo = open(out_file, 'w')
    pd.DataFrame.to_csv(pandas_df, oo)


def get_column_index(column, col_list):
    """
    :param column: the COLUMN_CHAT constant
    :param col_list: define wanted columns here
    :return: a list of indices
    """
    col_index_dic = {v: k for (k, v) in enumerate(column)}
    return [int(col_index_dic[col]) for col in col_list]


def csv2pd(in_file, column_constant, column_names, sep=",", quote=3, engine='python'):
    """
    user Function get_column_index to get a list of column indices
    :param in_file: a csv_file
    :param column_names: [column_names]
    :param column_constant: a predefined column header
    :param sep , by default
    :param quote: exlcude quotation marks
    :param engine: choose engine for reading data
    :return: a trimmed pandas table
    """
    column_indices = get_column_index(column=column_constant, col_list=column_names)
    return pd.read_csv(in_file, usecols=column_indices, sep=sep, quoting=quote, engine=engine)


def get_header(infile, sep=","):
    infile = open(infile)
    res = next(infile).strip('\n\r').split(sep)
    infile.close()
    return res


def get_last_files(base, time_range):
    """
    :param base: e.g.: '/Users/acepor/Work/gemii_data/datapool/input/wechat_wyeth_RoomMsgInfo_raw_'
    :param time_range: get_past_week() or get_past_month()
    :return: [file_paths]
    """
    return [''.join((base, ''.join(i.split('-')), '.csv')) for i in time_range]


def choose_file(in_file, time_format, day, header, sep, engine):
    """
    :param in_file: the file DIR
    :param time_format: set time category
    :param day: set date
    :param header: set header for extraction
    :param sep: set separator for extraction
    :return: a concated DF
    """
    if time_format == 'daily':
        in_file = in_file + day + '.csv'
        return csv2pd(open(in_file, 'r'), get_header(in_file, sep), header, sep=sep, engine=engine)
    elif time_format == 'weekly':
        return pd.concat([csv2pd(open(i, 'r'), get_header(i, sep), header, sep=sep, engine=engine) for i in
                          get_last_files(in_file, get_past_week(day))])
    elif time_format == 'monthly':
        return pd.concat([csv2pd(open(i, 'r'), get_header(i, sep), header, sep=sep, engine=engine) for i in
                          get_last_files(in_file, get_past_month(day))])


def df2single_excel(df, excel_file, length=60000):
    writer = ExcelWriter(excel_file)
    if len(df) < length:
        df.to_excel(writer, 'sheet1', index=False)
        writer.save()
        writer.close()
    else:
        for g, sub in df.groupby(np.arange(len(df)) // length):
            sheet_name = 'sheet%s' % g
            sub.to_excel(writer, sheet_name, index=False)
            writer.save()
            writer.close()


def prepare_keywords(keywords, in_file):
    """
    :param keywords:  set the keyword items here
    :param in_file: a keyword CONF file
    :return: [{keywords}]
    """
    config = ConfigParser()
    config.read(in_file)
    # keyword_lists = [{to_uni(i) for i in config.get('CONF', keyword).split(',')} for keyword in keywords]
    keyword_dic = {keyword: {i for i in config.get('CONF', keyword).split(',')} for keyword in keywords}
    return keyword_dic


########################################################################################################################


# REGEX

def get_re_or_from_iter(uni_iter):
    """
    :param uni_iter: an iterator of unicode strings: ['\u5b87\u513f|tang', '\u8d85\u7ea7\u5988\u5988--Nancy']
    :return: re.compile(ur'(\u5b87\u513f|tang|\u8d85\u7ea7\u5988\u5988--Nancy)')
    """
    return compile("(" + '|'.join(uni_iter) + ")")


########################################################################################################################


# Word Processing


NON_EOL = {ord(u'\n'): u' ', ord(u'\r'): u' '}


def remove_eol(uni_str):
    """
    :param uni_str: a unicode string containing '\n' or '\r' as eol
    """
    return uni_str.translate(NON_EOL)


def clean_list(items, rpls={u"～": u"~"}):
    """
    clean list, remove specific unicode chars
    :param items: item list (item is unicode)
    :param rpls: replacements dict
    :return:
    """
    res = []
    for item in items:
        for k, v in iter(rpls.items()):
            item = item.replace(k, v)
        res.append(item)
    return res


def clean_dataframe(df, columns=["TextMsg"], rpls={u"～": u"~"}):
    """
    clean pandas data frame, specific unicode chars
    :param df: data frame object
    :param columns: columns to clean
    :param rpls: replacements dict
    :return:
    """
    for clm in columns:
        for k, v in iter(rpls.items()):
            df[clm] = df[clm].str.replace(k, v)
    return df


def batch_to_uni(df, col_list):
    """
    :param df: a pd df
    :param col_list: [column names]
    :return: a pd df
    """
    for i in col_list:
        df[i] = df.loc[:, i].str.decode('utf-8')
    return df


def batch_to_str(df, col_list):
    """
    :param df: a pd df
    :param col_list: [column names]
    :return: a pd df
    """
    for i in col_list:
        df[i] = df.loc[:, i].str.encode('utf-8')
    return df


def batch_strip(df, col_list, strip_str):
    """
    :param df: a pd df
    :param col_list: [column names]
    :param strip_str: strings to be stripped, ''.join((strings))
    :return: a pd df
    """
    for i in col_list:
        df[i] = df.loc[:, i].str.rstrip(strip_str)
    return df


def batch_change_type(df, col_list, to_type):
    """
    :param df: a pd df
    :param col_list: [column names]
    :param to_type: targeted_type
    :return: a pd df
    """
    for i in col_list:
        df[i] = df[i].astype(to_type)
    return df


def batch_replace(df, col, sources, target):
    """
    :param df:
    :param col_list: a column list
    :param sources: to be replaced
    :param target: to replace
    :return:
    """
    for j in sources:
        df[col] = df[col].str.replace(j, target)
    return df


def batch_startswith(df, col, patterns):
    for i in patterns:
        df[col] = df[col].str.startswith(i)
    return df


########################################################################################################################


# Data Structure


def append2dic(dic, key, value):
    """
    :param dic: defaultdict(list)
    :param key:
    :param value:
    """
    return dic[key].append(value)


def sort_dic(dic, sort_key=0, rev=False):
    """
    :param dic:
    :param sort_key: 0: sort by key, 1: sort by value
    :param rev: false by default
    :return: sorted {(k, v)}
    """
    return OrderedDict(sorted(iter(dic.items()), key=itemgetter(sort_key), reverse=rev))


def cut_top_dic(dic, sort_key=1, rev=True, cut=0.1):
    sorted_dic = sort_dic(dic, sort_key=sort_key, rev=rev)
    top_list = islice(sorted_dic, cut * len(dic))
    return {k: v for (k, v) in top_list}


def list2pandas_df(lst, column_names):
    """
    :param lst: a matrix-like list
    :param column_names: [col1, col2...]
    :return a pandas DataFrame with column names
    """
    return pd.DataFrame(lst, columns=column_names)


def split_dic(dic, start=None, end=None, precision=3):
    """
    split a dic according to range[start, end]
    """
    return ', '.join([': '.join((k, str(round(v, precision)))) for (k, v) in dic[start:end]])


def join_data(in_file1, in_file2, out_file):
    """
    It joins two file together with tabs as delimiters.
    :param in_file1: a tab-separated file
    :param in_file2: a tab-separated file
    :param out_file: a tab-separated file
    """
    ii1, ii2, oo = open(in_file1, 'r'), open(in_file2, 'r'), open(out_file, 'w')
    for line1, line2 in zip(ii1, ii2):
        oo.write("{}\t{}\n".format(line1.strip('\n'), line2.strip('\n')))
    ii1.close(), ii2.close(), oo.flush(), oo.close()


def get_lists(list1, list2):
    """
    :param [list1]
    :param [list2]
    :return: {k, v}
    """
    return {k: v for (k, v) in zip(list1, list2)}


def dicts2pandas_df(dics, column_names):
    """
    Combine dics to a DataFrame by keys
    :param dics: [dic1, dic2...]
    :param column_names: [col1, col2...]
    :return: pd.DataFrame
    """
    return pd.DataFrame(dics, columns=column_names).T


def pandas_df2excel(df, path, name):
    """
    :param df: a pd.DataFrame
    :param path: path to excel file
    :param name: name of the file
    :return: an excel file
    """
    return df.to_excel(path, sheet_name=name)


def get_pandas_header(df):
    """
    :param df: a pandas DataFrame
    :return: [column names]
    """
    return list(df.head())


def split_evenly(obj, rev=True, pieces=5):
    """
    split an object evenly
    :param obj: any iterable or pandas df
    :param rev: True by default
    :param pieces: assign piece here
    :return: list of np.ndarray
    """
    sort_obj = sorted(obj, reverse=rev)
    return np.array_split(sort_obj, pieces)


def check_pandas_df(df, column, value):
    """
    :param df: a pandas DataFrame
    :param column: define column
    :param value: define the value
    :return: a extracted DataFrame
    """
    return df.loc[column == value]


def unique_multi_col_pd(df, col_list):
    """
    extract multiple columns from a pandas DataFrame, and unique them
    :param df: a pandas DataFrame
    :param col_list: [col1, col2]
    :return:
    """
    return df[col_list].drop_duplicates()


def nested_dic2pd_df(dic, col_list):
    """
    :param dic: {key: {v1, v2}}
    :param col_list:
    :return: {(k, v1), (k, v2)}
    """
    np_array = [(k, m) for (k, v) in iter(dic.items()) for m in v]
    return pd.DataFrame(np.array(np_array), columns=col_list)


def nested_dic2set(dic):
    """
    :param dic: {key: {v1, v2}}
    :return: {(k, v1), (k, v2)}
    """
    return {(k, m) for (k, v) in iter(dic.items()) for m in v}


def dic2pd_df(dic, cols):
    """
    :param dic:
    :param cols: set column names
    :return: a pandas DataFrame: col1: key, col2: original values
    """
    return pd.DataFrame(dic.items(), columns=cols)


def dic2extended_pd_df(dic, cols):
    """
    :param dic:
    :param cols: set column names
    :return: a pandas DataFrame: col1: key, col2: elements in values
    """
    return pd.DataFrame(((k, m) for (k, v) in dic.items() for m in v), columns=cols)


def combine_multi_df(dfs, on, how):
    """
    :param dfs: a list of pandas DataFrame
    :param on: set key here
    :param how: on columns
    :return: a combined padnas DataFrame
    """
    return reduce(lambda left, right: pd.merge(left, right, on=on, how=how), dfs)


def add_series(sr1, sr2):
    return sr1 + sr2


def combine_multi_series(df, col_name=['Extracted']):
    return pd.DataFrame(reduce(add_series, [df[col] for col in df]), columns=col_name)


########################################################################################################################


# JSON Processing


def json2list(corpus):
    return (loads(line) for line in corpus)


########################################################################################################################


# Math


def divide_pd_series(col1, col2, index, dec=2):
    """
    :param col1: a pandas DataFrame column
    :param col2: a pandas DataFrame column
    :param index: set index here
    :param dec: set decimal for rounding
    :return: a pandas Series
    """
    return pd.Series.round(pd.Series(100.0 * pd.Series(col1) / pd.Series(col2), index=index), decimals=dec)


########################################################################################################################


# Time processing


# date time range tool , by spenly


DATE_DELTA_TYPE = Enum("week", "day", "hour", "min", "sec", "ms", "mcs")

FMAP = {
    DATE_DELTA_TYPE.week: "weeks",
    DATE_DELTA_TYPE.day: "days",
    DATE_DELTA_TYPE.hour: "hours",
    DATE_DELTA_TYPE.min: "minutes",
    DATE_DELTA_TYPE.sec: "seconds",
    DATE_DELTA_TYPE.ms: "milliseconds",
    DATE_DELTA_TYPE.mcs: "microseconds",
}


def get_delta(ddtype):
    """
    return a datetime delta function
    :param ddtype: datetime delta type , default DATE_DELTA_TYPE
    :return: function object
    """
    return eval('lambda x: timedelta(%s=x)' % FMAP[ddtype])


def get_date_range(length, start=datetime.today(), out_format="%Y%m%d", by=DATE_DELTA_TYPE.day, step=1):
    """
    get a iter of date range
    :param length: element num
    :param start: start datetime, datetime object, default is today
    :param out_format: element format default "%Y%M%d" => "20160501"
    :param by: element delta type(day or hour etc.) ,default by day
    :param step: element delta value, default is 1
    :return:  a iter of date range object
    """
    return ((start + get_delta(by)(i)).strftime(out_format) for i in range(0, length, step))


def get_date_range_list(length, start=datetime.today(), out_format="%Y%m%d", by=DATE_DELTA_TYPE.day, step=1):
    """
    get a list of date range
    :param length: element num
    :param start: start datetime, datetime object, default is today
    :param out_format: element format default "%Y%M%d" => "20160501"
    :param by: element delta type(day or hour etc.) ,default by day
    :param step: element delta value, default is 1
    :return:  a list of date range object
    """
    return [(start + get_delta(by)(i)).strftime(out_format) for i in range(0, length, step)]


def get_time(ori_time, time_format="%Y-%m-%d %H:%M:%S.%f"):
    """
    :param ori_time: a time string
    :param time_format: define format
    :return: datetime object
    """
    return datetime.strptime(ori_time, time_format)


def extract_time(ori_time, time_format='%Y-%m'):
    """
    :param ori_time: a datetime.date object, e.g.: date.today()
    :param time_format: define format
    :return: str
    """
    return datetime.strftime(ori_time, time_format)


def get_np_time(ori_time):
    return np.datetime64(ori_time)


def get_past_week(time=0):
    """
    :return: [Day of Monday ~ Day of Sunday]
    """
    day = date.today() if time == 0 else datetime.strptime(str(time), '%Y%m%d').date()
    day_of_today, np_today = date.weekday(day), np.datetime64(day)
    first_day = np_today - np.timedelta64(1, 'W')  # use day to subtract one week
    return [str(i) for i in np.arange(first_day + 1, np_today + 1)]


def get_np_this_month(time, delimiter):
    """
    :param time: datetime.date
    :param delimiter: define delimiter here
    :return: numpy.datetime64('Y-M')
    """
    return np.datetime64(delimiter.join((str(time.year), str(time.month))), dtype='datetime64[D]')


def get_past_month(time=0):
    """
    Get date range of the last month at the 1st day of each month
    :return: [First day of last month ~ Last day of last month]
    """
    today = date.today() if time == 0 else datetime.strptime(str(time), '%Y%m').date()
    np_today = get_np_time(extract_time(today))
    last_begin = np_today - np.timedelta64(1, 'M')
    return [str(i) for i in np.arange(last_begin, np_today, dtype='datetime64[D]')]


def get_time_range(time_str, prd=300, freq='S'):
    """
    :param time_str: get
    :param prd:
    :param freq: minute: min, second: S
    :return:
    """
    return pd.date_range(time_str, periods=prd, freq=freq)


def get_now(format='%Y-%m-%d %H:%M:%S:%f'):
    return date.strftime(datetime.now(), format)
