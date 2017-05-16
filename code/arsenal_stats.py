# -*- coding: utf-8 -*-
from __future__ import (unicode_literals, print_function, division)

import hashlib
import random
from collections import OrderedDict, defaultdict
from datetime import date, datetime
from functools import reduce
from itertools import islice
from json import loads
from operator import itemgetter

import numpy as np
import pandas as pd
from pandas import ExcelWriter
from yaml import load


##########################################################################################


# CSV processing


def get_column_index(column, col_list):
    """
    :param column: the COLUMN_CHAT constant
    :param col_list: define wanted columns here
    :return: a list of indices
    """
    col_index_dic = {v: k for (k, v) in enumerate(column)}
    return [int(col_index_dic[col]) for col in col_list]


def csv2pd(in_file, needed_columns, sep=",", quote=3, engine='c'):
    """
    user Function get_column_index to get a list of column indices
    :param in_file: a csv_file
    :param needed_columns: the full column names of the csv
    :param original_columns: a predefined column header
    :param sep: ',' by default
    :param quote: exclude quotation marks
    :param engine: choose engine for reading data
    :return: a trimmed pandas table
    """
    return pd.read_csv(in_file, usecols=needed_columns, sep=sep, quoting=quote,
                       engine=engine)


def get_header(in_file, sep=","):
    in_file = open(in_file)
    result = next(in_file).strip('\n\r').split(sep)
    in_file.close()
    return result


def quickest_read_csv(in_file, column_names):
    """
    param: in_file: csv file
    """
    data = pd.read_csv(in_file, usecols=column_names, engine='c', quoting=0, sep=',')
    return data


##########################################################################################


## HDF5 Processing


def df2hdf(out_hdf, dfs, hdf_keys, mode='a'):
    """
    Store single or multiple dfs to one hdf5 file
    :param dfs: single of multiple dfs
    :param out_hdf: the output file
    :param hdf_keys: [key for hdf]
    """
    for j, k in zip(dfs, hdf_keys):
        j.to_hdf(out_hdf, k, table=True, mode=mode)


def hdf2df(in_hdf, hdf_keys):
    """
    Read a hdf5 file and return all dfs
    :param in_hdf: a hdf5 file 
    :param hdf_keys: 
    :return a dict of df
    """
    return {i: pd.read_hdf(in_hdf, i) for i in hdf_keys}


##########################################################################################


# Excel processing


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


def df2excel(df, path, name):
    """
    :param df: a pd.DataFrame
    :param path: path to excel file
    :param name: name of the file
    :return: an excel file
    """
    return df.to_excel(path, sheet_name=name)


##########################################################################################


# REGEX

def get_re_or_from_iter(uni_iter):
    """
    :param uni_iter: an iterator of unicode strings: ['\u5b87\u513f|tang', '\u8d85\u7ea7']
    :return: re.compile(ur'(\u5b87\u513f|tang|\u8d85\u7ea7\u5988\u5988--Nancy)')
    """
    return compile("(" + '|'.join(uni_iter) + ")")


##########################################################################################


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


def rename_series(df, old_col, new_col):
    """
    The rename in pandas is not easy to remember
    :param df: a pd df
    :param old_col: old column name
    :param new_col: new column name
    :return: a pd with new column name
    """
    return df.rename(columns={old_col: new_col})


##########################################################################################


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


def list2df(lst, column_names):
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
    return ', '.join(
        [': '.join((k, str(round(v, precision)))) for (k, v) in dic[start:end]])


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


def dicts2df(dics, column_names):
    """
    Combine dics to a DataFrame by keys
    :param dics: [dic1, dic2...]
    :param column_names: [col1, col2...]
    :return: pd.DataFrame
    """
    return pd.DataFrame(dics, columns=column_names).T


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


def check_df(df, column, value):
    """
    :param df: a pandas DataFrame
    :param column: define column
    :param value: define the value
    :return: a extracted DataFrame
    """
    return df.loc[column == value]


def nested_dic2df(dic, col_list):
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


def dic2df(dic, cols):
    """
    :param dic:
    :param cols: set column names
    :return: a pandas DataFrame: col1: key, col2: original values
    """
    return pd.DataFrame(dic.items(), columns=cols)


def dic2extended_df(dic, cols):
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


def add(a, b):
    return a + b


def combine_multi_series(df, col_name=['Extracted']):
    return pd.DataFrame(reduce(add, [df[col] for col in df]), columns=col_name)


def remap_series(df, col, new_col, label_set, new_label, misc='MISC'):
    """
    :param df:
    :param col: the column to be changed
    :param new_col: the new column
    :param label_set: labels to be changed
    :param new_label: new label
    :param misc: set the misc label
    :return: the original df with new_col
    """
    df[new_col] = np.where(df[col].isin(label_set), new_label, misc)
    return df


def line_file2set(in_file):
    """
    | Reading a line-based file, and converting it to a feature set
    :param in_file:
    :return: (feature1, feautre2...)
    """
    with open(in_file, 'r') as f:
        return set(i.strip('\n\r') for i in f)


def line_file2dict(in_file):
    """
    | Reading a line-based csv file, and converting it to a feature dic
    :param in_file:  token,value
    :return: {token: value}

    """
    with open(in_file, 'r') as data:
        result = defaultdict()
        for i in data:
            line = i.split(',')
            result[line[0]] = line[1].strip('\r\n')
        return result


def df2set(df, title=False):
    return {i for j in df.as_matrix() for i in j} if title == False else \
        {i.title() for j in df.as_matrix() for i in j}


def df2list(df):
    return [i for j in df.as_matrix() for i in j]


def df2dic(df):
    """
    use pd.DataFrame.iloc to extract specific columns or rows
    :param df: 
    :return: 
    """
    return {k: v for (k, v) in zip(df.iloc[:, 0], df.iloc[:, 1])}


########################################################################################################################


# JSON Processing


def json2list(corpus):
    return (loads(line) for line in corpus)


def json2pd(in_file, col_list, lines=True):
    """
    :param in_file: a json file
    :param col_list: set the extracted columns
    :param lines: True if the file is line-based
    :return: a pd df
    """
    data = pd.read_json(in_file, lines=lines)
    result = data[col_list]
    return result


##########################################################################################


# Math


def divide_series(col1, col2, index, dec=2):
    """
    :param col1: a pandas DataFrame column
    :param col2: a pandas DataFrame column
    :param index: set index here
    :param dec: set decimal for rounding
    :return: a pandas Series
    """
    return pd.Series.round(
        pd.Series(100.0 * pd.Series(col1) / pd.Series(col2), index=index), decimals=dec)


##########################################################################################


# Time processing


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
    return np.datetime64(delimiter.join((str(time.year), str(time.month))),
                         dtype='datetime64[D]')


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


##########################################################################################


# Randomization


def random_pick(df, size=10):
    return random.sample(range(0, len(df) - 1), size)


def random_rows(df, size, col_name):
    row_number = random_pick(df, size)
    result = df.iloc[row_number]
    return result


##########################################################################################


# Config


def load_yaml_conf(conf_f):
    with open(conf_f, 'r') as f:
        result = load(f)
    return result


##########################################################################################

def hashit(string):
    return hashlib.md5(string.encode('utf-8')).hexdigest()
