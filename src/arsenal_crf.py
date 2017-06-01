# -*- coding: utf-8 -*-

from collections import OrderedDict, defaultdict
from copy import deepcopy
from itertools import chain, groupby

import joblib as jl
import pandas as pd
import scipy.stats as sstats
import sklearn_crfsuite
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn_crfsuite import metrics

from .arsenal_logging import *
from .arsenal_stats import hdf2df, df2dic, df2set, map_dic2df

HEADER_CRF = ['TOKEN', 'POS', 'NER']

HEADER_REPORT = ['tag', 'precision', 'recall', 'f1', 'support']


##############################################################################


def process_annotated(in_file, col_names=HEADER_CRF):
    """
    :param in_file: CSV file: TOKEN, POS, NER
    :param col_names
    :return: [[sent]]
    """
    data = pd.read_csv(in_file, header=None, engine='c', quoting=0)
    data.columns = col_names
    data = data.dropna()
    return data


def batch_loading(crf_f, feature_hdf, hdf_keys):
    """
    :param crf_f: model file
    :param feature_hdf: feature dict file
    :param hdf_keys: hdfkey to extract dicts
    :return: 
    """
    crf = jl.load(crf_f) if crf_f else None
    loads = hdf2df(feature_hdf, hdf_keys)
    f_dics = prepare_features(loads)
    return crf, f_dics


def prepare_features(dfs):
    """
    :param dfs: a list of pd dfs
    :return: a list of feature sets and feature dicts
    """
    f_sets = {name: df2set(df) for (name, df) in dfs.items() if len(df.columns) == 1}
    f_dics = {name: df2dic(df) for (name, df) in dfs.items() if len(df.columns) == 2}
    f_sets_dics = {k: {i: True for i in j} for (k, j) in f_sets.items()}  # special case
    f_dics.update(f_sets_dics)
    return OrderedDict(sorted(f_dics.items()))


def batch_add_features(df, f_dics):
    """
    # This will generate multiple list of repeated dfs, so only extract the last list
    :param df: a single df
    :param f_dics: feature dicts
    :return: a single df
    """
    df_list = [map_dic2df(df, name, f_dic) for name, f_dic in f_dics.items()]
    return df_list[-1]


def df2crfsuite(df, delim='##END'):
    """
    
    :param df: 
    :param delim: 
    :return:[[(word, label, features)]] 
    """
    delimiter = tuple(df[df.iloc[:, 0] == delim].iloc[0, :].tolist())
    sents = zip(*[df[i].tolist() for i in df.columns])  # Use * to unpack a list
    sents = (list(x[1]) for x in groupby(sents, lambda x: x == delimiter))
    result = [i for i in sents if i != [] and i != [(delimiter)]]
    return result


##############################################################################


def feature_selector(word_tuple, feature_conf, window, hdf_key):
    """
    :param word_tuple: (word, label, features) 
    :param feature_conf: import from setting
    :param window: window size
    :param hdf_key: 
    :return: 
    """
    word, pos, other_features = word_tuple[0], word_tuple[1], word_tuple[3:]
    other_dict = {'_'.join((window, j)): k for j, k in
                  zip(sorted(hdf_key), other_features)}
    feature_func = {name: func for (name, func) in feature_conf.items() if
                    name.startswith(window)}
    feature_dict = {name: func(word) for (name, func) in feature_func.items()}
    feature_dict.update(other_dict)
    feature_dict.update({'_'.join((window, 'pos')): pos})
    return feature_dict


def word2features(sent, i, feature_conf, hdf_key, window_size):
    features = feature_selector(sent[i], feature_conf, 'current', hdf_key)
    features.update({'bias': 1.0})
    if i > window_size - 1:
        features.update(
            feature_selector(sent[i - window_size], feature_conf, 'previous', hdf_key))
    else:
        features['BOS'] = True
    if i < len(sent) - window_size:
        features.update(
            feature_selector(sent[i + window_size], feature_conf, 'next', hdf_key))
    else:
        features['EOS'] = True
    return features


def sent2features(line, feature_conf, hdf_key, window_size):
    return [word2features(line, i, feature_conf, hdf_key, window_size) for i in
            range(len(line))]


def sent2labels(line):
    return [i[1] for i in line]  # Use the correct column


##############################################################################


# CRF training

def feed_crf_trainer(in_data, feature_conf, hdf_key, window_size):
    """
    :param in_data: converted data 
    :param feature_conf: feature conf
    :param hdf_key: hdf keys
    :param window_size: window size
    :return: 
    """
    feature_conf = [sent2features(s, feature_conf, hdf_key, window_size) for s in in_data]
    labels = [sent2labels(s) for s in in_data]
    return feature_conf, labels


def train_crf(X_train, y_train, algm='lbfgs', c1=0.1, c2=0.1, max_iter=100,
              all_trans=True):
    """
    :param X_train:
    :param y_train:
    :param algm:
    :param c1:
    :param c2:
    :param max_iter:
    :param all_trans:
    :return:
    """
    crf = sklearn_crfsuite.CRF(
        algorithm=algm,
        c1=c1,
        c2=c2,
        max_iterations=max_iter,
        all_possible_transitions=all_trans
    )
    return crf.fit(X_train, y_train)


def make_param_space():
    return {
        'c1': sstats.expon(scale=0.5),
        'c2': sstats.expon(scale=0.05),
    }


def make_f1_scorer(labels, avg='weighted'):
    return make_scorer(metrics.flat_f1_score, average=avg, labels=labels)


def search_param(X_train, y_train, crf, params_space, f1_scorer, cv=10, iteration=50):
    rs = RandomizedSearchCV(crf, params_space,
                            cv=cv,
                            verbose=1,
                            n_jobs=-1,
                            n_iter=iteration,
                            scoring=f1_scorer)
    return rs.fit(X_train, y_train)


##############################################################################


# CRF predicting


def crf_predict(crf, test_sents, X_test):
    """
    :param crf: crf model
    :param test_sents:
    :param X_test:
    :return:
    """
    result = crf.predict(X_test)
    length = len(list(test_sents))
    crf_result = (
        [(test_sents[j][i][:2] + (result[j][i],)) for i in range(len(test_sents[j]))] for j in range(length))
    crf_result = [i + [('##END', '###', 'O')] for i in crf_result]
    return list(chain.from_iterable(crf_result))


##############################################################################

def crf_result2dict(crf_result):
    clean_sent = [(token, ner) for token, _, ner in crf_result if token != '##END']
    ner_candidate = [(index, token, ner) for index, (token, ner) in enumerate(clean_sent) if ner[0] != 'O']
    ner_index = [i for i in range(len(ner_candidate)) if
                 ner_candidate[i][2][0] == 'U' or ner_candidate[i][2][0] == 'L']
    new_index = [a + b for a, b in enumerate(ner_index)]
    ner_result = extract_ner_result(ner_candidate, new_index)
    return ner_result


def extract_ner_result(ner_candidate, new_index):
    new_candidate, final_dics = deepcopy(ner_candidate), defaultdict(list)
    for i in new_index:
        new_candidate[i + 1:i + 1] = [('##split', '##split', '##split')]
    ner_result_0 = (
    '##'.join(['##'.join((i[1].strip(), i[2].strip(), str(i[0]))) for i in new_candidate if i[2]]).split('##split'))
    ner_result_1 = ([i.strip(' ') for i in ner_result_0 if i and i != '##'])

    for result in ner_result_1:

        print(result)

        result = result.lstrip('##')
        print(result)
        result_split = result.split('##')[:-1]
        if len(result_split) == 3:
            token, ner = result_split[0], result_split[1][-3:]
            begin_index, end_index = result_split[-1], result_split[-1]
            final_dics['##'.join((token, ner))].append((begin_index, end_index))

        elif len(result_split) > 3:
            token, ner = result_split[0::3], result_split[1][-3:]
            begin_index, end_index = result_split[2], result_split[-1]
            print('##'.join((' '.join(token), ner)))
            final_dics['##'.join((' '.join(token), ner))].append((begin_index, end_index))

    final_lens = {k: str(len(v)) for (k, v) in final_dics.items()}
    final_result = {'##'.join((k, final_lens[k])): v for (k, v) in final_dics.items()}

    return final_result


def crf_result2json(crf_result, raw_df, col):
    ner_phrase = crf_result2dict(crf_result)
    raw_df[col].to_dict()[0]['ner_phrase'] = ner_phrase
    raw_df = raw_df.drop(['content'], axis=1)
    json_result = raw_df.to_json(orient='records', lines=True)
    return json_result
