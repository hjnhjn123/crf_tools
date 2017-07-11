# -*- coding: utf-8 -*-

from collections import OrderedDict, defaultdict, Counter
from copy import deepcopy
from functools import reduce
from itertools import chain, groupby

import joblib as jl
import pandas as pd
import scipy.stats as sstats
import sklearn_crfsuite
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn_crfsuite import metrics

from .arsenal_logging import basic_logging
from .arsenal_stats import hdf2df, df2dic, df2set, map_dic2df, sort_dic
from .arsenal_test import evaluate_ner_result

HEADER_CRF = ['TOKEN', 'NER', 'POS']

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


def batch_loading(feature_hdf, hdf_keys):
    """
    :param crf_f: model file
    :param feature_hdf: feature dict file
    :param hdf_keys: hdfkey to extract dicts
    :return: 
    """
    loads = hdf2df(feature_hdf, hdf_keys)
    f_dics = prepare_features(loads)
    return f_dics


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
    word, pos, other_features = word_tuple[0], word_tuple[2], word_tuple[3:]
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
    return (word2features(line, i, feature_conf, hdf_key, window_size) for i in range(len(line)))


def sent2labels(line):
    return (i[1] for i in line)  # Use the correct column


##############################################################################


# CRF training

def feed_crf_trainer(in_data, X, hdf_key, window_size):
    """
    :param in_data: converted data
    :param X: feature conf
    :param hdf_key: hdf keys
    :param window_size: window size
    :return:
    """
    X_set = (sent2features(s, X, hdf_key, window_size) for s in in_data)
    y_set = (sent2labels(s) for s in in_data)
    return X_set, y_set


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


def crf_train(train_df, f_dics, feature_conf, hdf_key, window_size):
    train_df = batch_add_features(train_df, f_dics)
    basic_logging('adding train features ends')
    train_sents = df2crfsuite(train_df)
    basic_logging('converting train to crfsuite ends')
    X_train, y_train = feed_crf_trainer(train_sents, feature_conf, hdf_key, window_size)
    basic_logging('computing train features ends')
    crf = train_crf(X_train, y_train)
    return crf, X_train, y_train


##############################################################################


# CRF predicting


def crf_predict(crf, test_sents, X_test):
    """
    :param crf: crf model
    :param test_sents:
    :param X_test:
    :return:
    """
    X_test = list(X_test)
    result = crf.predict(X_test)
    length = len(list(test_sents))
    crf_result = (
        [((test_sents[j][i][0], result[j][i], test_sents[j][i][2])) for i in range(len(test_sents[j]))] for j in
        range(length))
    # crf_result = [((test_sents[j][i][0], test_sents[j][i][2]) + (result[j][i],)) for i in range(len(test_sents[j]))] for j in
    # range(length))
    crf_result = [i + [('##END', '###', 'O')] for i in crf_result]
    return list(chain.from_iterable(crf_result))


def crf_fit(df, crf, f_dics, feature_conf, hdf_key, window_size, result_f):
    test = batch_add_features(df, f_dics)
    basic_logging('adding test features ends')
    test_sents = df2crfsuite(test)
    basic_logging('converting to test crfsuite ends')
    X_test, y_test = feed_crf_trainer(test_sents, feature_conf, hdf_key, window_size)
    basic_logging('test conversion ends')
    y_pred = crf.predict(X_test)
    basic_logging('testing ends')
    if result_f:
        result, indexed_ner = evaluate_ner_result(y_pred, y_test)
        result.to_csv(result_f, index=False)
        basic_logging('testing ends')
    return y_pred


##############################################################################


def crf2dict(crf_result):
    """
    :param crf_result: [[token, pos, ner]]
    :return: 
    """
    clean_sent = [(token, ner) for token, ner, _ in crf_result if token != '##END']
    ner_candidate = [(token, ner) for token, ner in clean_sent if ner[0] != 'O']
    ner_index = [i for i in range(len(ner_candidate)) if
                 ner_candidate[i][1][0] == 'U' or ner_candidate[i][1][0] == 'L']
    new_index = [a + b for a, b in enumerate(ner_index)]
    ner_result = extract_ner(ner_candidate, new_index)
    return ner_result


def extract_ner(ner_candidate, new_index):
    new_candidate = deepcopy(ner_candidate)
    for i in new_index:
        new_candidate[i + 1:i + 1] = [('##split', '##split')]
    grouped_ner = [list(x[1]) for x in groupby(new_candidate, lambda x: x == ('##split', '##split'))]
    ner_result = []
    for group in grouped_ner:
        if group != [('##split', '##split')]:
            phrase = ' '.join([k for k, v in group])
            tag = [v.split('-')[1] for k, v in group][0]
            ner_result.append('##'.join((phrase, tag)))
    ner_counts = Counter(ner_result)
    final_result = {'##'.join((k, str(v))): [] for k, v in ner_counts.items()}
    return final_result


def crf_result2json(crf_result, raw_df, col):
    ner_phrase = crf2dict(crf_result)
    raw_df[col].to_dict()[0]['ner_phrase'] = ner_phrase
    raw_df = raw_df.drop(['content'], axis=1)
    json_result = raw_df.to_json(orient='records', lines=True)
    return json_result


##############################################################################


def merge_ner_tags(df, col, ner_tags):
    tags = df[col].unique()
    tag_dicts = [dict([(i, i) if i.endswith(t) else (i, 'O') for i in tags]) for t in ner_tags]
    dic = reduce(merge_dict_values, tag_dicts)
    df[col] = df[col].map(dic)
    return df


def merge_dict_values(d1, d2, tag='O'):
    dd = defaultdict()
    for k, v in d1.items():
        if v != d2[k]:
            dd[k] = v if v != tag else d2[k]
        else:
            dd[k] = tag
    return dd


def voting(crf_results, head=HEADER_CRF):
    crf_dfs = [pd.DataFrame(crf_list, columns=head).add_suffix('_' + name) for name, crf_list in crf_results.items()]
    combined = pd.concat(crf_dfs, axis=1)
    cols = [i for i in combined.columns if i.startswith('NER')]
    # to_vote = combined[cols].apply(tuple, axis=1).tolist()  # convert a df to zipped list
    to_vote = sort_dic({col.split('_')[1]: combined[col].tolist() for col in cols})
    if len(cols) == 2:
        vote_result = merge_list_dic(to_vote)
    elif len(cols) > 2:
        specific = {name: lst for name, lst in to_vote.items() if name != 'NER_GEN'}
        pass  # todo fix more than three models
    return list(zip(combined.iloc[:, 0].tolist(), vote_result, combined.iloc[:, 2].tolist(), ))


def merge_list_dic(list_dict):
    l1, l2 = list_dict.values()
    name1, name2 = list_dict.keys()
    l2 = ['O' if i.endswith(name1) else i for i in l2]
    return [l1[i] if l1[i].endswith(name1) else l2[i] for i in range(len(l1))]


def load_multi_models(model_fs):
    model_dics = {model.split('.')[0].split('_')[-1]: jl.load(model) for model in model_fs}
    return model_dics
