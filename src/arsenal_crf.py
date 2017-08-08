# -*- coding: utf-8 -*-

import gc
from collections import OrderedDict, defaultdict, Counter
from copy import deepcopy
from functools import reduce
from itertools import chain, groupby, tee
from os import listdir
from zhon.hanzi import punctuation
from string import punctuation as Enpun

import joblib as jl
import pandas as pd
import scipy.stats as sstats
import sklearn_crfsuite
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn_crfsuite import metrics

from .arsenal_logging import basic_logging
from .arsenal_spacy import spacy_batch_processing
from .arsenal_stats import hdf2df, df2dic, df2set, map_dic2df, sort_dic, random_rows
from .arsenal_test import evaluate_ner_result, show_crf_label, evaluate_reslt
from .settings import *

# HEADER_CRF = ['TOKEN', 'NER', 'POS']
HEADER_CRF = ['TOKEN', 'NER']

HEADER_REPORT = ['tag', 'precision', 'recall', 'f1', 'support']
CHAR_PUN = punctuation + Enpun


##############################################################################
def tag_convert(filename, mode):
    '''
    data processing:transform the data into [Token,tag]
    :param filename: input file
    :return: dataframe:[Token,tag]
    '''
    text_list = []
    tag_list = []
    with open(filename,encoding='utf-8') as file:
        for line in file.readlines():
            if not line.strip():
                continue
            line_tag, line_list = line_process(line, mode)
            tag_list.append(line_tag)
            # line_list.append(['##END'])
            text_list.append(line_list)
    text_list = [k for j in text_list for i in j for k in i]
    tag_list = [k for j in tag_list for i in j for k in i]
    text_df = pd.DataFrame(list(zip(text_list, tag_list)))
    text_df.columns = HEADER_CRF
    # text_df.to_csv(DATA_PROCESS + filename, header=False, index=False)

    return text_df


def line_process(line, mode):
    '''
    transform the line into [word,tag]
    :param line:
    :param mode: train:spilt by space or test:normal text
    :return: [word,tag]
    '''
    line_tag = []
    if mode == 'train':
        line_list = line.strip('\n').strip().split(' ')
    elif mode == 'test':
        line_list = line.strip('\n').strip().split()
    for token in line_list:
        if len(token) == 1:
            if token not in CHAR_PUN:
                line_tag.append(['U-w'])
            elif token in CHAR_PUN:
                line_tag.append('O')
        elif len(token) == 2:
            line_tag.append(['B-w', 'L-w'])
        else:
            line_tag.append(['B-w'])
            line_tag.append(['I-w'] * (len(token) - 2))
            line_tag.append(['L-w'])
    line_tag.append('O')
    line_list.append(['##END'])

    return line_tag, line_list



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
    index = [i for i, x in enumerate(df['TOKEN'].tolist()) if x == delim]
    delimiter = tuple(df[df.iloc[:, 0] == delim].iloc[0, :].tolist())
    sents = zip(*[df[i].tolist() for i in df.columns])  # Use * to unpack a list
    sents = (list(x[1]) for x in groupby(sents, lambda x: x == delimiter))
    result = [i for i in sents if i != [] and i != [(delimiter)]]
    return result, index


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
    sentence = [i[0] for i in sent]
    for j in range(window_size):
        win = window_size - j
        if i >= win:
            features.update({'previous' + str(win) + 'cur': ''.join(sentence[i - win:i + 1])})
        elif i < win - window_size + 1:
            features['BOS'] = True
        if i < len(sent) - win:
            features.update({'cur_' + 'next' + str(win): ''.join(sentence[i:win + 1 + i])})
        elif i - window_size + 1 >= len(sent) - win:
            features['EOS'] = True
    return features


def sent2features(line, feature_conf, hdf_key, window_size):
    return [word2features(line, i, feature_conf, hdf_key, window_size) for i in range(len(line))]


def sent2labels(line):
    return [i[1] for i in line]  # Use the correct column


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
    y_set = [sent2labels(s) for s in in_data]
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


def search_param(X_train, y_train, crf, params_space, f1_scorer, cv=3, iteration=50):
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


##############################################################################

# todo add /n according to the index
def token_text(test_df, y_pred, space_index):
    text_df = test_df[test_df["TOKEN"] != '##END']
    # flattern_test = [i for j in y_pred for i in j]
    # crf_results = list(zip(text_df["TOKEN"].tolist(), flattern_test))
    crf_results = list(zip(text_df["TOKEN"].tolist(), y_pred))
    # text=[]
    # for i, index in enumerate(space_index):
    #     if i == 0:
    #         sentence = crf_results[:index]
    #     else:
    #         sentence = crf_results[space_index[i - 1] - (i - 1):index - i]
    #     text.append(' '.join(token_generate(sentence)))
    text = [' '.join(token_generate(crf_results[:index])) if i == 0 else ' '.join(
        token_generate(crf_results[space_index[i - 1] - (i - 1):index - i])) for i, index in enumerate(space_index)]
    pd.DataFrame(text).to_csv(PREDICT_FILE, header=False, index=False)
    basic_logging('writting data ends')


#
def token_generate(sentence):
    token = ''
    phrase = []
    for word_tag in sentence:
        if word_tag[1] not in ['L-w', 'U-w', 'O']:
            token += word_tag[0]
        else:
            token += word_tag[0]
            phrase.append(token)
            token = ''
    return phrase


def crf2dict(crf_result):
    """
    :param crf_result: [[token, pos, ner]]
    :return: DICT {ENTITY##NER##COUNT:[]}
    """
    # clean_sent = [(token, ner) for token, ner, _ in crf_result if token != '##END']
    clean_sent = [(token, ner) for token, ner in crf_result]
    # ner_candidate = [(token, ner) for token, ner in clean_sent if ner[0] != 'O']
    ner_candidate = [(token, ner) for token, ner in clean_sent]
    ner_index = [i for i in range(len(ner_candidate)) if
                 ner_candidate[i][1] == 'U-w' or ner_candidate[i][1] == 'L-w' or ner_candidate[i][1] == 'O']
    new_index = [a + b for a, b in enumerate(ner_index)]
    ner_result = extract_ner(ner_candidate, new_index)
    return ner_result


def extract_ner(ner_candidate, new_index):
    """
    :param ner_candidate:
    :param new_index:
    :return: DICT {ENTITY##NER##COUNT:[]}
    """
    new_candidate = deepcopy(ner_candidate)
    for i in new_index:
        new_candidate[i + 1:i + 1] = [('##split', '##split')]
    grouped_ner = [list(x[1]) for x in groupby(new_candidate, lambda x: x == ('##split', '##split'))]
    ner_result = [''.join([k for k, v in group]) for group in grouped_ner if group != [('##split', '##split')]]
    basic_logging('add index ends')
    text = []
    sentence = []
    # todo see the structure of grouped_ner
    for token in ner_result:
        if token != '##END':
            sentence.append(token)
        else:
            text.append(sentence)
            sentence = []
    text_line = [' '.join(i) for i in text]
    pd.DataFrame(text_line).to_csv(PREDICT_FILE, header=False, index=False)
    basic_logging('writting data ends')

    return ner_result


def prepare_remap(remap_f):
    """
    Prepare remap dict from file
    :param remap_f:
    :return:
    """
    remap_df = pd.read_csv(remap_f, engine='c')
    remap_dic = {k: v for (k, v) in zip(remap_df.Wrong.tolist(), remap_df.Correct.tolist())}
    return remap_dic


def remap_ner(ner_phrase, remap_dic):
    """
    Add rule-based mapping
    :param ner_phrase: DICT {ENTITY##NER##COUNT:[]} processed ner phrases
    :param remap_dic: DICT DICT {ENTITY##WRONG_NER: ENTITY##CORRECT_NER}
    :return: DICT {ENTITY##NER##COUNT:[]}
    """
    temp = (('##'.join(i.split('##')[:2]), i.split('##')[2]) for i in ner_phrase.keys())
    result = ((remap_dic[k], v) if k in remap_dic.keys() else (k, v) for k, v in temp)
    final = {'##'.join((m, n)): [] for m, n in result}
    return final


def crf_result2json(crf_result, raw_df, col, remap_dic):
    ner_phrase = crf2dict(crf_result)
    ner_phrase = remap_ner(ner_phrase, remap_dic) if remap_dic else ner_phrase
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


##############################################################################

# Modules
#line_crf_fit
def line_crf_fit(df, crf, f_dics, feature_conf, hdf_key, window_size, result_f):
    test = batch_add_features(df, f_dics)
    # print(test)
    basic_logging('adding test features ends')
    test_sents, index_line = df2crfsuite(test)
    basic_logging('converting to test crfsuite ends')
    X_test, y_test = feed_crf_trainer(test_sents, feature_conf, hdf_key, window_size)
    X_a, X_b = tee(X_test, 2)
    y_a, y_b = tee(y_test, 2)
    basic_logging('test conversion ends')
    y_pred = crf.predict(X_a)
    basic_logging('testing ends')
    if result_f:
        result, indexed_ner = evaluate_ner_result(y_pred, y_a)
        result.to_csv(result_f, index=False)
        basic_logging('testing ends')
    return y_pred, list(X_b), list(y_b)

def module_crf_fit(df, crf, f_dics, feature_conf, hdf_key, window_size, result_f):
    test = batch_add_features(df, f_dics)
    basic_logging('adding test features ends')
    test_sents, index_line = df2crfsuite(test)
    basic_logging('converting to test crfsuite ends')
    X_test, y_test = feed_crf_trainer(test_sents, feature_conf, hdf_key, window_size)
    X_a, X_b = tee(X_test, 2)
    y_a, y_b = tee(y_test, 2)
    basic_logging('test conversion ends')
    y_pred = crf.predict(X_a)
    basic_logging('testing ends')
    if result_f:
        result, indexed_ner = evaluate_ner_result(y_pred, y_a)
        result.to_csv(result_f, index=False)
        basic_logging('testing ends')
    return y_pred, list(X_b), list(y_b), index_line


def module_crf_train(train_df, f_dics, feature_conf, hdf_key, window_size):
    train_df = batch_add_features(train_df, f_dics)
    basic_logging('adding train features ends')
    train_sents, _ = df2crfsuite(train_df)
    # print(train_sents)
    basic_logging('converting train to crfsuite ends')
    X_train, y_train = feed_crf_trainer(train_sents, feature_conf, hdf_key, window_size)
    X_a, X_b = tee(X_train, 2)
    y_a, y_b = tee(y_train, 2)
    basic_logging('computing train features ends')
    crf = train_crf(X_a, y_a)
    return crf, list(X_b), list(y_b)


def module_crf_cv(crf, X_train, y_train, cv, iteration):
    labels = show_crf_label(crf)
    params_space = make_param_space()
    f1_scorer = make_f1_scorer(labels)
    gc.collect()
    basic_logging('cv begins')
    rs_cv = search_param(X_train, y_train, crf, params_space, f1_scorer, cv, iteration)
    basic_logging('cv ends')
    best_predictor = rs_cv.best_estimator_
    return best_predictor


def module_prepare_news_jsons(in_folder, col, row_count, col_names):
    raw_list = (pd.read_json('/'.join((in_folder, in_f)), col_names) for in_f in listdir(in_folder))
    basic_logging('reading files ends')
    raw_df = pd.concat(raw_list, axis=0)
    random_df = random_rows(raw_df, row_count)
    basic_logging('selecting lines ends')
    random_df['content'] = random_df[col].apply(lambda x: x['content'])
    parsed_data = chain.from_iterable(spacy_batch_processing(random_df, '', 'content', ['content'], 'crf'))
    prepared_df = pd.DataFrame(list(parsed_data))
    basic_logging('extracting data ends')
    return prepared_df


def module_prepare_folder(col, in_folder, ner_tags):
    train_df = pd.concat(
        [process_annotated('/'.join((in_folder, in_f)), col) for in_f in listdir(in_folder) if 'train' in in_f], axis=0)
    print(train_df.info())
    test_df = pd.concat(
        [process_annotated('/'.join((in_folder, in_f)), col) for in_f in listdir(in_folder) if 'test' in in_f], axis=0)
    basic_logging('loading data ends')
    if ner_tags:
        train_df = merge_ner_tags(train_df, 'NER', ner_tags)
        test_df = merge_ner_tags(test_df, 'NER', ner_tags)
    return test_df, train_df
