# -*- coding: utf-8 -*-

from collections import Counter
from copy import deepcopy
from itertools import chain, groupby
import six
import imp
import re
import joblib as jl
import scipy.stats as sstats
import sklearn_crfsuite
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn_crfsuite import metrics

from arsenal_stats import *
from arsenal_logging import *

HEADER_CRF = ['tag', 'precision', 'recall', 'f1', 'support']
LABEL_COMPANY = ['PUB', 'EXT', 'SUB', 'PT', 'MUT', 'UMB', 'PVF', 'HOL', 'MUC', 'TRU',
                 'OPD', 'PEF', 'FND', 'FNS', 'JVT', 'VEN', 'HED', 'UIT', 'MUE', 'ABS',
                 'GOV', 'ESP', 'PRO', 'FAF', 'SOV', 'COR', 'IDX', 'BAS', 'PRT', 'SHP']
LABEL_COLLEGE = ['COL']
LABEL_REMAPPED = ['ORG', 'MISC']
HEADER_NER = ['TOKEN', 'POS', 'NER']

RE_WORDS = re.compile(r"[\w\d\.-]+")
QUOTATION = {"'": '"'}


##############################################################################


# Data preparation

def process_annotated(in_file, col_names=HEADER_NER, delimiter=('##END', '###', 'O')):
    """
    | following python-crfsuit, sklearn_crfsuit doesn't support pandas DF, so a feature
    | dic is used instead
    | http://python-crfsuite.readthedocs.io/en/latest/pycrfsuite.html#pycrfsuite.ItemSequence
    :param in_file: CSV file: TOKEN, POS, NER
    :param col_names
    :param delimiter
    :return: [[sent]]
    """
    data = pd.read_csv(in_file, header=None, engine='c', quoting=0)
    data.columns = col_names
    data = data.dropna()
    zipped_list = zip(data[i].tolist() for i in col_names)
    sents = (tuple(i) for i in zipped_list)
    sents = (list(x[1])[:-1] for x in groupby(sents, lambda x: x == delimiter)
             if not x[0])
    sents = [i for i in sents if i != []]
    return sents


def process_annotated_(in_file, col_names=HEADER_NER):
    """
    :param in_file: CSV file: TOKEN, POS, NER
    :param col_names
    :return: [[sent]]
    """
    data = pd.read_csv(in_file, header=None, engine='c', quoting=0)
    data.columns = col_names
    data = data.dropna()
    return data


def prepare_features_(dfs):
    # Move to arsenal
    """
    :param dfs: a list of pd dfs
    :return: a list of feature sets and feature dicts
    """
    f_sets = [df2set(df) for df in dfs if len(df.columns) == 1]
    f_dics = [df2dic(df) for df in dfs if len(df.columns) == 2]
    return f_sets, f_dics


def batch_loading_(crf_f, feature_hdf, hdf_keys, crf_model=False):
    # Move to arsenal
    """
    :param dict_conf:
    :param crf_f:
    :param feature_hdf:
    :param hdf_keys:
    :param crf_model:
    :return:
    """
    crf = jl.load(crf_f) if crf_model else None
    loads = hdf2df(feature_hdf, hdf_keys)
    f_sets, f_dics = prepare_features_(loads)
    return crf, f_sets, f_dics


def batch_add_features_(df, f_sets, f_dics):
    f_sets = [{i: i for i in j} for j in f_sets]  # A special case of comprehension
    all = f_sets + f_dics
    all_names = [str(i) for i in range(len(all))]
    df_list = [map_dic2df(df, c_name, f_dic) for c_name, f_dic in zip(all_names, all)]
    return df_list[-1]


def df2crfsuite(df, delim='##END'):
    delimiter = tuple(df[df.iloc[:, 0] == delim].iloc[0, :].tolist())
    sents = zip(*[df[i].tolist() for i in df.columns])  # Use * to unpack a list
    sents = (list(x[1]) for x in groupby(sents, lambda x: x == delimiter))
    result = [i for i in sents if i != [] and i != [(delimiter)]]
    return result

##############################################################################


def map_set_2_matrix(sent, feature_set):
    """
    :param sent: [(word, pos, ner)]
    :param feature_set: {feature1, feature2}
    :return: [(word, pos, ner, other_features)]
    """
    feature_list = ['1' if line[0] in feature_set else '0' for line in sent]
    return [(sent[i] + (feature_list[i],)) for i in range(len(list(sent)))]


def map_dict_2_matrix(sent, feature_dic):
    """
    :param sent: [(word, pos, ner)]
    :param feature_set: {feature1:value1, feature2:value2}
    :return: [(word, pos, ner, other_features)]
    """
    feature_list = [
        str(feature_dic.get(line[0].lower())) if line[0].lower() in feature_dic.keys()
        else '0' for line in sent]
    return [(sent[i] + (feature_list[i],)) for i in range(len(list(sent)))]


def map_set2df(df, col_name, feature_set):
    feature_dict = {str(i): str(i) for i in feature_set}  # Construct a dic from a set
    df[col_name] = df.iloc[:, 0].map(feature_dict)
    return df.replace(np.nan, '0')


def map_dic2df(df, col_name, feature_dict):
    df[col_name] = df.iloc[:, 0].map(feature_dict)
    return df.replace(np.nan, 0)


##############################################################################


# Feature extraction


def feature_selector(word, feature_conf, conf_switch, postag, aca, com_single, com_suffix,
                     location, name, ticker, tfdf, tfidf):
    """
    Set the feature dict here
    :param word: word itself
    :param feature_conf: feature config
    :param conf_switch: select the right config from feature_config
    :param postag:
    :param name:
    :param com_suffix:
    :param country:
    :param city:
    :param com_single:
    :param tfidf:
    :param tfdf:
    :return:
    """
    feature_dict = {
        'bias': 1.0,
        conf_switch + '_word.lower()': word.lower(),
        conf_switch + '_word[-3]': word[-3:],
        conf_switch + '_word[-2]': word[-2:],
        conf_switch + '_word.isupper()': word.isupper(),
        conf_switch + '_word.istitle()': word.istitle(),
        conf_switch + '_word.isdigit()': word.isdigit(),
        conf_switch + '_word.islower()': word.islower(),
        conf_switch + '_postag': postag,
        conf_switch + '_aca': aca,
        conf_switch + '_com_single': com_single,
        conf_switch + '_com_suffix': com_suffix,
        conf_switch + '_location': location,
        conf_switch + '_name': name,
        conf_switch + '_ticker': ticker,
        conf_switch + '_tfidf': tfidf,
        conf_switch + '_tfdf': tfdf,
    }
    return {i: feature_dict.get(i) for i in feature_conf[conf_switch] if
            i in feature_dict.keys()}


def word2features(sent, i, feature_conf):
    word, postag, _, aca, com_single, com_suffix, location, name, ticker, tfdf, tfidf = \
        sent[i]
    features = feature_selector(word, feature_conf, 'current', postag, aca, com_single,
                                com_suffix, location, name, ticker, tfdf, tfidf)
    if i > 0:
        word1, postag1, _, aca1, com_single1, com_suffix1, location1, name1, ticker1, tfidf1, tfdf1 = \
            sent[i - 1]
        features.update(
            feature_selector(word1, feature_conf, 'previous', postag1, aca1, com_single1,
                             com_suffix1, location1, name1, ticker1, tfidf1, tfdf1))
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        word1, postag1, _, aca1, com_single1, com_suffix1, location1, name1, ticker1, tfidf1, tfdf1 = \
            sent[i + 1]
        features.update(
            feature_selector(word1, feature_conf, 'next', postag1, aca1, com_single1,
                             com_suffix1, location1, name1, ticker1, tfidf1, tfdf1))
    else:
        features['EOS'] = True

    return features


def feature_selector_(word_tuple, feature_conf, conf_switch):
    """
    Set the feature dict here
    :param word: word itself
    :param feature_conf: feature config
    :param conf_switch: select the right config from feature_config
    :return:
    """

    word, pos, other_features = word_tuple[0], word_tuple[1], word_tuple[3:]
    other_length = len(other_features)
    other_dict = {'_'.join((conf_switch, str(j))): k for j, k in zip(range(other_length), other_features)}
    feature_func = {'_'.join((conf_switch, name)): func for (name, func) in feature_conf.items()}
    feature_dict = {name: func(word) for (name, func) in feature_func.items()}
    feature_dict.update(other_dict)
    return feature_dict


def word2features_(sent, i, feature_conf):
    features = feature_selector_(sent[i], feature_conf, 'current')
    if i > 0:
        features.update(
            feature_selector_(sent[i - 1], feature_conf, 'previous'))
    else:
        features['BOS'] = True
    if i < len(sent) - 1:
        features.update(
            feature_selector_(sent[i + 1], feature_conf, 'next'))
    else:
        features['EOS'] = True
    return features


def sent2features(line, feature_conf):
    return [word2features_(line, i, feature_conf) for i in range(len(line))]


def sent2labels(line):
    return [i[2] for i in line]  # Use the correct column


def sent2label_spfc(line, label):
    return [i[2] if i[2].endswith(label) else '0' for i in line]


##############################################################################


# CRF training

def feed_crf_trainer(in_data, conf):
    """
    :param in_data:
    :param conf_f:
    :return: nested lists of lists
    """
    basic_logging('begins')    
    features = [sent2features(s, conf) for s in in_data]
    basic_logging('feaures')
    labels = [sent2labels(s) for s in in_data]
    basic_logging('label')
    return features, labels


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


def show_crf_label(crf, remove_list=['O', 'NER', '']):
    labels = list(crf.classes_)
    return [i for i in labels if i not in remove_list]


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


# CRF testing and predicting


def convert_tags(data):
    converted = []
    for sent in data:
        test_result = []
        for tag in sent:
            if tag == 'O':
                test_result.append('0')
            else:
                test_result.append('1')
        converted.append(test_result)
    return converted


def export_test_result(labels, y_test, y_pred):
    details = metrics.flat_classification_report(y_test, y_pred, digits=3, labels=labels)
    details = [i for i in [re.findall(RE_WORDS, i) for i in details.split('\n')] if i !=
               []][
              1:-1]
    details = pd.DataFrame(details, columns=HEADER_CRF)
    details = details.sort_values('f1', ascending=False)
    return details


def test_crf_prediction(crf, X_test, y_test, test_switch='spc'):
    """

    :param crf:
    :param X_test:
    :param y_test:
    :param test_switch: 'spc' for specific labels, 'bin' for binary labels
    :return:
    """
    y_pred = crf.predict(X_test)

    if test_switch == 'spc':
        labels = show_crf_label(crf)

        result = metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)
        details = export_test_result(labels, y_test, y_pred)
        return result, details

    elif test_switch == 'bin':

        y_pred_converted = convert_tags(y_pred)
        y_test_converted = convert_tags(y_test)
        labels = ['1']

        result = metrics.flat_f1_score(y_test_converted, y_pred_converted,
                                       average='weighted',
                                       labels=labels)
        y_test_flatten = ['0' if j == 'O' else '1' for i in y_test for j in i]
        details = export_test_result(labels, y_test_flatten, y_pred_converted)
        return result, details


def crf_predict(crf, new_data, processed_data):
    result = crf.predict(processed_data)
    length = len(list(new_data))
    crf_result = (
        [(new_data[j][i][:2] + (result[j][i],)) for i in range(len(new_data[j]))] for j in
        range(length))
    crf_result = [i + [('##END', '###', 'O')] for i in crf_result]
    return list(chain.from_iterable(crf_result))


##############################################################################


def crf_result2dict(crf_result):
    ner_candidate = [(token, ner) for token, _, ner in crf_result if ner[0] != 'O']
    ner_index = (i for i in range(len(ner_candidate)) if
                 ner_candidate[i][1][0] == 'U' or ner_candidate[i][1][0] == 'L')
    new_index = (a + b for a, b in enumerate(ner_index))
    ner_result = extract_ner_result(ner_candidate, new_index)
    return ner_result


def sort_dic(dic, sort_key=0, rev=False):
    """
    :param dic:
    :param sort_key: 0: sort by key, 1: sort by value
    :param rev: false by default
    :return: sorted {(k, v)}
    """
    return OrderedDict(sorted(iter(dic.items()), key=itemgetter(sort_key), reverse=rev))


def extract_ner_result(ner_candidate, new_index):
    new_candidate = deepcopy(ner_candidate)
    for i in new_index:
        new_candidate[i + 1:i + 1] = [('##split', '##split')]
    ner_result = (
        ' '.join(
            [(i[0].strip() + '##' + i[1].strip()) for i in new_candidate if i[1]]).split(
            '##split'))
    ner_result = ([i.strip(' ') for i in ner_result if i and i != '##'])
    ner_result = ('##'.join((' '.join([i.split('##')[0] for i in tt.split()]), tt[-3:]))
                  for tt in
                  ner_result)
    ner_result = sort_dic(Counter(i for i in ner_result if i), sort_key=1, rev=True)
    return ner_result
