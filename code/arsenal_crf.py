# -*- coding: utf-8 -*-

from itertools import groupby, chain
from re import findall, compile

import pandas as pd
import scipy.stats as sstats
import sklearn_crfsuite
from sklearn.metrics import make_scorer
# from sklearn.grid_search import RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn_crfsuite import metrics

HEADER_CRF = ['tag', 'precision', 'recall', 'f1', 'support']
LABEL_COMPANY = ['PUB', 'EXT', 'SUB', 'PT', 'MUT', 'UMB', 'PVF', 'HOL', 'MUC', 'TRU', 'OPD', 'PEF', 'FND', 'FNS',
                 'JVT', 'VEN', 'HED', 'UIT', 'MUE', 'ABS', 'GOV', 'ESP', 'PRO', 'FAF', 'SOV', 'COR', 'IDX', 'BAS',
                 'PRT', 'SHP']
LABEL_COLLEGE = ['COL']
LABEL_REMAPPED = ['ORG', 'MISC']

RE_WORDS = compile(r"[\w\d\.-]+")


##############################################################################


# Data preparation


def map_list_2_matrix(sent, feature_set):
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
    feature_list = [str(feature_dic.get(line[0].lower())) if line[0].lower() in feature_dic.keys() else '0' for line in sent]
    return [(sent[i] + (feature_list[i],)) for i in range(len(list(sent)))]


##############################################################################


# Feature extraction


def feature_selector(word, feature_conf, conf_switch, postag, aca, com_single, com_suffix, location, name, ticker, tfdf, tfidf):
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
    return {i: feature_dict.get(i) for i in feature_conf[conf_switch] if i in feature_dict.keys()}


def word2features(sent, i, feature_conf):

    word, postag, _, aca, com_single, com_suffix, location, name, ticker, tfdf, tfidf = sent[i]
    features = feature_selector(word, feature_conf, 'current', postag, aca, com_single, com_suffix, location, name, ticker, tfdf, tfidf)
    if i > 0:
        word1, postag1, _, aca1, com_single1, com_suffix1, location1, name1, ticker1, tfidf1, tfdf1 = sent[i - 1]
        features.update(
            feature_selector(word1, feature_conf, 'previous', postag1, aca1, com_single1, com_suffix1, location1, name1, ticker1, tfidf1, tfdf1))
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        word1, postag1, _, aca1, com_single1, com_suffix1, location1, name1, ticker1, tfidf1, tfdf1 = sent[i + 1]
        features.update(
            feature_selector(word1, feature_conf, 'next', postag1, aca1, com_single1, com_suffix1, location1, name1, ticker1, tfidf1, tfdf1))
    else:
        features['EOS'] = True

    return features


def sent2features(line, feature_conf):
    return [word2features(line, i, feature_conf) for i in range(len(line))]


def sent2labels(line):
    return [i[2] for i in line]  # Use the correct column


def sent2label_spfc(line, label):
    return [i[2] if i[2].endswith(label) else '0' for i in line]


##############################################################################


def feature_selector_(word, feature_conf, conf_switch, postag, **features):
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
    return {i: feature_dict.get(i) for i in feature_conf[conf_switch] if i in feature_dict.keys()}


def word2features_(sent, i, feature_conf, *features):

    word, postag, _, features = sent[i]
    features = feature_selector(word, feature_conf, 'current', postag)
    if i > 0:
        word1, postag1, _, features  = sent[i - 1]
        features.update(feature_selector(word1, feature_conf, 'previous', postag1))
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        word1, postag1, _, features  = sent[i + 1]
        features.update(feature_selector(word1, feature_conf, 'next', postag1))
    else:
        features['EOS'] = True

    return features

def sent2features_(line, feature_conf, *features):
    return [word2features_(line, i, feature_conf, *features) for i in range(len(line))]


def feed_crf_trainer_(in_data, conf, *features):
    """
    :param in_data:
    :param conf_f:
    :return:
    """
    features = [sent2features_(s, conf, *features) for s in in_data]
    labels = [sent2labels(s) for s in in_data]
    return features, labels


##############################################################################


# CRF training


def feed_crf_trainer(in_data, conf):
    """
    :param in_data:
    :param conf_f:
    :return: nested lists of lists
    """
    features = [sent2features(s, conf) for s in in_data]
    labels = [sent2labels(s) for s in in_data]
    return features, labels


def train_crf(X_train, y_train, algm='lbfgs', c1=0.1, c2=0.1, max_iter=100, all_trans=True):
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


def show_crf_label(crf, remove_list = ['O','NER', '']):
    labels = list(crf.classes_)
    print(labels)
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


def concert_tags(data):
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
        details = metrics.flat_classification_report(y_test, y_pred, digits=3, labels=labels)
        details = [i for i in [findall(RE_WORDS, i) for i in details.split('\n')] if i != []][1:-1]
        details = pd.DataFrame(details, columns=HEADER_CRF)
        details = details.sort_values('f1', ascending=False)

        return result, details

    elif test_switch == 'bin':

        y_pred_converted = concert_tags(y_pred)
        y_test_converted = concert_tags(y_test)

        result = metrics.flat_f1_score(y_test_converted, y_pred_converted, average='weighted', labels=['1'])

        y_test_converted = ['0' if j == 'O' else '1' for i in y_test for j in i]
        details = metrics.flat_classification_report(y_test_converted, y_pred_converted, digits=3, labels=['1'])

        details = [i for i in [findall(RE_WORDS, i) for i in details.split('\n')] if i != []][1:-1]
        details = pd.DataFrame(details, columns=HEADER_CRF)
        details = details.sort_values('f1', ascending=False)

        return result, details




def crf_predict(crf, new_data, processed_data):
    result = crf.predict(processed_data)
    length = len(list(new_data))
    crf_result = ([(new_data[j][i][:2] + (result[j][i],)) for i in range(len(new_data[j]))] for j in
                  range(length))
    crf_result = [i + [('##END', '###', 'O')] for i in crf_result]
    return list(chain.from_iterable(crf_result))
