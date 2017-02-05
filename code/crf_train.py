# -*- coding: utf-8 -*-

from .arsenal_stats import *
import numpy as np
from itertools import groupby
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from re import findall

HEADER_FS = ['fact', 'entity_proper_name', 'entity_type']
HEADER_SN = ['factset_entity_id', 'short_name']
HEADER_SN_TYPE = ['entity_type', 'short_name']
HEADER_SCHWEB = ['Language', 'Title', 'Type']
HEADER_ANNOTATION = ['TOKEN', 'POS', 'NER']

LABEL_COMPANY = ['PUB', 'EXT', 'SUB', 'PVT', 'MUT', 'UMB', 'PVF', 'HOL', 'MUC', 'TRU', 'OPD', 'PEF', 'FND', 'FNS',
                 'JVT', 'VEN', 'HED', 'UIT', 'MUE', 'ABS', 'GOV', 'ESP', 'PRO', 'FAF', 'SOV', 'COR',
                 'IDX', 'BAS', 'PRT', 'SHP']
LABEL_COLLEGE = ['COL']
LABEL_REMAPPED = ['ORG', 'MISC']
LABEL_ANS = ['category', 'nname_en']


##############################################################################

# Data preparation


def prepare_ans_dataset(in_file, out_file, col_list=LABEL_ANS):
    """
    It read ANS dataset
    :param in_file: an ANS json file
    :param col_list:
    :return: a df for gold parser to train
    """
    data = json2pd(in_file, col_list)
    data = rename_series(data, 'category', 'entity_types')
    data = rename_series(data, 'nname_en', 'entity_names')
    data['entity_names'] = data['entity_names'].str.title()
    data.to_csv(out_file, index=False)


def prepare_schweb_dataset(in_file, out_file):
    """
    :param in_file: schweb raw csv
    :param out_file: schweb csv
    """
    data = csv2pd(in_file, HEADER_SCHWEB, HEADER_SCHWEB, sep='\t')
    en_data = data[data.Language == 'en']
    result = en_data[en_data.Type.str.contains('Location|Personal|Organisation')]
    result['entity_type'] = np.where(result.Type.str.contains('Personal'), 'PERSON',
                                     np.where(result.Type.str.contains('Location'), 'GPE',
                                              np.where(result.Type.str.contains('Organisation'), 'ORG', 'MISC')))
    result = rename_series(result, 'Title', 'entity_name')
    result = result.drop(['Language', 'Type'], axis=1)
    result.to_csv(out_file, index=False)


def process_annotated(in_file):
    """
    following python-ccrfsuit, sklearn_crfsuit doesn't support pandas DF, so feature dic is used instead
    http://python-crfsuite.readthedocs.io/en/latest/pycrfsuite.html#pycrfsuite.ItemSequence
    :param in_file: CSV file: TOKEN, POS, NER
    :return: [[sent]]
    """
    with open(in_file) as data:
        sents = [tuple(i.split(',')) for i in data.read().split('\n')]
        # convert file to tuples
        sents = [list(x[1])[:-1] for x in groupby(sents, lambda x: x == ('##END', '###', 'O')) if not x[0]]
        # split each sentences, use [:1] to remove the empty end
        sents = [i for i in sents if i != []]
        # Remove empty sent
        return sents


##############################################################################


# Feature extraction


def update_features(features, postag1, word1):
    features.update({
        '-1:word.lower()': word1.lower(),
        '-1:word.istitle()': word1.istitle(),
        '-1:word.isupper()': word1.isupper(),
        '-1:postag': postag1,
        '-1:postag[:2]': postag1[:2],
    })


def set_features(postag, word):
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word.islower()': word.islower(), # iPhone
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    return features


def word2features(sent, i):
    word, postag = sent[i][0], sent[i][1]
    features = set_features(postag, word)

    if i > 0:
        word1, postag1 = sent[i - 1][0], sent[i - 1][1]
        update_features(features, postag1, word1)
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        word1, postag1 = sent[i + 1][0], sent[i + 1][1]
        update_features(features, postag1, word1)
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [i[-1] for i in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]


##############################################################################


# CRF training


def feed_crf_trainer(train_sents, test_sents):
    X_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]

    X_test = [sent2features(s) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]

    return X_train, y_train, X_test, y_test


def train_crf(X_train, y_train):
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    return crf.fit(X_train, y_train)


def show_crf_label(crf):
    labels = list(crf.classes_)
    labels.remove('O')
    labels.remove('')
    return labels


def predict_crf(crf, X_test, y_test):
    col = ['precision', 'recall', 'f1', 'score', 'support']
    labels = show_crf_label(crf)
    y_pred = crf.predict(X_test)
    result = metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)
    details = metrics.flat_classification_report(y_test, y_pred, digits=3)
    details = [i for i in [findall(r"[\w\d\.-]+", i) for i in details.split('\n')] if i !=[]][1:-1]
    details = pd.DataFrame(details, columns = col)
    return result, details


# 2017-02-04 F1: 0.44002688708718896

##############################################################################


def output_factset_sn_type(type_file, sn_file, out_file):
    sn = quickest_read_csv(sn_file, HEADER_SN)
    ty = quickest_read_csv(type_file, HEADER_FS)
    result = pd.merge(ty, sn, on='factset_entity_id', how='inner')
    result = result.dropna()
    result.tocsv(out_file, index=False)


def remap_factset_sn_type(in_file, out_file):
    data = quickest_read_csv(in_file, HEADER_SN_TYPE)
    result = remap_series(data, 'entity_type', 'new_entity_type', LABEL_COMPANY, 'ORG')
    result = result.drop(['entity_type'], axis=1)
    result.to_csv(out_file, index=False)
