# -*- coding: utf-8 -*-

from .arsenal_stats import *
from itertools import groupby
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.metrics import make_scorer
from re import findall, compile
from sklearn.grid_search import RandomizedSearchCV
import scipy.stats
import logging

logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

HEADER_ANNOTATION = ['TOKEN', 'POS', 'NER']

LABEL_COMPANY = ['PUB', 'EXT', 'SUB', 'PVT', 'MUT', 'UMB', 'PVF', 'HOL', 'MUC', 'TRU', 'OPD', 'PEF', 'FND', 'FNS',
                 'JVT', 'VEN', 'HED', 'UIT', 'MUE', 'ABS', 'GOV', 'ESP', 'PRO', 'FAF', 'SOV', 'COR',
                 'IDX', 'BAS', 'PRT', 'SHP']
LABEL_COLLEGE = ['COL']
LABEL_REMAPPED = ['ORG', 'MISC']

RE_WORDS = compile(r"[\w\d\.-]+")


##############################################################################


# Data preparation


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


def prepare_features_set(in_file):
    return set(i.strip('\n\r') for i in open(in_file, 'r'))


def add_one_word_features(sent, feature_set):
    """

    :param sent:
    :param feature_set:
    :return:
    """
    feature_list = ['1' if line[0] in feature_set else '0' for line in sent]
    new_sent = [', '.join(i) for i in sent]
    return [tuple(', '.join(i).split(', ')) for i in zip(new_sent, feature_list)]


def add_multi_word_features(sent, feature_set):
    tokens = [i[0] for i in sent]
    feature_list = []
    for i in range(len(tokens)):
        if tokens[i].istitle() or tokens[i].isupper():
            for names in feature_set:
                name_split = names.split(' ')
                for j in range(-1, len(name_split)-1):
                    print('tt', tokens[i])
                    print('cc', name_split[j+1])
                    if tokens[i] == name_split[j+1]:
                        feature_list.append('1')
                        break
                    else:
                        feature_list.append('0')
    return feature_list



def batch_add_features(pos_file, name_file, com_suffix_file, country_file, city_file, com_single_file, com_multi_file):
    pos_data = process_annotated(pos_file)
    name_set = prepare_features_set(name_file)
    com_suffix = prepare_features_set(com_suffix_file)
    country_set = prepare_features_set(country_file)
    city_set = prepare_features_set(city_file)
    com_single_set = prepare_features_set(com_single_file)
    com_multi_set = prepare_features_set(com_multi_file)

    name_added = [add_one_word_features(chunk, name_set) for chunk in pos_data]
    company_added = [add_one_word_features(chunk, com_suffix) for chunk in name_added]
    country_added = [add_one_word_features(chunk, country_set) for chunk in company_added]
    city_added = [add_one_word_features(chunk, city_set) for chunk in country_added]
    result = [add_one_word_features(chunk, com_single_set) for chunk in city_added]

    return result


##############################################################################


# Feature extraction

# def if_two_words(word):
#     if word.isupper() or word.istitle():




def update_features(features, word1, postag1, name1, com_suffix1, country1, city1, com_single1):
    features.update({
        '-1:word.lower()': word1.lower(),
        '-1:word.istitle()': word1.istitle(),
        '-1:word.isupper()': word1.isupper(),
        '-1:postag': postag1,
        '-1name': name1,
        '-1com_suffix': com_suffix1,
        '-1com_single': com_single1,
        '-1city': city1,
        '-1country': country1,
    })


def set_features(word, postag, name, com_suffix, country, city, com_single):
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word.islower()': word.islower(),  # iPhone
        'postag': postag,
        'name': name,
        'comp_suffix': com_suffix,
        'com_single': com_single,
        'city': city,
        'country': country,
    }
    return features


def word2features(sent, i):
    word, postag, name, company, city = sent[i][0], sent[i][1], sent[i][3], sent[i][4], sent[i][5]
    country, com_single =  sent[i][6], sent[i][7]
    features = set_features(word, postag, name, company, city, country, com_single)

    if i > 0:
        word1, postag1, name1 = sent[i - 1][0], sent[i - 1][1], sent[i - 1][3],
        company1, city1, country1 = sent[i - 1][4], sent[i - 1][5], sent[i - 1][6]
        com_single1 = sent[i - 1][7]
        update_features(features, word1, postag1, name1, company1, city1, country1, com_single1)
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        word1, postag1, name1 = sent[i - 1][0], sent[i - 1][1], sent[i - 1][3],
        company1, city1, country1 = sent[i - 1][4], sent[i - 1][5], sent[i - 1][6]
        com_single1 = sent[i - 1][7]
        update_features(features, word1, postag1, name1, company1, city1, country1, com_single1)
    else:
        features['EOS'] = True

    return features


def sent2features(line):
    return [word2features(line, i) for i in range(len(line))]


def sent2labels(line):
    return [i[-6] for i in line]  # Use the right column


def sent2tokens(line):
    return [token for token, postag, label in line]


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
    if 'NER' in labels:
        labels.remove('NER')
    if '' in labels:
        labels.remove('')
    return labels


def make_param_space():
    return {
        'c1': scipy.stats.expon(scale=0.5),
        'c2': scipy.stats.expon(scale=0.05),
    }


def make_f1_scorer(labels):
    return make_scorer(metrics.flat_f1_score,
                       average='weighted', labels=labels)


def predict_crf(crf, X_test, y_test):
    col = ['tag', 'precision', 'recall', 'f1', 'support']
    labels = show_crf_label(crf)
    print(labels)
    y_pred = crf.predict(X_test)
    result = metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)
    details = metrics.flat_classification_report(y_test, y_pred, digits=3, labels=labels)
    details = [i for i in [findall(RE_WORDS, i) for i in details.split('\n')] if i != []][1:-1]
    details = pd.DataFrame(details, columns=col)
    return result, details


def cv_crf(X_train, y_train, crf, params_space, f1_scorer, cv=3, iteration=50):
    rs = RandomizedSearchCV(crf, params_space,
                            cv=cv,
                            verbose=1,
                            n_jobs=-1,
                            n_iter=iteration,
                            scoring=f1_scorer)
    return rs.fit(X_train, y_train)


##############################################################################


def pipeline_crf_train(train_file, test_file, name_file, company_file, country_file, city_file, com_single_file, com_multi_file):
    train_sents = batch_add_features(train_file, name_file, company_file, country_file, city_file, com_single_file, com_multi_file)
    test_sents = batch_add_features(test_file, name_file, company_file, country_file, city_file, com_single_file, com_multi_file)
    X_train, y_train, X_test, y_test = feed_crf_trainer(train_sents, test_sents)
    crf = train_crf(X_train, y_train)
    result, details = predict_crf(crf, X_test, y_test)
    return crf, result, details


def pipeline_crf_cv(train_file, test_file, name_file, company_file, country_file, city_file, com_single_file, com_multi_file, cv, iteration):
    train_sents = batch_add_features(train_file, name_file, company_file, country_file, city_file, com_single_file, com_multi_file)
    test_sents = batch_add_features(test_file, name_file, company_file, country_file, city_file, com_single_file, com_multi_file)
    X_train, y_train, _, _ = feed_crf_trainer(train_sents, test_sents)
    crf = train_crf(X_train, y_train)
    labels = show_crf_label(crf)
    params_space = make_param_space()
    f1_scorer = make_f1_scorer(labels)
    result = cv_crf(X_train, y_train, crf, params_space, f1_scorer, cv, iteration)
    return crf, result
