# -*- coding: utf-8 -*-

from .arsenal_stats import *
from itertools import groupby
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.metrics import make_scorer
from re import findall, compile
from sklearn.grid_search import RandomizedSearchCV
import scipy.stats as ss
import logging

logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

HEADER_ANNOTATION = ['TOKEN', 'POS', 'NER']
HEADER_CRF = ['tag', 'precision', 'recall', 'f1', 'support']

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
    | following python-crfsuit, sklearn_crfsuit doesn't support pandas DF, so a feature dic is used instead
    | http://python-crfsuite.readthedocs.io/en/latest/pycrfsuite.html#pycrfsuite.ItemSequence
    :param in_file: CSV file: TOKEN, POS, NER
    :return: [[sent]]
    """
    data = pd.read_csv(in_file, header=None, engine='c', quoting=0)
    data.columns = HEADER_ANNOTATION
    data = data.dropna()
    sents = (tuple(i) for i in zip(data['TOKEN'].tolist(), data['POS'].tolist(), data['NER'].tolist()))
    sents = (list(x[1])[:-1] for x in groupby(sents, lambda x: x == ('##END', '###', 'O')) if not x[0])
    sents = [i for i in sents if i != []]
    return sents


def prepare_features_dict(in_file):
    """
    | Reading a line-based csv file, and converting it to a feature dic
    :param in_file:  token,value
    :return: {token: value}
    """
    return {i.split(',')[0]: i.split(',')[1].strip('\r\n') for i in open(in_file, 'r')}


def add_one_features_list(sent, feature_set):
    """
    :param sent: [(word, pos, ner)]
    :param feature_set: {feature1, feature2}
    :return: [(word, pos, ner, other_features)]
    """
    feature_list = ['1' if line[0] in feature_set else '0' for line in sent]
    return [(sent[i] + (feature_list[i],)) for i in range(len(sent))]


def add_one_feature_dict(sent, feature_dic):
    """
    :param sent: [(word, pos, ner)]
    :param feature_set: {feature1:value1, feature2:value2}
    :return: [(word, pos, ner, other_features)]
    """
    feature_list = [str(feature_dic.get(line[0])) if line[0] in feature_dic.keys() else '0' for line in sent]
    return [(sent[i] + (feature_list[i],)) for i in range(len(sent))]


def batch_add_features(text_file, name_f, com_suffix_f, country_f, city_f, com_single_f, com_multi_f, tfidf_f):
    pos_data = process_annotated(text_file)

    set_name, set_country = line_file2set(name_f), line_file2set(country_f)
    set_city, set_com_single = line_file2set(city_f), line_file2set(com_single_f)
    set_com_suffix = {i.title() for i in line_file2set(com_suffix_f)}
    set_com_multi = line_file2set(com_multi_f)
    dict_tfidf = prepare_features_dict(tfidf_f)

    name_added = [add_one_features_list(chunk, set_name) for chunk in pos_data]
    com_suffix_added = [add_one_features_list(chunk, set_com_suffix) for chunk in name_added]
    country_added = [add_one_features_list(chunk, set_country) for chunk in com_suffix_added]
    city_added = [add_one_features_list(chunk, set_city) for chunk in country_added]
    com_single_added = [add_one_features_list(chunk, set_com_single) for chunk in city_added]
    result = [add_one_feature_dict(chunk, dict_tfidf) for chunk in com_single_added]

    # result = [add_multi_features(chunk, set_com_multi) for chunk in com_single_added]
    # print(get_now(), 'multi_com')

    return result


##############################################################################


# Feature extraction


def set_features(word, postag, name, com_suffix, country, city, com_single, tfidf):
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
        'tf_idf': tfidf
    }
    return features


def update_features(features, word1, postag1, name1, com_suffix1, country1, city1, com_single1, tfidf1):
    features.update({
        '-1:word.lower()': word1.lower(),
        '-1:word.istitle()': word1.istitle(),
        '-1:word.isupper()': word1.isupper(),
        '-1:postag': postag1,
        '-1:name': name1,
        '-1:com_suffix': com_suffix1,
        '-1:com_single': com_single1,
        '-1:city': city1,
        '-1:country': country1,
        '-1:tfidf': tfidf1
    })


def word2features(sent, i):
    word, postag, _, name, company, city, country, com_single, tfidf = sent[i]
    features = set_features(word, postag, name, company, city, country, com_single, tfidf)

    if i > 0:
        word1, postag1, ner1, name1, company1, city1, country1, com_single1, tfidf1 = sent[i - 1]
        update_features(features, word1, postag1, name1, company1, city1, country1, com_single1, tfidf1)
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        word1, postag1, ner1, name1, company1, city1, country1, com_single1, tfidf1 = sent[i + 1]
        update_features(features, word1, postag1, name1, company1, city1, country1, com_single1, tfidf1)
    else:
        features['EOS'] = True
    return features


def sent2features(line):
    return [word2features(line, i) for i in range(len(line))]


def sent2labels(line):
    return [i[2] for i in line]  # Use the right column


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
    print(labels)
    if 'O' in labels:
        labels.remove('O')
    if 'NER' in labels:
        labels.remove('NER')
    if '' in labels:
        labels.remove('')
    return labels


def make_param_space():
    return {
        'c1': ss.expon(scale=0.5),
        'c2': ss.expon(scale=0.05),
    }


def make_f1_scorer(labels):
    return make_scorer(metrics.flat_f1_score, average='weighted', labels=labels)


def predict_crf(crf, X_test, y_test):
    labels = show_crf_label(crf)
    y_pred = crf.predict(X_test)
    result = metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)
    details = metrics.flat_classification_report(y_test, y_pred, digits=3, labels=labels)
    details = [i for i in [findall(RE_WORDS, i) for i in details.split('\n')] if i != []][1:-1]
    details = pd.DataFrame(details, columns=HEADER_CRF)
    return result, details


def search_param(X_train, y_train, crf, params_space, f1_scorer, cv=3, iteration=50):
    rs = RandomizedSearchCV(crf, params_space,
                            cv=cv,
                            verbose=1,
                            n_jobs=-1,
                            n_iter=iteration,
                            scoring=f1_scorer)
    return rs.fit(X_train, y_train)


##############################################################################


def pipeline_crf_train(train_f, test_f, name_f, com_suffix_f, country_f, city_f, com_single_f, com_multi_f, tfidf_f):
    train_sents = batch_add_features(train_f, name_f, com_suffix_f, country_f, city_f, com_single_f, com_multi_f,
                                     tfidf_f)
    test_sents = batch_add_features(test_f, name_f, com_suffix_f, country_f, city_f, com_single_f, com_multi_f, tfidf_f)
    print(get_now(), 'converted')
    X_train, y_train, X_test, y_test = feed_crf_trainer(train_sents, test_sents)
    print(get_now(), 'feed')
    crf = train_crf(X_train, y_train)
    print(get_now(), 'train')
    result, details = predict_crf(crf, X_test, y_test)
    print(get_now(), 'predict')
    return crf, result, details


def pipeline_crf_cv(train_f, test_f, name_f, com_suffix_f, country_f, city_f, com_single_f, com_multi_f, tfidf_f, cv, iteration):
    train_sents = batch_add_features(train_f, name_f, com_suffix_f, country_f, city_f, com_single_f, com_multi_f, tfidf_f)
    test_sents = batch_add_features(test_f, name_f, com_suffix_f, country_f, city_f, com_single_f, com_multi_f, tfidf_f)
    X_train, y_train, _, _ = feed_crf_trainer(train_sents, test_sents)
    crf = train_crf(X_train, y_train)
    labels = show_crf_label(crf)
    params_space = make_param_space()
    f1_scorer = make_f1_scorer(labels)
    rs_cv = search_param(X_train, y_train, crf, params_space, f1_scorer, cv, iteration)
    print('best params:', rs_cv.best_params_)
    print('best CV score:', rs_cv.best_score_)
    print('model size: {:0.2f}M'.format(rs_cv.best_estimator_.size_ / 1000000))
    return crf, rs_cv


def pipeline_best_predict(train_f, test_f, name_f, com_suffix_f, country_f, city_f, com_single_f, com_multi_f, tfidf_f, cv, iteration):
    train_sents = batch_add_features(train_f, name_f, com_suffix_f, country_f, city_f, com_single_f, com_multi_f, tfidf_f)
    test_sents = batch_add_features(test_f, name_f, com_suffix_f, country_f, city_f, com_single_f, com_multi_f, tfidf_f)
    X_train, y_train, X_test, y_test = feed_crf_trainer(train_sents, test_sents)
    crf = train_crf(X_train, y_train)
    labels = show_crf_label(crf)
    params_space = make_param_space()
    f1_scorer = make_f1_scorer(labels)
    rs_cv = search_param(X_train, y_train, crf, params_space, f1_scorer, cv, iteration)
    print(get_now(), 'predict')
    best_predictor = rs_cv.best_estimator_
    best_result, best_details = predict_crf(best_predictor, X_test, y_test)
    return crf, best_predictor, rs_cv, best_result, best_details


##############################################################################


def add_multi_word_features(sent, feature_set):
    tks = [i[0] for i in sent]
    feature_list = ['0' for i in range(len(tks))]
    for i in range(len(tks) - 1):
        if (tks[i].istitle() or tks[i].isupper()) and (tks[i + 1].istitle() or tks[i + 1].isupper()):
            print(tks[i:i + 2])
            for names in feature_set:
                n_split = names.split(' ')
                if len(n_split) == 2:
                    if tks[i] == n_split[0] and tks[i + 1] == n_split[1]:
                        feature_list[i:i + 2] = ['1', '1']
                        break
                    else:
                        tks = tks
                else:
                    tks = tks

        if i < (len(tks) - 2):
            if (tks[i].istitle() or tks[i].isupper()) and (tks[i + 1].istitle() or tks[i + 1].isupper()) and (
                        tks[i + 2].istitle() or tks[i + 2].isupper()):
                print(tks[i:i + 3])
                for names in feature_set:
                    n_split = names.split(' ')
                    if len(n_split) == 3:
                        if tks[i] == n_split[0] and tks[i + 1] == n_split[1] and tks[i + 2] == n_split[2]:
                            feature_list[i:i + 3] = ['1', '1', '1']
                            break
                    else:
                        tks = tks
            else:
                tks = tks
        else:
            tks = tks
    new_sent = [', '.join(i) for i in sent]
    return [tuple(', '.join(i).split(', ')) for i in zip(new_sent, feature_list)]


def add_multi_features(sent, feature_set):
    token_list = [i[0] for i in sent]
    token_dic = {v: k for (k, v) in enumerate(token_list)}
    tokens = ' '.join(token_list)
    feature_list = ['0' for i in range(len(sent))]
    if len([i[0] for i in sent if i[0].isupper() or i[0].istitle()]) >= 2:
        for feature in feature_set:
            if feature in tokens:
                feature_words = feature.split(' ')
                feature_start = token_dic.get(feature_words[0])
                feature_end = feature_start + len(feature_words)
                feature_list[feature_start: feature_end] = ['1' for i in range(len(feature_words))]
                break
    new_sent = [', '.join(i) for i in sent]
    return [tuple(', '.join(i).split(', ')) for i in zip(new_sent, feature_list)]
