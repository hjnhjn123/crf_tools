# -*- coding: utf-8 -*-

from collections import defaultdict
from itertools import chain
from itertools import groupby
from re import findall, compile

import scipy.stats as ss
import sklearn_crfsuite
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn_crfsuite import metrics
from yaml import load

from .arsenal_spacy import spacy_batch_processing
from .arsenal_stats import *

HEADER_ANNOTATION = ['TOKEN', 'POS', 'NER']
HEADER_CRF = ['tag', 'precision', 'recall', 'f1', 'support']

LABEL_COMPANY = ['PUB', 'EXT', 'SUB', 'PT', 'MUT', 'UMB', 'PVF', 'HOL', 'MUC', 'TRU', 'OPD', 'PEF', 'FND', 'FNS',
                 'JVT', 'VEN', 'HED', 'UIT', 'MUE', 'ABS', 'GOV', 'ESP', 'PRO', 'FAF', 'SOV', 'COR', 'IDX', 'BAS',
                 'PRT', 'SHP']
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
    # TODO no use of two splits
    with open(in_file, 'r') as data:
        result = defaultdict()
        for i in data:
            line = i.split(',')
            result[line[0]] = line[1].strip('\r\n')
        return result


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


##############################################################################


# Feature extraction


def feature_selector(word, feature_conf, conf_switch, postag, name, com_suffix, country, city, com_single, tfidf, tfdf):
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
        conf_switch + '_name': name,
        conf_switch + '_comp_suffix': com_suffix,
        conf_switch + '_com_single': com_single,
        conf_switch + '_city': city,
        conf_switch + '_country': country,
        conf_switch + '_tfidf': tfidf,
        conf_switch + '_tfdf': tfdf,
        conf_switch + '_1:word.lower()': word.lower(),
    }
    return {i: feature_dict.get(i) for i in feature_conf[conf_switch] if i in feature_dict.keys()}


def load_yaml_conf(conf_f):
    with open(conf_f, 'r') as f:
        result = load(f)
    return result


def word2features(sent, i, feature_conf):
    word, postag, _, name, comp_suffix, city, country, com_single, tfidf, tfdf = sent[i]
    features = feature_selector(word, feature_conf, 'current', postag, name, comp_suffix, country, city, com_single,
                                tfidf, tfdf)
    if i > 0:
        word1, postag1, _, name1, comp_suffix1, city1, country1, com_single1, tfidf1, tfdf1 = sent[i - 1]
        features.update(
            feature_selector(word1, feature_conf, 'previous', postag1, name1, comp_suffix1, country1, city1,
                             com_single1, tfidf1, tfdf1))
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        word1, postag1, _, name1, comp_suffix1, city1, country1, com_single1, tfidf1, tfdf1 = sent[i + 1]
        features.update(
            feature_selector(word1, feature_conf, 'next', postag1, name1, comp_suffix1, country1, city1,
                             com_single1, tfidf1, tfdf1))
    else:
        features['EOS'] = True
    return features


def sent2features(line, feature_conf):
    return [word2features(line, i, feature_conf) for i in range(len(line))]


def sent2labels(line):
    return [i[2] for i in line]  # Use the right column


def sent2tokens(line):
    return [token for token, postag, label in line]


def batch_add_features(pos_data, name_f, com_suffix_f, country_f, city_f, com_single_f, tfidf_f, tfdf_f):
    set_name, set_country = line_file2set(name_f), line_file2set(country_f)
    set_city, set_com_single = line_file2set(city_f), line_file2set(com_single_f)
    set_com_suffix = {i.title() for i in line_file2set(com_suffix_f)}
    dict_tfidf = prepare_features_dict(tfidf_f)
    dict_tfdf = prepare_features_dict(tfdf_f)

    name_added = (add_one_features_list(chunk, set_name) for chunk in pos_data)
    com_suffix_added = (add_one_features_list(chunk, set_com_suffix) for chunk in name_added)
    country_added = (add_one_features_list(chunk, set_country) for chunk in com_suffix_added)
    city_added = (add_one_features_list(chunk, set_city) for chunk in country_added)
    com_single_added = (add_one_features_list(chunk, set_com_single) for chunk in city_added)
    tfidf_added = (add_one_feature_dict(chunk, dict_tfidf) for chunk in com_single_added)
    result = [add_one_feature_dict(chunk, dict_tfdf) for chunk in tfidf_added]

    return result


##############################################################################


# CRF training


def feed_crf_trainer(train_sents, test_sents, conf_f):
    conf = load_yaml_conf(conf_f)
    X_train = [sent2features(s, conf) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]

    X_test = [sent2features(s, conf) for s in test_sents]
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


def test_crf_prediction(crf, X_test, y_test):
    labels = show_crf_label(crf)
    y_pred = crf.predict(X_test)
    result = metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)
    details = metrics.flat_classification_report(y_test, y_pred, digits=3, labels=labels)
    details = [i for i in [findall(RE_WORDS, i) for i in details.split('\n')] if i != []][1:-1]
    details = pd.DataFrame(details, columns=HEADER_CRF)
    return result, details


def crf_predict(crf, new_data, processed_data):
    result = crf.predict(processed_data)
    crf_result = ([(new_data[j][i][:2] + (result[j][i],)) for i in range(len(new_data[j]))] for j in
                  range(len(new_data)))
    crf_result = [i + [('##END', '###', 'O')] for i in crf_result]
    return [chain.from_iterable(crf_result)]


##############################################################################


def pipeline_crf_train(train_f, test_f, conf_f, name_f, com_suffix_f, country_f, city_f, com_single_f, tfidf_f, tfdf_f):
    train_data, test_data = process_annotated(train_f), process_annotated(test_f)
    train_sents = batch_add_features(train_data, name_f, com_suffix_f, country_f, city_f, com_single_f, tfidf_f, tfdf_f)
    test_sents = batch_add_features(test_data, name_f, com_suffix_f, country_f, city_f, com_single_f, tfidf_f, tfdf_f)
    print(get_now(), 'converted')

    X_train, y_train, X_test, y_test = feed_crf_trainer(train_sents, test_sents, conf_f)
    print(get_now(), 'feed')
    crf = train_crf(X_train, y_train)
    print(get_now(), 'train')
    result, details = test_crf_prediction(crf, X_test, y_test)
    print(get_now(), 'predict')
    return crf, result, details


def pipeline_crf_cv(train_f, test_f, conf_f, name_f, com_suffix_f, country_f, city_f, com_single_f, com_multi_f,
                    tfidf_f, tfdf_f, cv, iteration):
    train_data, test_data = process_annotated(train_f), process_annotated(test_f)
    train_sents = batch_add_features(train_data, name_f, com_suffix_f, country_f, city_f, com_single_f, tfidf_f, tfdf_f)
    test_sents = batch_add_features(test_data, name_f, com_suffix_f, country_f, city_f, com_single_f, tfidf_f, tfdf_f)
    print(get_now(), 'converted')

    X_train, y_train, X_test, y_test = feed_crf_trainer(train_sents, test_sents, conf_f)
    crf = train_crf(X_train, y_train)
    labels = show_crf_label(crf)
    params_space = make_param_space()
    f1_scorer = make_f1_scorer(labels)
    rs_cv = search_param(X_train, y_train, crf, params_space, f1_scorer, cv, iteration)
    print('best params:', rs_cv.best_params_)
    print('best CV score:', rs_cv.best_score_)
    print('model size: {:0.2f}M'.format(rs_cv.best_estimator_.size_ / 1000000))
    return crf, rs_cv


def pipeline_train_best_predict(train_f, test_f, conf_f, name_f, com_suffix_f, country_f, city_f, com_single_f, tfidf_f,
                                tfdf_f, cv, iteration):
    train_data, test_data = process_annotated(train_f), process_annotated(test_f)
    train_sents = batch_add_features(train_data, name_f, com_suffix_f, country_f, city_f, com_single_f, tfidf_f, tfdf_f)
    test_sents = batch_add_features(test_data, name_f, com_suffix_f, country_f, city_f, com_single_f, tfidf_f, tfdf_f)
    print(get_now(), 'converted')

    X_train, y_train, X_test, y_test = feed_crf_trainer(train_sents, test_sents, conf_f)
    crf = train_crf(X_train, y_train)
    labels = show_crf_label(crf)
    params_space = make_param_space()
    f1_scorer = make_f1_scorer(labels)
    rs_cv = search_param(X_train, y_train, crf, params_space, f1_scorer, cv, iteration)
    print(get_now(), 'predict')
    best_predictor = rs_cv.best_estimator_
    best_result, best_details = test_crf_prediction(best_predictor, X_test, y_test)
    return crf, best_predictor, rs_cv, best_result, best_details


def pipeline_crf_predict(train_f, test_f, conf_f, name_f, com_suffix_f, country_f, city_f, com_single_f, tfidf_f,
                         tfdf_f, out_f):
    train_data, test_data = process_annotated(train_f), process_annotated(test_f)
    train_sents = batch_add_features(train_data, name_f, com_suffix_f, country_f, city_f, com_single_f, tfidf_f, tfdf_f)
    test_sents = batch_add_features(test_data, name_f, com_suffix_f, country_f, city_f, com_single_f, tfidf_f, tfdf_f)

    print(get_now(), 'converted')
    X_train, y_train, X_test, y_test = feed_crf_trainer(train_sents, test_sents, conf_f)
    print(get_now(), 'feed')
    crf = train_crf(X_train, y_train)
    print(get_now(), 'train')
    result = crf_predict(crf, test_data, X_test)
    print(get_now(), 'predict')
    out = pd.DataFrame(result)
    out.to_csv(out_f, header=False, index=False)
    return crf, result


def pipeline_pos_crf(in_file, out_f, train_f, conf_f, name_f, com_suffix_f, country_f, city_f, com_single_f, tfidf_f,
                     tfdf_f, cols, pieces=10):
    data = json2pd(in_file, cols, lines=True)
    data = data.drop_duplicates()
    random_data = random_rows(data, pieces, 'content')
    random_data = random_data.dropna()
    parsed_data = spacy_batch_processing(random_data, ['chk'], '', 'content', ['content'])
    parsed_data = chain.from_iterable(parsed_data)
    pos_data = [list(x[1])[:-1] for x in groupby(parsed_data, lambda x: x == ('##END', '###', 'O')) if not x[0]]

    train_data = process_annotated(train_f)

    train_sents = batch_add_features(train_data, name_f, com_suffix_f, country_f, city_f, com_single_f, tfidf_f, tfdf_f)
    test_sents = batch_add_features(pos_data, name_f, com_suffix_f, country_f, city_f, com_single_f, tfidf_f, tfdf_f)
    X_train, y_train, X_test, y_test = feed_crf_trainer(train_sents, test_sents, conf_f)
    print(get_now(), 'feed')
    crf = train_crf(X_train, y_train)
    print(get_now(), 'train')
    result = crf_predict(crf, pos_data, X_test)
    print(get_now(), 'predict')
    out = pd.DataFrame(result)
    out.to_csv(out_f, header=False, index=False)
    return crf, result


##############################################################################


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
