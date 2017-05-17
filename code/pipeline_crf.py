# -*- coding: utf-8 -*-

from math import modf
from os import listdir, path
from sys import path

import redis

from arsenal_crf import *
from arsenal_logging import *
from arsenal_spacy import *
from arsenal_stats import *
import conf.pat360ner_crf_en_settings as settings

FEATURES = settings.FEATURE_FUNCTION
TRAIN_F = settings.TRAIN_F
TEST_F = settings.TEST_F
MODEL_F = settings.MODEL_F
HDF_F = settings.HDF_F
HDF_KEY = settings.HDF_KEY
REPORT_TYPE = settings.REPORT_TYPE

##############################################################################


# Pipelines


def pipeline_train(train_f, test_f, model_f, conf, hdf_f, hdf_key, report_type):
    basic_logging('loading conf begins')
    _, f_dics = batch_loading('', hdf_f, hdf_key)
    basic_logging('loading conf ends')
    train_df, test_df = process_annotated(train_f), process_annotated(test_f)
    basic_logging('loading data ends')
    train_df = batch_add_features(train_df, f_dics)
    test_df = batch_add_features(test_df, f_dics)
    basic_logging('adding features ends')
    train_sents = df2crfsuite(train_df)
    test_sents = df2crfsuite(test_df)
    basic_logging('converting to crfsuite ends')
    X_train, y_train = feed_crf_trainer(train_sents, conf, hdf_key)
    X_test, y_test = feed_crf_trainer(test_sents, conf, hdf_key)
    basic_logging('computing features ends')
    crf = train_crf(X_train, y_train)
    basic_logging('training ends')
    result, details = test_crf_prediction(crf, X_test, y_test, report_type)
    basic_logging('testing ends')
    # jl.dump(crf, model_f)
    return crf, result, details


def pipeline_best_predict(train_f, test_f, model_f, conf, hdf_f, hdf_key, report_type,
                          cv, iteration):
    basic_logging('loading conf begins')
    _, f_dics = batch_loading('', hdf_f, hdf_key)
    basic_logging('loading conf ends')
    train_df, test_df = process_annotated(train_f), process_annotated(test_f)
    basic_logging('loading data ends')
    train_df = batch_add_features(train_df, f_dics)
    test_df = batch_add_features(test_df, f_dics)
    train_sents = df2crfsuite(train_df)
    test_sents = df2crfsuite(test_df)
    X_train, y_train = feed_crf_trainer(train_sents, conf, hdf_key)
    X_test, y_test = feed_crf_trainer(test_sents, conf, hdf_key)
    basic_logging('Conversion ends')
    crf = train_crf(X_train, y_train)
    labels = show_crf_label(crf)
    params_space = make_param_space()
    f1_scorer = make_f1_scorer(labels)
    rs_cv = search_param(X_train, y_train, crf, params_space, f1_scorer, cv, iteration)
    basic_logging('Training ends')
    best_predictor = rs_cv.best_estimator_
    best_result, best_details = test_crf_prediction(best_predictor, X_test, y_test,
                                                    report_type)
    basic_logging('Testing ends')
    # jl.dump(best_predictor, model_f)
    return crf, best_predictor, rs_cv, best_result, best_details


def pipeline_test(test_f, model_f, crf_f, conf, hdf_f, hdf_key, report_type):
    basic_logging('loading conf begins')
    crf, f_dics = batch_loading(crf_f, hdf_f, hdf_key)
    basic_logging('loading conf ends')
    test_df = process_annotated(test_f)
    test_df = batch_add_features(test_df, f_dics)
    test_sents = df2crfsuite(test_df)
    X_test, y_test = feed_crf_trainer(test_sents, conf, hdf_key)
    basic_logging('Conversion ends')
    result, details = test_crf_prediction(crf, X_test, y_test, report_type)
    return result, details


##############################################################################


# Streaming


def streaming_pos_crf(in_f, hdf_f, hdf_key, conf):
    crf, f_dics = batch_loading('', hdf_f, hdf_key)
    raw_df = pd.read_json(in_f, lines=True)
    raw_df['content'] = raw_df.result.to_dict()[0]['content']

    parsed_data = chain.from_iterable(
        spacy_batch_processing(raw_df, '', 'content', ['content'], 'crf'))
    prepared_data = [list(x[1]) for x in
                     groupby(parsed_data, lambda x: x == ('##END', '###', 'O'))
                     if not x[0]]
    test_sents = batch_add_features(prepared_data, f_dics)

    X_test, y_test = feed_crf_trainer(test_sents, conf)
    crf_result = crf_predict(crf, prepared_data, X_test)

    return crf_result, raw_df


##############################################################################


# Refactoring

# todo


def pipeline_pos_crf(train_f, test_f, model_f, conf, hdf_f, hdf_key, report_type, cols,
                     pieces=10):
    crf, f_dics = batch_loading('', hdf_f, hdf_key)
    raw_df = pd.read_json(train_f, lines=True)
    basic_logging('Reading ends')
    data = pd.DataFrame(raw_df.result.values.tolist())['content'].reset_index()
    data['content'] = data['content'].drop_duplicates()
    # data = random_rows(data, pieces, 'content')
    data = data.dropna()
    basic_logging('Cleaning ends')

    parsed_data = spacy_batch_processing(data, '', 'content', ['content'], 'crf')
    basic_logging('Spacy ends')

    parsed_data = chain.from_iterable(parsed_data)
    pos_data = [list(x[1])[:-1] for x in groupby(parsed_data,
                                                 lambda x: x == ('##END', '###', 'O'))
                if not x[0]]

    test_sents = batch_add_features(pos_data, f_dics)
    basic_logging('Adding features ends')

    X_test, y_test = feed_crf_trainer(test_sents, conf)
    basic_logging('Conversion ends')
    result = crf_predict(crf, pos_data, X_test)
    basic_logging('Predicting ends')
    out = pd.DataFrame(result)
    out.to_csv(out_f, header=False, index=False)



def pipeline_streaming_folder(in_folder, out_folder, dict_conf, crf_f, feature_hdf,
                              hdf_keys, switch):
    loads = batch_loading(dict_conf, crf_f, feature_hdf, hdf_keys, switch)
    conf, crf, aca, com_single, com_suffix, location, name, ticker, tfdf, tfidf = loads

    i = 0
    root_dic = defaultdict()
    for in_f in listdir(in_folder):
        ff = path.join(in_folder, in_f)
        crf_result, raw_df = streaming_pos_crf(ff, crf, conf, aca, com_single, com_suffix,
                                               location, name, ticker, tfdf, tfidf)
        # json_result = convert_crf_result_json(crf_result, raw_df)
        if modf(i / 100)[0] == 0.0:
            print(get_now(), i)
        # with open(path.join(out_folder, str(in_f) + '.json'), 'w') as out:
        with open(path.join(out_folder, str(in_f) + '.csv'), 'w') as out:

            # out.write(json_result)
            pd.DataFrame(crf_result).to_csv(out, index=None, header=None)

        i += 1
        if modf(i / 100)[0] == 0.0:
            basic_logging('%d pieces have been processed', i)
    result = pd.DataFrame.from_dict(root_dic, orient='index').reset_index()
    result.columns = ['Token', 'Freq']
    result = result.sort_values(by='Freq', ascending=False)
    result.to_csv(out_folder, header=None, index=False)


def pipeline_streaming_queue(redis_conf, dict_conf, crf_f, feature_hdf, hdf_keys, switch):
    loads = batch_loading(dict_conf, crf_f, feature_hdf, hdf_keys, switch)

    conf, crf, aca, com_single, com_suffix, location, name, ticker, tfdf, tfidf = loads
    r_address, r_port, r_db, r_key = OrderedDict(
        load_yaml_conf(redis_conf)['test_read']).values()
    w_address, w_port, w_db, w_key = OrderedDict(
        load_yaml_conf(redis_conf)['test_write']).values()

    r = redis.StrictRedis(host=r_address, port=r_port, db=r_db)
    w = redis.StrictRedis(host=w_address, port=w_port, db=w_db)

    i = 0

    while True:
        queue = r.lpop(r_key).decode('utf-8')
        json_result = streaming_pos_crf(queue, crf, conf, aca, com_single, com_suffix,
                                        location, name, ticker, tfdf, tfidf)
        w.lpush(w_key, json_result)
        i += 1
        if modf(i / 10)[0] == 0.0:
            print(get_now(), i)
