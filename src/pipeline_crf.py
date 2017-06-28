# -*- coding: utf-8 -*-

import joblib as jl
import pandas as pd
from os import listdir

from .arsenal_crf import process_annotated, batch_add_features, batch_loading, feed_crf_trainer, df2crfsuite, train_crf, \
    make_param_space, make_f1_scorer, search_param, merge_ner_tags
from .arsenal_logging import basic_logging
from .arsenal_stats import get_now
from .arsenal_test import show_crf_label, evaluate_ner_result
from .settings import *


##############################################################################


# Pipelines


def pipeline_train(train_f, test_f, model_f, result_f, hdf_f, hdf_key, feature_conf, window_size):
    """
    A pipeline for CRF training
    :param train_f: train dataset in a 3-column csv (TOKEN, POS, LABEL)
    :param test_f: test dataset in a 3-column csv (TOKEN, POS, LABEL)
    :param model_f: model file
    :param feature_conf: feature configurations
    :param hdf_f: feature HDF5 file
    :param hdf_key: keys of feature HDF5 file
    """
    basic_logging('loading conf begins')
    f_dics = batch_loading(hdf_f, hdf_key)
    basic_logging('loading conf ends')
    train_df, test_df = process_annotated(train_f), process_annotated(test_f)
    basic_logging('loading data ends')
    train_df = batch_add_features(train_df, f_dics)
    test_df = batch_add_features(test_df, f_dics)
    basic_logging('adding features ends')
    train_sents = df2crfsuite(train_df)
    test_sents = df2crfsuite(test_df)
    basic_logging('converting to crfsuite ends')
    X_train, y_train = feed_crf_trainer(train_sents, feature_conf, hdf_key, window_size)
    X_test, y_test = feed_crf_trainer(test_sents, feature_conf, hdf_key, window_size)
    basic_logging('computing features ends')
    crf = train_crf(X_train, y_train)
    basic_logging('training ends')
    y_pred = crf.predict(X_test)
    result, indexed_ner = evaluate_ner_result(y_pred, y_test)
    result.to_csv(result_f, index=False)
    basic_logging('testing ends')
    if model_f:
        jl.dump(crf, model_f)
    return crf, result


def pipeline_train_mix(test_f, model_f, result_f, hdf_f, hdf_key, feature_conf, window_size, ner_tags, *train_fs):
    """
    A pipeline for CRF training
    :param train_fs: train dataset in a 3-column csv (TOKEN, POS, LABEL)
    :param test_f: test dataset in a 3-column csv (TOKEN, POS, LABEL)
    :param model_f: model file
    :param feature_conf: feature configurations
    :param hdf_f: feature HDF5 file
    :param hdf_key: keys of feature HDF5 file
    """
    basic_logging('loading conf begins')
    f_dics = batch_loading(hdf_f, hdf_key)
    basic_logging('loading conf ends')
    train_df = pd.concat([process_annotated(f) for f in train_fs])
    print(train_df.info())
    test_df = process_annotated(test_f)
    basic_logging('loading data ends')
    if ner_tags:
        train_df = merge_ner_tags(train_df, 'NER', ner_tags)
        test_df = merge_ner_tags(test_df, 'NER', ner_tags)
    train_df = batch_add_features(train_df, f_dics)
    test_df = batch_add_features(test_df, f_dics)
    basic_logging('adding features ends')
    train_sents = df2crfsuite(train_df)
    test_sents = df2crfsuite(test_df)
    basic_logging('converting to crfsuite ends')
    X_train, y_train = feed_crf_trainer(train_sents, feature_conf, hdf_key, window_size)
    X_test, y_test = feed_crf_trainer(test_sents, feature_conf, hdf_key, window_size)
    basic_logging('computing features ends')
    crf = train_crf(X_train, y_train)
    basic_logging('training ends')
    y_pred = crf.predict(X_test)
    result, indexed_ner = evaluate_ner_result(y_pred, y_test)
    result.to_csv(result_f, index=False)
    basic_logging('testing ends')
    if model_f:
        jl.dump(crf, model_f)
    return crf, result


def pipeline_best_predict(train_f, test_f, model_f, result_f, feature_conf, hdf_f, hdf_key, cv, iteration,
                          window_size):
    """
    A pipeline for CRF training
    :param train_f: train dataset in a 3-column csv (TOKEN, POS, LABEL)
    :param test_f: test dataset in a 3-column csv (TOKEN, POS, LABEL)
    :param model_f: model file
    :param feature_conf: feature configurations
    :param hdf_f: feature HDF5 file
    :param hdf_key: keys of feature HDF5 file
    :param cv: cv scale
    :param iteration: iteration time
    """
    basic_logging('loading conf begins')
    f_dics = batch_loading(hdf_f, hdf_key)
    basic_logging('loading conf ends')
    train_df, test_df = process_annotated(train_f), process_annotated(test_f)
    basic_logging('loading data ends')
    train_df = batch_add_features(train_df, f_dics)
    test_df = batch_add_features(test_df, f_dics)
    basic_logging('adding features ends')
    train_sents = df2crfsuite(train_df)
    test_sents = df2crfsuite(test_df)
    basic_logging('converting to crfsuite ends')
    X_train, y_train = feed_crf_trainer(train_sents, feature_conf, hdf_key, window_size)
    X_test, y_test = feed_crf_trainer(test_sents, feature_conf, hdf_key, window_size)
    basic_logging('Conversion ends')
    crf = train_crf(X_train, y_train)
    labels = show_crf_label(crf)
    params_space = make_param_space()
    f1_scorer = make_f1_scorer(labels)
    basic_logging('cv begins')
    rs_cv = search_param(X_train, y_train, crf, params_space, f1_scorer, cv, iteration)
    basic_logging('cv ends')
    best_predictor = rs_cv.best_estimator_
    y_pred = crf.predict(X_test)
    result, indexed_ner = evaluate_ner_result(y_pred, y_test)
    # result.to_csv(result_f, index=False)
    basic_logging('testing ends')
    if model_f:
        jl.dump(best_predictor, model_f)
    return crf, best_predictor, rs_cv, result


def pipeline_best_predict_mix(test_f, model_f, result_f, feature_conf, hdf_f, hdf_key, cv, iteration, window_size,
                              ner_tags, *train_fs):
    """
    A pipeline for CRF training
    :param train_f: train dataset in a 3-column csv (TOKEN, POS, LABEL)
    :param test_f: test dataset in a 3-column csv (TOKEN, POS, LABEL)
    :param model_f: model file
    :param feature_conf: feature configurations
    :param hdf_f: feature HDF5 file
    :param hdf_key: keys of feature HDF5 file
    :param cv: cv scale
    :param iteration: iteration time
    """
    basic_logging('loading conf begins')
    f_dics = batch_loading(hdf_f, hdf_key)
    basic_logging('loading conf ends')
    train_df = pd.concat([process_annotated(f) for f in train_fs])
    print(train_df.info())
    test_df = process_annotated(test_f)
    basic_logging('loading data ends')
    if ner_tags:
        train_df = merge_ner_tags(train_df, 'NER', ner_tags)
        test_df = merge_ner_tags(test_df, 'NER', ner_tags)
    train_df = batch_add_features(train_df, f_dics)
    test_df = batch_add_features(test_df, f_dics)
    basic_logging('adding features ends')
    train_sents = df2crfsuite(train_df)
    test_sents = df2crfsuite(test_df)
    basic_logging('converting to crfsuite ends')
    X_train, y_train = feed_crf_trainer(train_sents, feature_conf, hdf_key, window_size)
    X_test, y_test = feed_crf_trainer(test_sents, feature_conf, hdf_key, window_size)
    basic_logging('Conversion ends')
    crf = train_crf(X_train, y_train)
    labels = show_crf_label(crf)
    params_space = make_param_space()
    f1_scorer = make_f1_scorer(labels)
    basic_logging('cv begins')
    rs_cv = search_param(X_train, y_train, crf, params_space, f1_scorer, cv, iteration)
    basic_logging('cv ends')
    best_predictor = rs_cv.best_estimator_
    y_pred = crf.predict(X_test)
    result, indexed_ner = evaluate_ner_result(y_pred, y_test)
    # result.to_csv(result_f, index=False)
    basic_logging('testing ends')
    if model_f:
        jl.dump(best_predictor, model_f)
    return crf, best_predictor, rs_cv, result


def pipeline_validate(valid_f, model_f, feature_conf, hdf_f, result_f, hdf_key, window_size, ner_tags, diff_f):
    """
    A pipeline for CRF training
    :param valid_f: test dataset in a 3-column csv (TOKEN, POS, LABEL)
    :param model_f: model file
    :param feature_conf: feature configurations
    :param hdf_f: feature HDF5 file
    :param hdf_key: keys of feature HDF5 file
    :param window_size:
    :param ner_tags: a list of tags
    """
    basic_logging('loading conf begins')
    crf = jl.load(model_f)
    f_dics = batch_loading(hdf_f, hdf_key)
    basic_logging('loading conf ends')
    valid_df = process_annotated(valid_f)
    if ner_tags:
        valid_df = merge_ner_tags(valid_df, 'NER', ner_tags)
    valid_df = batch_add_features(valid_df, f_dics)
    basic_logging('adding features ends')
    test_sents = df2crfsuite(valid_df)
    basic_logging('converting to crfsuite ends')
    X_test, y_test = feed_crf_trainer(test_sents, feature_conf, hdf_key, window_size)
    basic_logging('Conversion ends')
    y_pred = crf.predict(X_test)
    result, indexed_ner = evaluate_ner_result(y_pred, y_test)
    diff = compare_pred_test(X_test, indexed_ner)
    diff.to_csv(diff_f, index=False)
    result.to_csv(result_f, index=False)
    return result


def pipelinne_batch_annotate(in_folder, out_f, model_fs, col):
    model_dics = load_multi_models(model_fs)
    f_dics = batch_loading(hdf_f, hdf_key)
    raw_list = [pd.read_json('/'.join((in_folder, in_f))) for in_f in listdir(in_folder)]
    print('files: ', len(raw_list))
    raw_df = pd.concat(raw_list, axis=0)
    raw_df['content'] = raw_df[col].to_dict()[0]['content']
    parsed_data = chain.from_iterable(spacy_batch_processing(raw_df, '', 'content', ['content'], 'crf'))
    prepared_data = pd.DataFrame(list(parsed_data))
    test_df = batch_add_features(prepared_data, f_dics)
    test_sents = df2crfsuite(test_df)
    X_test, y_test = feed_crf_trainer(test_sents, feature_conf, hdf_key, window_size)
    crf_results = {name: crf_predict(model, test_sents, X_test) for name, model in model_dics.items()}
    final_result = voting(crf_results)
    final_result.to_csv(out_f, index=False, head=None)

##############################################################################


RESULT_F = '_'.join((RESULT_F, get_now(), '.csv'))


def main(argv):
    print(argv)
    print()
    dic = {
        'train': lambda: pipeline_train(train_f=TRAIN_F, test_f=TEST_F, model_f=MODEL_F,
                                        result_f=RESULT_F, hdf_f=HDF_F, hdf_key=HDF_KEY,
                                        feature_conf=FEATURE_CONF,
                                        window_size=WINDOW_SIZE),
        'cv': lambda: pipeline_best_predict(train_f=TRAIN_F, test_f=TEST_F,
                                            model_f=MODEL_F,
                                            result_f=RESULT_F, hdf_f=HDF_F,
                                            hdf_key=HDF_KEY,
                                            feature_conf=FEATURE_CONF,
                                            window_size=WINDOW_SIZE, cv=CV,
                                            iteration=ITERATION),
        'validate': lambda: pipeline_validate(valid_f=VALIDATE_F, model_f=MODEL_F,
                                              result_f=RESULT_F, hdf_f=HDF_F,
                                              hdf_key=HDF_KEY,
                                              feature_conf=FEATURE_CONF,
                                              window_size=WINDOW_SIZE)
    }
    dic[argv]()
