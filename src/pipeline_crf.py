# -*- coding: utf-8 -*-

import gc
from itertools import chain
from os import listdir

import joblib as jl
import pandas as pd

from .arsenal_crf import process_annotated, batch_add_features, batch_loading, feed_crf_trainer, df2crfsuite, \
    make_param_space, make_f1_scorer, search_param, merge_ner_tags, voting, load_multi_models, crf_train, crf_fit
from .arsenal_logging import basic_logging
from .arsenal_spacy import spacy_batch_processing
from .arsenal_stats import get_now, random_rows
from .arsenal_test import show_crf_label, evaluate_ner_result, compare_pred_test
from .settings import *


##############################################################################


# Pipelines


def pipeline_train(train_f, test_f, model_f, result_f, hdf_f, hdf_key, feature_conf, window_size, col_names):
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
    train_df, test_df = process_annotated(train_f, col_names), process_annotated(test_f, col_names)
    basic_logging('loading data ends')

    crf, _, _ = crf_train(train_df, f_dics, feature_conf, hdf_key, window_size)
    _, _ = crf_fit(test_df, crf, f_dics, feature_conf, hdf_key, window_size, result_f)

    if model_f:
        jl.dump(crf, model_f)
    return crf


def pipeline_train_mix(in_folder, model_f, result_f, hdf_f, hdf_key, feature_conf, window_size, ner_tags, col_names):
    """
    A pipeline for CRF training                                                                                                         
    :param train_fs: train dataset in a 3-column csv (TOKEN, POS, LABEL)
    :param test_f: test dataset in a 3-column csv (TOKEN, POS, LABEL)
    :param model_f: model file
    :param feature_conf: feature configurations
    :param                                                                                                                                                                               hdf_f: feature HDF5 file
    :param hdf_key: keys of feature HDF5 file
    """
    basic_logging('loading conf begins')
    f_dics = batch_loading(hdf_f, hdf_key)
    basic_logging('loading conf ends')
    train_df = pd.concat(
        [process_annotated('/'.join((in_folder, in_f)), col_names) for in_f in listdir(in_folder) if 'train' in in_f], axis=0)
    print(train_df.info())
    test_df = pd.concat(
        [process_annotated('/'.join((in_folder, in_f)), col_names) for in_f in listdir(in_folder) if 'test' in in_f], axis=0)
    basic_logging('loading data ends')
    if ner_tags:
        train_df = merge_ner_tags(train_df, 'NER', ner_tags)
        test_df = merge_ner_tags(test_df, 'NER', ner_tags)

    crf, _, _ = crf_train(train_df, f_dics, feature_conf, hdf_key, window_size)

    _, _ = crf_fit(test_df, crf, f_dics, feature_conf, hdf_key, window_size, result_f)

    if model_f:
        jl.dump(crf, model_f)

    return crf


def pipeline_best_predict(train_f, test_f, model_f, result_f, feature_conf, hdf_f, hdf_key, cv, iteration,
                          window_size, col_names):
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
    train_df, test_df = process_annotated(train_f, col_names), process_annotated(test_f, col_names)
    basic_logging('loading data ends')

    crf, X_train, y_train = crf_train(train_df, f_dics, feature_conf, hdf_key, window_size)

    labels = show_crf_label(crf)
    params_space = make_param_space()
    f1_scorer = make_f1_scorer(labels)
    gc.collect()
    basic_logging('cv begins')
    rs_cv = search_param(X_train, y_train, crf, params_space, f1_scorer, cv, iteration)
    basic_logging('cv ends')
    best_predictor = rs_cv.best_estimator_

    _, _ = crf_fit(test_df, crf, f_dics, feature_conf, hdf_key, window_size)

    if model_f:
        jl.dump(best_predictor, model_f)
    return crf, best_predictor, rs_cv, result


def pipeline_best_predict_mix(in_folder, model_f, result_f, feature_conf, hdf_f, hdf_key, cv, iteration, window_size,
                              ner_tags, col_names):

    """q
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
    # train_df = pd.concat([process_annotated(f) for f in train_fs])
    # test_df = process_annotated(test_f)

    train_df = pd.concat(
        [process_annotated('/'.join((in_folder, in_f)), col_names) for in_f in listdir(in_folder) if 'train' in in_f], axis=0)
    print(train_df.info())
    test_df = pd.concat(
        [process_annotated('/'.join((in_folder, in_f)), col_names) for in_f in listdir(in_folder) if 'test' in in_f], axis=0)
    basic_logging('loading data ends')
    print(test_df.info())

    basic_logging('loading data ends')
    if ner_tags:
        train_df = merge_ner_tags(train_df, 'NER', ner_tags)
        test_df = merge_ner_tags(test_df, 'NER', ner_tags)

    crf, X_train, y_train = crf_train(train_df, f_dics, feature_conf, hdf_key, window_size)

    labels = show_crf_label(crf)
    params_space = make_param_space()
    f1_scorer = make_f1_scorer(labels)
    gc.collect()
    basic_logging('cv begins')
    rs_cv = search_param(X_train, y_train, crf, params_space, f1_scorer, cv, iteration)
    basic_logging('cv ends')
    best_predictor = rs_cv.best_estimator_

    _, _ = crf_fit(test_df, crf, f_dics, feature_conf, hdf_key, window_size)

    if model_f:
        jl.dump(best_predictor, model_f)
    return crf, best_predictor, rs_cv, result


def pipeline_validate(valid_f, model_f, feature_conf, hdf_f, result_f, hdf_key, window_size, ner_tags, diff_f, col_names):
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
    valid_df = process_annotated(valid_f, col_names)
    if ner_tags:
        valid_df = merge_ner_tags(valid_df, 'NER', ner_tags)

    y_pred, y_test, X_test = crf_fit(valid_df, crf, f_dics, feature_conf, hdf_key, window_size, result_f)

    result, indexed_ner = evaluate_ner_result(y_pred, y_test)
    # diff = compare_pred_test(X_test, indexed_ner)
    # diff.to_csv(diff_f, index=False)
    result.to_csv(result_f, index=False)
    return result


def pipeline_batch_annotate_single_model(in_folder, out_f, model_f, col, hdf_f, hdf_key, row_count, feature_conf,
                                         window_size, col_names):
    """
    It reads all files in a single folder, and randomly select of them to annotate
    :param in_folder: DIR, the folder waiting for annotation
    :param out_f: CSV FILE, the ouptut file
    :param model_f: NUMPY PICKLE FILE, the model file
    :param col: LIST, the column used to annotate
    :param hdf_f: HDF FILE, the hdf file of feature dicts or lists
    :param hdf_key: LIST, the key to extract hdf file
    :param row_count: INT, random rows
    :param feature_conf: DICT, features used to compute
    :param window_size: INT, set window size for CRF tagging
    :param col_names: LIST, the column in json file to be used
    """
    basic_logging('loading conf begins')
    model = jl.load(model_f)
    f_dics = batch_loading(hdf_f, hdf_key)
    basic_logging('loading conf ends')
    raw_list = (pd.read_json('/'.join((in_folder, in_f)), col_names) for in_f in listdir(in_folder))
    basic_logging('reading files ends')
    raw_df = pd.concat(raw_list, axis=0)
    print('files: ', len(raw_df))


    random_df = random_rows(raw_df, row_count)
    basic_logging('selecting lines ends')
    random_df['content'] = random_df[col].apply(lambda x: x['content'])

    parsed_data = chain.from_iterable(spacy_batch_processing(random_df, '', 'content', ['content'], 'crf'))
    prepared_data = pd.DataFrame(list(parsed_data))
    basic_logging('extracting data ends')

    y_pred, _ = crf_fit(prepared_data, model, f_dics, feature_conf, hdf_key, window_size, '')

    recovered_pred = [i + ['O'] for i in y_pred]
    crf_result = [i for j in recovered_pred for i in j]
    final_result = pd.concat([prepared_data[0], pd.DataFrame(crf_result), prepared_data[2]], axis=1)
    basic_logging('converting results ends')
    pd.DataFrame(final_result).to_csv(out_f, index=False, header=None)


def pipeline_batch_annotate_multi_model(in_folder, out_f, model_fs, col, hdf_f, hdf_key, row_count, feature_conf,
                                        window_size, col_names):
    """
    It reads all files in a single folder, and randomly select of them to annotate
    :param in_folder: DIR, the folder waiting for annotation
    :param out_f: CSV FILE, the ouptut file
    :param model_fs: NUMPY PICKLE FILES, the model files
    :param col: LIST, the column used to annotate
    :param hdf_f: HDF FILE, the hdf file of feature dicts or lists
    :param hdf_key: LIST, the key to extract hdf file
    :param row_count: INT, random rows
    :param feature_conf: DICT, features used to compute
    :param window_size: INT, set window size for CRF tagging
    :param col_names: LIST, the column in json file to be used
    """
    basic_logging('loading conf begins')
    model_dics = load_multi_models(model_fs)
    f_dics = batch_loading(hdf_f, hdf_key)
    basic_logging('loading conf ends')
    raw_list = (pd.read_json('/'.join((in_folder, in_f)), col_names) for in_f in listdir(in_folder))
    basic_logging('reading files ends')
    print('files: ', len(raw_list))
    raw_df = pd.concat(raw_list, axis=0)
    random_df = random_rows(raw_df, row_count)
    basic_logging('selecting lines ends')
    random_df['content'] = random_df[col].apply(lambda x: x['content'])
    parsed_data = chain.from_iterable(spacy_batch_processing(random_df, '', 'content', ['content'], 'crf'))
    prepared_data = pd.DataFrame(list(parsed_data))
    basic_logging('extracting data ends')
    test_df = batch_add_features(prepared_data, f_dics)
    test_sents = df2crfsuite(test_df)
    basic_logging('converting features ends')
    X_test, y_test = feed_crf_trainer(test_sents, feature_conf, hdf_key, window_size)
    crf_results = {name: crf_predict(model, test_sents, X_test) for name, model in model_dics.items()}
    final_result = voting(crf_results)
    basic_logging('converting results ends')
    pd.DataFrame(final_result).to_csv(out_f, index=False, header=None)


##############################################################################


RESULT_F = '_'.join((RESULT_F, get_now(), '.csv'))


def main(argv):
    print(argv)
    print()
    dic = {
        'train': lambda: pipeline_train(train_f=TRAIN_F, test_f=TEST_F, model_f=MODEL_F, result_f=RESULT_F, hdf_f=HDF_F,
                                        hdf_key=HDF_KEY, feature_conf=FEATURE_CONF, window_size=WINDOW_SIZE,
                                        col_names=HEADER),
        'cv': lambda: pipeline_best_predict(train_f=TRAIN_F, test_f=TEST_F, model_f=MODEL_F, result_f=RESULT_F,
                                            hdf_f=HDF_F, hdf_key=HDF_KEY, feature_conf=FEATURE_CONF,
                                            window_size=WINDOW_SIZE, cv=CV, iteration=ITERATION, col_names=HEADER),
        'validate': lambda: pipeline_validate(valid_f=VALIDATE_F, model_f=MODEL_F, result_f=RESULT_F, hdf_f=HDF_F,
                                              hdf_key=HDF_KEY, feature_conf=FEATURE_CONF, window_size=WINDOW_SIZE),
        'annotate': lambda: pipeline_batch_annotate_single_model(in_folder=TRAIN_F, out_f=TEST_F, model_f=MODEL_F,
                                                                 result_f=RESULT_F, hdf_f=HDF_F, hdf_key=HDF_KEY,
                                                                 feature_conf=FEATURE_CONF, window_size=WINDOW_SIZE, col_names=HEADER),
    }
    dic[argv]()
