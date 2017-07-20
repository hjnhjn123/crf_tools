# -*- coding: utf-8 -*-

import joblib as jl
import pandas as pd

from .arsenal_crf import process_annotated, batch_add_features, batch_loading, feed_crf_trainer, df2crfsuite, \
    voting, load_multi_models, module_crf_train, module_crf_fit, module_prepare_folder, module_prepare_news_jsons, \
    module_crf_cv
from .arsenal_logging import basic_logging
from .arsenal_stats import get_now
from .arsenal_test import evaluate_ner_result
from .settings import *


##############################################################################


# Training


def pipeline_train(train_f, test_f, model_f, result_f, hdf_f, hdf_key, feature_conf, window_size, col_names):
    """
    A pipeline for CRF training
    :param train_f: train dataset in a 3-column csv (TOKEN, LABEL, POS)
    :param test_f: test dataset in a 3-column csv (TOKEN, LABEL, POS)
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

    crf, _, _ = module_crf_train(train_df, f_dics, feature_conf, hdf_key, window_size)
    _, _, _ = module_crf_fit(test_df, crf, f_dics, feature_conf, hdf_key, window_size, result_f)

    if model_f:
        jl.dump(crf, model_f)
    return crf


def pipeline_train_mix(in_folder, model_f, result_f, hdf_f, hdf_key, feature_conf, window_size, ner_tags, col_names):
    """
    A pipeline for CRF training                                                                                                         
    :param train_fs: train dataset in a 3-column csv (TOKEN, LABEL, POS)
    :param test_f: test dataset in a 3-column csv (TOKEN, LABEL, POS)
    :param model_f: model file
    :param feature_conf: feature configurations
    :param                                                                                                                                                                               hdf_f: feature HDF5 file
    :param hdf_key: keys of feature HDF5 file
    """
    basic_logging('loading conf begins')
    f_dics = batch_loading(hdf_f, hdf_key)
    basic_logging('loading conf ends')

    test_df, train_df = module_prepare_folder(col_names, in_folder, ner_tags)
    crf, _, _ = module_crf_train(train_df, f_dics, feature_conf, hdf_key, window_size)
    _, _, _ = module_crf_fit(test_df, crf, f_dics, feature_conf, hdf_key, window_size, result_f)

    if model_f:
        jl.dump(crf, model_f)

    return crf


##############################################################################


def pipeline_cv(train_f, test_f, model_f, result_f, feature_conf, hdf_f, hdf_key, cv, iteration,
                window_size, col_names):
    """
    A pipeline for CRF training
    :param train_f: train dataset in a 3-column csv (TOKEN, LABEL, POS)
    :param test_f: test dataset in a 3-column csv (TOKEN, LABEL, POS)
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

    crf, X_train, y_train = module_crf_train(train_df, f_dics, feature_conf, hdf_key, window_size)
    best_predictor = module_crf_cv(crf, X_train, y_train, cv, iteration)
    y_pred, _, y_test = module_crf_fit(test_df, best_predictor, f_dics, feature_conf, hdf_key, window_size, result_f)

    result, _ = evaluate_ner_result(y_pred, y_test)
    result.to_csv(result_f, index=False)
    if model_f:
        jl.dump(best_predictor, model_f)
    return crf, best_predictor


def pipeline_cv_mix(in_folder, model_f, result_f, feature_conf, hdf_f, hdf_key, cv, iteration,
                    window_size, ner_tags, col_names):
    """
    A pipeline for CRF training
    :param train_f: train dataset in a 3-column csv (TOKEN, LABEL, POS)
    :param test_f: test dataset in a 3-column csv (TOKEN, LABEL, POS)
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

    test_df, train_df = module_prepare_folder(col_names, in_folder, ner_tags)
    crf, X_train, y_train = module_crf_train(train_df, f_dics, feature_conf, hdf_key, window_size)
    best_predictor = module_crf_cv(crf, X_train, y_train, cv, iteration)
    y_pred, _, y_test = module_crf_fit(test_df, best_predictor, f_dics, feature_conf, hdf_key, window_size, result_f)

    result, _ = evaluate_ner_result(y_pred, y_test)
    result.to_csv(result_f, index=False)
    if model_f:
        jl.dump(best_predictor, model_f)
    return crf, best_predictor


##############################################################################

# Validation


def pipeline_validate(valid_f, model_f, feature_conf, hdf_f, result_f, hdf_key, window_size, col_names):
    """
    A pipeline for CRF validating
    :param valid_df: validate dataset with at least two columns (TOKEN, LABEL)
    :param model_f: model file
    :param feature_conf: feature configurations
    :param hdf_f: feature HDF5 file
    :param hdf_key: keys of feature HDF5 file
    :param window_size:
    """
    valid_df = process_annotated(valid_f, col_names)
    basic_logging('loading conf begins')
    crf = jl.load(model_f)
    f_dics = batch_loading(hdf_f, hdf_key)
    basic_logging('loading conf ends')
    y_pred, X_test, y_test = module_crf_fit(valid_df, crf, f_dics, feature_conf, hdf_key, window_size, result_f)
    result, _ = evaluate_ner_result(y_pred, y_test)
    result.to_csv(result_f, index=False)
    return result


##############################################################################


def module_batch_annotate_single_model(prepared_df, model_f, hdf_f, hdf_key, feature_conf, window_size):
    """
    :param prepared_df: a df with at-least two columns
    :param out_f: CSV FILE, the ouptut file
    :param model_f: NUMPY PICKLE FILE, the model file
    :param hdf_f: HDF FILE, the hdf file of feature dicts or lists
    :param hdf_key: LIST, the key to extract hdf file
    :param feature_conf: DICT, features used to compute
    :param window_size: INT, set window size for CRF tagging
    :param col_names: LIST, the column in json file to be used
    """
    basic_logging('loading conf begins')
    model = jl.load(model_f)
    f_dics = batch_loading(hdf_f, hdf_key)
    basic_logging('loading conf ends')

    # raw_list = (pd.read_json('/'.join((in_folder, in_f)), col_names) for in_f in listdir(in_folder))
    # basic_logging('reading files ends')

    y_pred, _, _ = module_crf_fit(prepared_df, model, f_dics, feature_conf, hdf_key, window_size, '')

    recovered_pred = [i + ['O'] for i in y_pred]
    crf_result = [i for j in recovered_pred for i in j]
    final_result = pd.concat([prepared_data[0], pd.DataFrame(crf_result), prepared_data[2]], axis=1)
    basic_logging('converting results ends')
    return pd.DataFrame(final_result)


def module_batch_annotate_multi_model(prepared_df, model_fs, hdf_f, hdf_key, feature_conf, window_size):
    """
    :param prepared_df: a df with at-least wo columns
    :param out_f: CSV FILE, the ouptut file
    :param model_fs: NUMPY PICKLE FILES, the model files
    :param hdf_f: HDF FILE, the hdf file of feature dicts or lists
    :param hdf_key: LIST, the key to extract hdf file
    :param feature_conf: DICT, features used to compute
    :param window_size: INT, set window size for CRF tagging
    :param col_names: LIST, the column in json file to be used
    """

    basic_logging('loading conf begins')
    model_dics = load_multi_models(model_fs)
    f_dics = batch_loading(hdf_f, hdf_key)
    basic_logging('loading conf ends')

    test_df = batch_add_features(prepared_df, f_dics)
    test_sents = df2crfsuite(test_df)
    X_test, y_test = feed_crf_trainer(test_sents, feature_conf, hdf_key, window_size)
    basic_logging('converting features ends')

    crf_results = {name: crf_predict(model, test_sents, X_test) for name, model in model_dics.items()}
    final_result = voting(crf_results)
    basic_logging('converting results ends')
    return pd.DataFrame(final_result)


##############################################################################


def pipeline_predict_jsons_single_model(in_folder, out_f, model_f, col, hdf_f, hdf_key, row_count, feature_conf,
                                        window_size, col_names):
    """
    It reads all files in a single folder, and randomly select of them to annotate
    :param in_folder: DIR, the folder waiting for annotation
    :param out_f: CSV FILE, the ouptut file
    :param model_fs: LIST, the model files
    :param col: LIST, the column used to annotate
    :param hdf_f: HDF FILE, the hdf file of feature dicts or lists
    :param hdf_key: LIST, the key to extract hdf file
    :param row_count: INT, random rows
    :param feature_conf: DICT, features used to compute
    :param window_size: INT, set window size for CRF tagging
    :param col_names: LIST, the column in json file to be used
    """
    prepared_df = module_prepare_news_jsons(in_folder, col, row_count, feature_conf, col_names)
    result = module_batch_annotate_single_model(prepared_df, out_f, model_f, hdf_f, hdf_key, feature_conf, window_size)
    result.to_csv(out_f, index=False, header=None)


def pipeline_batch_annotate_multi_model(in_folder, out_f, model_fs, col, hdf_f, hdf_key, row_count, feature_conf,
                                        window_size, col_names):
    """
    It reads all files in a single folder, and randomly select of them to annotate
    :param in_folder: DIR, the folder waiting for annotation
    :param out_f: CSV FILE, the ouptut file
    :param model_fs: LIST, the model files
    :param col: LIST, the column used to annotate
    :param hdf_f: HDF FILE, the hdf file of feature dicts or lists
    :param hdf_key: LIST, the key to extract hdf file
    :param row_count: INT, random rows
    :param feature_conf: DICT, features used to compute
    :param window_size: INT, set window size for CRF tagging
    :param col_names: LIST, the column in json file to be used
    """
    prepared_df = module_prepare_news_jsons(in_folder, col, row_count, feature_conf, col_names)
    result = module_batch_annotate_multi_model(prepared_df, out_f, model_fs, hdf_f, hdf_key, feature_conf, window_size)
    result.to_csv(out_f, index=False, header=None)


##############################################################################


RESULT_F = '_'.join((RESULT_F, get_now(), '.csv'))


def main(argv):
    print(argv)
    print()
    dic = {
        'train': lambda: pipeline_train(train_f=TRAIN_F, test_f=TEST_F, model_f=MODEL_F, result_f=RESULT_F, hdf_f=HDF_F,
                                        hdf_key=HDF_KEY, feature_conf=FEATURE_CONF, window_size=WINDOW_SIZE,
                                        col_names=HEADER),
        'cv': lambda: pipeline_cv(train_f=TRAIN_F, test_f=TEST_F, model_f=MODEL_F, result_f=RESULT_F, hdf_f=HDF_F,
                                  hdf_key=HDF_KEY, feature_conf=FEATURE_CONF, window_size=WINDOW_SIZE, cv=CV,
                                  iteration=ITERATION, col_names=HEADER),
        'validate': lambda: pipeline_validate(valid_f=VALIDATE_F, model_f=MODEL_F, result_f=RESULT_F, hdf_f=HDF_F,
                                              hdf_key=HDF_KEY, feature_conf=FEATURE_CONF, window_size=WINDOW_SIZE,
                                              col_names=HEADER),
        'annotate': lambda: pipeline_predict_jsons_single_model(in_folder=TRAIN_F, out_f=TEST_F, model_f=MODEL_F,
                                                                result_f=RESULT_F, hdf_f=HDF_F, hdf_key=HDF_KEY,
                                                                feature_conf=FEATURE_CONF, window_size=WINDOW_SIZE,
                                                                col_names=HEADER),
    }
    dic[argv]()
