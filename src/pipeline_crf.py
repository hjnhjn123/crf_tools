# -*- coding: utf-8 -*-

from .arsenal_crf import process_annotated, batch_add_features, batch_loading, feed_crf_trainer, test_crf_prediction, \
    df2crfsuite, train_crf, show_crf_label, make_param_space, make_f1_scorer
from .arsenal_logging import basic_logging
from .arsenal_test import test_crf_prediction
from .settings import *


##############################################################################


# Pipelines


def pipeline_train(train_f, test_f, model_f, result_f, hdf_f, hdf_key, feature_conf, report_type, window_size):
    """
    A pipeline for CRF training
    :param train_f: train dataset in a 3-column csv (TOKEN, POS, LABEL)
    :param test_f: test dataset in a 3-column csv (TOKEN, POS, LABEL)
    :param model_f: model file
    :param feature_conf: feature configurations
    :param hdf_f: feature HDF5 file
    :param hdf_key: keys of feature HDF5 file
    :param report_type: 'spc' for a specific report and 'bin' for binary report
    """
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
    X_train, y_train = feed_crf_trainer(train_sents, feature_conf, hdf_key, window_size)
    X_test, y_test = feed_crf_trainer(test_sents, feature_conf, hdf_key, window_size)
    basic_logging('computing features ends')
    crf = train_crf(X_train, y_train)
    basic_logging('training ends')
    y_pred = crf.predict(X_test)
    result = evaluate_ner_result([i for j in y_pred for i in j], [i for j in y_test for i in j])

    # overall_f1, details = test_crf_prediction(crf, y_pred, y_test, report_type)
    # details['overall_f1'] = overall_f1
    # result.to_csv(result_f, index=False)
    basic_logging('testing ends')
    if model_f:
        jl.dump(crf, model_f)
    return crf, details


def pipeline_best_predict(train_f, test_f, model_f, result_f, feature_conf, hdf_f, hdf_key, report_type, cv, iteration,
                          window_size):
    """
    A pipeline for CRF training
    :param train_f: train dataset in a 3-column csv (TOKEN, POS, LABEL)
    :param test_f: test dataset in a 3-column csv (TOKEN, POS, LABEL)
    :param model_f: model file
    :param feature_conf: feature configurations
    :param hdf_f: feature HDF5 file
    :param hdf_key: keys of feature HDF5 file
    :param report_type: 'spc' for a specific report and 'bin' for binary report
    :param cv: cv scale
    :param iteration: iteration time
    """
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
    best_f1, best_details = test_crf_prediction(best_predictor, y_pred, y_test, report_type)
    basic_logging('Testing ends')
    best_details['overall_f1'] = best_f1
    best_details.to_csv(result_f, index=False)
    if model_f:
        jl.dump(best_predictor, model_f)
    return crf, best_predictor, rs_cv, best_details


def pipeline_validate(validate_f, model_f, feature_conf, hdf_f, result_f, hdf_key, report_type, window_size):
    """
    A pipeline for CRF training
    :param validate_f: test dataset in a 3-column csv (TOKEN, POS, LABEL)
    :param model_f: model file
    :param feature_conf: feature configurations
    :param hdf_f: feature HDF5 file
    :param hdf_key: keys of feature HDF5 file
    :param report_type: 'spc' for a specific report and 'bin' for binary report
    """
    basic_logging('loading conf begins')
    crf, f_dics = batch_loading(model_f, hdf_f, hdf_key)
    basic_logging('loading conf ends')
    test_df = process_annotated(validate_f)
    test_df = batch_add_features(test_df, f_dics)
    basic_logging('adding features ends')
    test_sents = df2crfsuite(test_df)
    basic_logging('converting to crfsuite ends')
    X_test, y_test = feed_crf_trainer(test_sents, feature_conf, hdf_key, window_size)
    basic_logging('Conversion ends')

    y_pred = crf.predict(X_test)
    result = evaluate_ner_result([i for j in y_pred for i in j], [i for j in y_test for i in j])

    overall_f1, details = test_crf_prediction(crf, y_pred, y_test, report_type)
    details['overall_f1'] = overall_f1
    details.to_csv(result_f, index=False)
    return result, details


##############################################################################


RESULT_F = '_'.join((RESULT_F, get_now(), '.csv'))


def main(argv):
    print(argv)
    print()
    dic = {
        'train': lambda: pipeline_train(train_f=TRAIN_F, test_f=TEST_F, model_f=MODEL_F,
                                        result_f=RESULT_F, hdf_f=HDF_F, hdf_key=HDF_KEY,
                                        feature_conf=FEATURE_FUNCTION,
                                        report_type=REPORT_TYPE,
                                        window_size=WINDOW_SIZE),
        'cv': lambda: pipeline_best_predict(train_f=TRAIN_F, test_f=TEST_F,
                                            model_f=MODEL_F,
                                            result_f=RESULT_F, hdf_f=HDF_F,
                                            hdf_key=HDF_KEY,
                                            feature_conf=FEATURE_FUNCTION,
                                            report_type=REPORT_TYPE,
                                            window_size=WINDOW_SIZE, cv=CV,
                                            iteration=ITERATION),
        'validate': lambda: pipeline_validate(validate_f=VALIDATE_F, model_f=MODEL_F,
                                              result_f=RESULT_F, hdf_f=HDF_F,
                                              hdf_key=HDF_KEY,
                                              feature_conf=FEATURE_FUNCTION,
                                              report_type=REPORT_TYPE,
                                              window_size=WINDOW_SIZE)
    }
    dic[argv]()
