# -*- coding: utf-8 -*-

from .arsenal_crf import *

# Pipelines


def pipeline_crf_train(train_f, test_f, conf_f, name_f, com_suffix_f, country_f, city_f, com_single_f, tfidf_f, tfdf_f):
    train_data, test_data = process_annotated(train_f), process_annotated(test_f)
    train_sents = batch_add_features(train_data, name_f, com_suffix_f, country_f, city_f, com_single_f, tfidf_f, tfdf_f)
    test_sents = batch_add_features(test_data, name_f, com_suffix_f, country_f, city_f, com_single_f, tfidf_f, tfdf_f)
    print(get_now(), 'converted')

    X_train, y_train = feed_crf_trainer(train_sents, conf_f)
    X_test, y_test = feed_crf_trainer(test_sents, conf_f)
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

    X_train, y_train = feed_crf_trainer(train_sents, conf_f)
    X_test, y_test = feed_crf_trainer(test_sents, conf_f)
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

    X_train, y_train = feed_crf_trainer(train_sents, conf_f)
    X_test, y_test = feed_crf_trainer(test_sents, conf_f)
    crf = train_crf(X_train, y_train)
    labels = show_crf_label(crf)
    params_space = make_param_space()
    f1_scorer = make_f1_scorer(labels)
    rs_cv = search_param(X_train, y_train, crf, params_space, f1_scorer, cv, iteration)
    print(get_now(), 'predict')
    best_predictor = rs_cv.best_estimator_
    best_result, best_details = test_crf_prediction(best_predictor, X_test, y_test)
    return crf, best_predictor, rs_cv, best_result, best_details


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
    X_train, y_train = feed_crf_trainer(train_sents, conf_f)
    X_test, y_test = feed_crf_trainer(test_sents, conf_f)
    print(get_now(), 'feed')
    crf = train_crf(X_train, y_train)
    print(get_now(), 'train')
    result = crf_predict(crf, pos_data, X_test)
    print(get_now(), 'predict')
    out = pd.DataFrame(result)
    out.to_csv(out_f, header=False, index=False)
    return crf, result


def pipeline_crf_predict(model_f, test_f, conf_f, name_f, com_suffix_f, country_f, city_f, com_single_f, tfidf_f,
                         tfdf_f, out_f):
    test_data = process_annotated(test_f)
    test_sents = batch_add_features(test_data, name_f, com_suffix_f, country_f, city_f, com_single_f, tfidf_f, tfdf_f)

    print(get_now(), 'converted')
    X_test, y_test = feed_crf_trainer(test_sents, conf_f)
    print(get_now(), 'feed')
    crf = sklearn_crfsuite.CRF(model_filename=model_f)
    print(get_now(), 'train')
    result = crf_predict(crf, test_data, X_test)
    print(get_now(), 'predict')
    out = pd.DataFrame(result)
    out.to_csv(out_f, header=False, index=False)
    return result

##############################################################################
