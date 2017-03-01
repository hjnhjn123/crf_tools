# -*- coding: utf-8 -*-

from json import dumps

import joblib as jl
from os import listdir, path

from .arsenal_crf import *
from .arsenal_spacy import *
from .arsenal_stats import *


def prepare_feature_dict(city_f, com_single_f, com_suffix_f, country_f, name_f, tfdf_f, tfidf_f):
    name, country = line_file2set(name_f), line_file2set(country_f)
    city, com_single = line_file2set(city_f), line_file2set(com_single_f)
    com_suffix = {i.title() for i in line_file2set(com_suffix_f)}
    tfidf = prepare_features_dict(tfidf_f)
    tfdf = prepare_features_dict(tfdf_f)
    return tfdf, tfidf, city, com_single, com_suffix, country, name


def batch_add_features(pos_data, tfdf, tfidf, city, com_single, com_suffix, country, name):
    name_added = (add_one_features_list(chunk, name) for chunk in pos_data)
    com_suffix_added = (add_one_features_list(chunk, com_suffix) for chunk in name_added)
    country_added = (add_one_features_list(chunk, country) for chunk in com_suffix_added)
    city_added = (add_one_features_list(chunk, city) for chunk in country_added)
    com_single_added = (add_one_features_list(chunk, com_single) for chunk in city_added)
    tfidf_added = (add_one_feature_dict(chunk, tfidf) for chunk in com_single_added)
    result = [add_one_feature_dict(chunk, tfdf) for chunk in tfidf_added]
    return result


def crf_result2list(crf_re):
    text_list, ner_list = [i[0] for i in crf_re], [i[2] for i in crf_re]
    ner_candidate = [(token, ner) for token, _, ner in crf_re if ner[0] != 'O']
    ner_index = [i for i in range(len(ner_candidate)) if ner_candidate[i][1][0] == 'U' or ner_candidate[i][1][0] == 'L']
    new_index = [a + b for a, b in enumerate(ner_index)]
    for i in new_index:
        ner_candidate[i + 1:i + 1] = [(' ##split ', '##split')]
    ner_result = list(set(' '.join([i[0].strip() for i in ner_candidate]).split(' ##split ')))
    return text_list, ner_list, ner_result


##############################################################################


# Pipelines


def pipeline_crf_train(train_f, test_f, model_f, conf_f, tfdf, tfidf, city, com_single, com_suffix, country, name):
    train_data, test_data = process_annotated(train_f), process_annotated(test_f)
    train_sents = batch_add_features(train_data, tfdf, tfidf, city, com_single, com_suffix, country, name)
    test_sents = batch_add_features(test_data, tfdf, tfidf, city, com_single, com_suffix, country, name)
    print(get_now(), 'converted')

    X_train, y_train = feed_crf_trainer(train_sents, conf_f)
    X_test, y_test = feed_crf_trainer(test_sents, conf_f)
    print(get_now(), 'feed')
    crf = train_crf(X_train, y_train)
    print(get_now(), 'train')
    result, details = test_crf_prediction(crf, X_test, y_test)
    print(get_now(), 'predict')
    jl.dump(crf, model_f)
    return crf, result, details


def pipeline_crf_cv(train_f, test_f, conf_f, name_f, tfdf, tfidf, city, com_single, com_suffix, country, name, cv,
                    iteration):
    train_data, test_data = process_annotated(train_f), process_annotated(test_f)
    train_sents = batch_add_features(train_data, tfdf, tfidf, city, com_single, com_suffix, country, name)
    test_sents = batch_add_features(test_data, tfdf, tfidf, city, com_single, com_suffix, country, name)
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


def pipeline_train_best_predict(train_f, test_f, conf_f, tfdf, tfidf, city, com_single, com_suffix, country, name, cv,
                                iteration):
    train_data, test_data = process_annotated(train_f), process_annotated(test_f)
    train_sents = batch_add_features(train_data, tfdf, tfidf, city, com_single, com_suffix, country, name)
    test_sents = batch_add_features(test_data, tfdf, tfidf, city, com_single, com_suffix, country, name)
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


def pipeline_pos_crf(in_file, out_f, train_f, conf_f, tfdf, tfidf, city, com_single, com_suffix, country, name, cols,
                     pieces=10):
    data = json2pd(in_file, cols, lines=True)
    data = data.drop_duplicates()
    # data = random_rows(data, pieces, 'content')
    data = data.dropna()
    parsed_data = spacy_batch_processing(data, ['chk'], '', 'content', ['content'])
    parsed_data = chain.from_iterable(parsed_data)
    pos_data = [list(x[1])[:-1] for x in groupby(parsed_data, lambda x: x == ('##END', '###', 'O')) if not x[0]]

    train_data = process_annotated(train_f)

    train_sents = batch_add_features(train_data, tfdf, tfidf, city, com_single, com_suffix, country, name)
    test_sents = batch_add_features(pos_data, tfdf, tfidf, city, com_single, com_suffix, country, name)
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


def pipeline_crf_predict(model_f, test_f, conf_f, tfdf, tfidf, city, com_single, com_suffix, country, name, out_f):
    test_data = process_annotated(test_f)
    test_sents = batch_add_features(test_data, tfdf, tfidf, city, com_single, com_suffix, country, name)
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


def streaming_pos_crf(in_f, crf, conf, tfdf, tfidf, city, com_single, com_suffix, country, name,
                      cols=['url', 'content']):
    data = json2pd(in_f, cols, lines=True).dropna()
    url = data['url'].to_string(index=False)
    taskid = hashit(url)
    parsed_data = chain.from_iterable(spacy_batch_processing(data, ['chk'], '', 'content', ['content']))
    pos_data = [list(x[1])[:-1] for x in groupby(parsed_data, lambda x: x == ('##END', '###', 'O')) if not x[0]]
    test_sents = batch_add_features(pos_data, tfdf, tfidf, city, com_single, com_suffix, country, name)

    X_test, y_test = feed_crf_trainer(test_sents, conf)
    crf_result = crf_predict(crf, pos_data, X_test)
    text_list, ner_complete, ner_phrase = crf_result2list(crf_result)

    result = defaultdict()
    result['url'] = url
    result['taskid'] = taskid
    # result['text'] = text_list
    # result['ner_complete'] = list(zip(text_list, ner_complete))
    result['ner_phrase'] = ner_phrase

    json_result = dumps(result)

    # out = open(out_f, 'w')
    # out.write(json_result)
    # out.flush(), out.close()
    return taskid, json_result


def pipeline_loading(conf_f, crf_f, city_f, com_single_f, com_suffix_f, country_f, name_f, tfdf_f, tfidf_f):
    conf, crf = load_yaml_conf(conf_f), jl.load(crf_f)
    tfdf, tfidf, city, com_single, com_suffix, country, name = prepare_feature_dict(city_f, com_single_f, com_suffix_f,
                                                                                    country_f, name_f, tfdf_f, tfidf_f)
    return conf, crf, tfdf, tfidf, city, com_single, com_suffix, country, name


def pipeline_streaming(in_folder, out_folder, conf_f, crf_f, city_f, com_single_f, com_suffix_f, country_f, name_f,
                       tfdf_f, tfidf_f):
    conf, crf, tfdf, tfidf, city, com_single, com_suffix, country, name = pipeline_loading(conf_f, crf_f, city_f,
                                                                                           com_single_f, com_suffix_f,
                                                                                           country_f, name_f, tfdf_f,
                                                                                           tfidf_f)
    for in_f in listdir(in_folder):
        ff = path.join(in_folder, in_f)
        taskid, json_result = streaming_pos_crf(ff, crf, conf, tfdf, tfidf, city, com_single, com_suffix, country, name,
                          cols=['url', 'content'])
        with open(path.join(out_folder, taskid+'.json'), 'w') as out:
            out.write(json_result)