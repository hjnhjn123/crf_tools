# -*- coding: utf-8 -*-

from math import modf
from os import listdir, path

import joblib as jl
import redis

from .arsenal_crf import *
from .arsenal_spacy import *
from .arsenal_stats import *


def prepare_feature_dict(city_f, com_single_f, com_suffix_f, country_f, name_f, tfdf_f, tfidf_f):
    name, country = line_file2set(name_f), line_file2set(country_f)
    city, com_single = line_file2set(city_f), line_file2set(com_single_f)
    com_suffix = {i.title() for i in line_file2set(com_suffix_f)}
    tfidf, tfdf = line_file2dict(tfidf_f), line_file2dict(tfdf_f)
    return tfdf, tfidf, city, com_single, com_suffix, country, name


def batch_add_features(pos_data, tfdf, tfidf, city, com_single, com_suffix, country, name):
    added_name = (add_one_features_list(chunk, name) for chunk in pos_data)
    added_city = (add_one_features_list(chunk, city) for chunk in added_name)
    added_country = (add_one_features_list(chunk, country) for chunk in added_city)
    added_com_suffix = (add_one_features_list(chunk, com_suffix) for chunk in added_country)
    added_com_single = (add_one_features_list(chunk, com_single) for chunk in added_com_suffix)
    added_tfidf = (add_one_feature_dict(chunk, tfidf) for chunk in added_com_single)
    result = [add_one_feature_dict(chunk, tfdf) for chunk in added_tfidf]
    return result


def crf_result2list(crf_result):
    text_list, ner_list = [i[0] for i in crf_result], [i[2] for i in crf_result]
    ner_candidate = [(token, ner) for token, _, ner in crf_result if ner[0] != 'O']
    # Remove non NER words
    ner_index = (i for i in range(len(ner_candidate)) if ner_candidate[i][1][0] == 'U' or ner_candidate[i][1][0] == 'L')
    # Fetch the index of the ending of an NER
    new_index = (a + b for a, b in enumerate(ner_index))
    # Generate a new index
    for i in new_index:
        ner_candidate[i + 1:i + 1] = [('##split', '##split')]
    # Add the split to each NER phrases
    ner_result = (' '.join([i[0].strip() for i in ner_candidate]).split(' ##split'))
    # Split each NER phrases
    ner_result = list(set(i.strip() for i in ner_result if i))
    # Clean up
    return text_list, ner_list, ner_result


def batch_loading(conf_f, crf_f, city_f, com_single_f, com_suffix_f, country_f, name_f, tfdf_f, tfidf_f, swtich):
    conf = load_yaml_conf(conf_f)
    crf = jl.load(crf_f) if swtich == 'test' else None
    features = prepare_feature_dict(city_f, com_single_f, com_suffix_f, country_f, name_f, tfdf_f, tfidf_f)
    tfdf, tfidf, city, com_single, com_suffix, country, name = features
    return conf, crf, tfdf, tfidf, city, com_single, com_suffix, country, name


##############################################################################


# Streaming


def streaming_pos_crf(in_f, crf, conf, tfdf, tfidf, city, com_single, com_suffix, country, name):
    raw_df = pd.read_json(in_f, lines=True)
    raw_df['content'] = raw_df.result.to_dict()[0]['content']

    parsed_data = chain.from_iterable(spacy_batch_processing(raw_df, '', 'content', ['content']))
    prepared_data = [list(x[1])[:-1] for x in groupby(parsed_data, lambda x: x == ('##END', '###', 'O')) if not x[0]]
    test_sents = batch_add_features(prepared_data, tfdf, tfidf, city, com_single, com_suffix, country, name)

    X_test, y_test = feed_crf_trainer(test_sents, conf)
    crf_result = crf_predict(crf, prepared_data, X_test)
    text_list, ner_complete, ner_phrase = crf_result2list(crf_result)

    raw_df.result.to_dict()[0]['ner_phrase'] = ner_phrase
    raw_df = raw_df.drop(['content'], axis=1)
    json_result = raw_df.to_json(orient='records', lines=True)

    return json_result


##############################################################################


# Pipelines


def pipeline_crf_train(train_f, test_f, model_f, dict_conf, tfdf_f, tfidf_f, city_f, com_single_f, com_suffix_f,
                       country_f, name_f):
    train_data, test_data = process_annotated(train_f), process_annotated(test_f)
    loads = batch_loading(dict_conf, '', city_f, com_single_f, com_suffix_f, country_f, name_f, tfdf_f, tfidf_f,
                          'train')
    conf, crf, tfdf, tfidf, city, com_single, com_suffix, country, name = loads
    train_sents = batch_add_features(train_data, tfdf, tfidf, city, com_single, com_suffix, country, name)
    test_sents = batch_add_features(test_data, tfdf, tfidf, city, com_single, com_suffix, country, name)
    print(get_now(), 'converted')
    X_train, y_train = feed_crf_trainer(train_sents, conf)
    X_test, y_test = feed_crf_trainer(test_sents, conf)
    print(get_now(), 'feed')
    crf = train_crf(X_train, y_train)
    print(get_now(), 'train')
    result, details = test_crf_prediction(crf, X_test, y_test)
    print(get_now(), 'predict')
    jl.dump(crf, model_f)
    return crf, result, details


def pipeline_train_best_predict(train_f, test_f, model_f, dict_conf, tfdf_f, tfidf_f, city_f, com_single_f,
                                com_suffix_f, country_f, name_f, cv, iteration):
    train_data, test_data = process_annotated(train_f), process_annotated(test_f)
    loads = batch_loading(dict_conf, '', city_f, com_single_f, com_suffix_f, country_f, name_f, tfdf_f, tfidf_f,
                          'train')
    conf, crf, tfdf, tfidf, city, com_single, com_suffix, country, name = loads
    train_sents = batch_add_features(train_data, tfdf, tfidf, city, com_single, com_suffix, country, name)
    test_sents = batch_add_features(test_data, tfdf, tfidf, city, com_single, com_suffix, country, name)
    print(get_now(), 'converted')
    X_train, y_train = feed_crf_trainer(train_sents, conf)
    X_test, y_test = feed_crf_trainer(test_sents, conf)
    crf = train_crf(X_train, y_train)
    labels = show_crf_label(crf)
    params_space = make_param_space()
    f1_scorer = make_f1_scorer(labels)
    rs_cv = search_param(X_train, y_train, crf, params_space, f1_scorer, cv, iteration)
    print(get_now(), 'predict')
    best_predictor = rs_cv.best_estimator_
    best_result, best_details = test_crf_prediction(best_predictor, X_test, y_test)
    jl.dump(best_predictor, model_f)
    return crf, best_predictor, rs_cv, best_result, best_details


def pipeline_pos_crf(in_file, out_f, crf_f, dict_conf, city_f, com_single_f, com_suffix_f, country_f, name_f, tfdf_f,
                     tfidf_f, switch, cols, pieces=10):
    loads = batch_loading(dict_conf, crf_f, city_f, com_single_f, com_suffix_f, country_f, name_f, tfdf_f, tfidf_f,
                          switch)
    conf, crf, tfdf, tfidf, city, com_single, com_suffix, country, name = loads
    data = json2pd(in_file, cols, lines=True)
    data = data.drop_duplicates()
    data = random_rows(data, pieces, 'content')
    data = data.dropna()
    parsed_data = spacy_batch_processing(data, '', 'content', ['content'])
    parsed_data = chain.from_iterable(parsed_data)
    pos_data = [list(x[1])[:-1] for x in groupby(parsed_data, lambda x: x == ('##END', '###', 'O')) if not x[0]]
    test_sents = batch_add_features(pos_data, tfdf, tfidf, city, com_single, com_suffix, country, name)
    X_test, y_test = feed_crf_trainer(test_sents, conf)
    print(get_now(), 'feed')
    result = crf_predict(crf, pos_data, X_test)
    print(get_now(), 'predict')
    out = pd.DataFrame(result)
    out.to_csv(out_f, header=False, index=False)
    return crf, result


def pipeline_crf_test(test_f, dict_conf, crf_f, city_f, com_single_f, com_suffix_f, country_f, name_f, tfdf_f,
                      tfidf_f, switch):
    test_data = process_annotated(test_f)
    loads = batch_loading(dict_conf, crf_f, city_f, com_single_f, com_suffix_f, country_f, name_f, tfdf_f, tfidf_f,
                          switch)
    conf, crf, tfdf, tfidf, city, com_single, com_suffix, country, name = loads
    test_sents = batch_add_features(test_data, tfdf, tfidf, city, com_single, com_suffix, country, name)
    print(get_now(), 'converted')
    X_test, y_test = feed_crf_trainer(test_sents, conf)
    print(get_now(), 'feed')
    result, details = test_crf_prediction(crf, X_test, y_test)
    return result, details


def pipeline_streaming_folder(in_folder, out_folder, dict_conf, crf_f, city_f, com_single_f, com_suffix_f, country_f,
                              name_f, tfdf_f, tfidf_f, switch):
    loads = batch_loading(dict_conf, crf_f, city_f, com_single_f, com_suffix_f, country_f, name_f, tfdf_f, tfidf_f,
                          switch)
    conf, crf, tfdf, tfidf, city, com_single, com_suffix, country, name = loads
    for in_f in listdir(in_folder):
        ff = path.join(in_folder, in_f)
        json_result = streaming_pos_crf(ff, crf, conf, tfdf, tfidf, city, com_single, com_suffix, country, name)
        # with open(path.join(out_folder, taskid + '.json'), 'w') as out:
        #     out.write(json_result)


def pipeline_streaming_queue(redis_conf, dict_conf, crf_f, city_f, com_single_f, com_suffix_f, country_f, name_f,
                             tfdf_f, tfidf_f, switch):
    loads = batch_loading(dict_conf, crf_f, city_f, com_single_f, com_suffix_f, country_f, name_f, tfdf_f, tfidf_f,
                          switch)
    conf, crf, tfdf, tfidf, city, com_single, com_suffix, country, name = loads
    r_address, r_port, r_db, r_key = OrderedDict(load_yaml_conf(redis_conf)['test_read']).values()
    w_address, w_port, w_db, w_key = OrderedDict(load_yaml_conf(redis_conf)['test_write']).values()

    r = redis.StrictRedis(host=r_address, port=r_port, db=r_db)
    w = redis.StrictRedis(host=w_address, port=w_port, db=w_db)

    i = 0

    while True:
        queue = r.lpop(r_key).decode('utf-8')
        json_result = streaming_pos_crf(queue, crf, conf, tfdf, tfidf, city, com_single, com_suffix, country, name)
        w.lpush(w_key, json_result)
        i += 1
        if modf(i / 10)[0] == 0.0:
            print(get_now(), i)
