# -*- coding: utf-8 -*-

from collections import Counter
from copy import deepcopy
from math import modf
from os import listdir, path

import joblib as jl
import redis

from .arsenal_crf import *
from .arsenal_spacy import *
from .arsenal_stats import *

STOP_ROOTS = {'is', 'are', 'was', 'were', 'been', 'be'}


def prepare_feature_dict(city_f, com_single_f, com_suffix_f, country_f, name_f, tfdf_f, tfidf_f):
    name, country = line_file2set(name_f), line_file2set(country_f)
    city, com_single = line_file2set(city_f), line_file2set(com_single_f)
    com_suffix = {i.title() for i in line_file2set(com_suffix_f)}
    tfidf, tfdf = line_file2dict(tfidf_f), line_file2dict(tfdf_f)
    return tfdf, tfidf, city, com_single, com_suffix, country, name


def batch_add_features(pos_data, tfdf, tfidf, city, com_single, com_suffix, country, name):
    added_name = (add_feature_str(chunk, name) for chunk in pos_data)
    added_city = (add_feature_str(chunk, city) for chunk in added_name)
    added_country = (add_feature_str(chunk, country) for chunk in added_city)
    added_com_suffix = (add_feature_str(chunk, com_suffix) for chunk in added_country)
    added_com_single = (add_feature_str(chunk, com_single) for chunk in added_com_suffix)
    added_tfidf = (add_one_feature_dict(chunk, tfidf) for chunk in added_com_single)
    result = [add_one_feature_dict(chunk, tfdf) for chunk in added_tfidf]
    return result


def crf_result2dict(crf_result):
    ner_candidate = [(token, ner) for token, _, ner in crf_result if ner[0] != 'O']
    ner_index = (i for i in range(len(ner_candidate)) if ner_candidate[i][1][0] == 'U' or ner_candidate[i][1][0] == 'L')
    new_index = (a + b for a, b in enumerate(ner_index))
    ner_result = extract_ner_result(ner_candidate, new_index)
    return ner_result


def extract_ner_result(ner_candidate, new_index):
    new_candidate = deepcopy(ner_candidate)
    for i in new_index:
        new_candidate[i + 1:i + 1] = [('##split', '##split')]
    ner_result = (' '.join([(i[0].strip() + '##' + i[1].strip()) for i in new_candidate if i[1]]).split('##split'))
    ner_result = ([i.strip(' ') for i in ner_result if i and i != '##'])
    # ner_result = ((' '.join([i.split('##')[0] for i in tt.split()]), tt[-3:]) for tt in ner_result)
    ner_result = ('##'.join((' '.join([i.split('##')[0] for i in tt.split()]), tt[-3:])) for tt in ner_result)
    ner_result = sort_dic(Counter(i for i in ner_result if i), sort_key=1, rev=True)
    return ner_result


def crf_dep_result2dict(crf_result, dep_data, tfidf):
    ner_candidate = [(token, ner, index) for index, (token, _, ner) in enumerate(crf_result) if
                     ner != 'O' and not ner.endswith('DAT') and not ner.endswith('MON')]
    ner_index = [i for i in range(len(ner_candidate)) if ner_candidate[i][1][0] == 'U' or ner_candidate[i][1][0] == 'L']
    new_index = (a + b for a, b in enumerate(ner_index))
    ner_result = extract_ner_result(ner_candidate, new_index)
    root_result = extract_dep_result(dep_data, ner_candidate, tfidf)
    return ner_result, root_result


def extract_dep_result(dep_data, ner_candidate, tfidf):
    dep_lower = [(m[0].lower(), m[1]) for m in dep_data]
    dep_tfidf = add_one_feature_dict(dep_lower, tfidf)
    sent_index = [i for i in range(len(dep_data)) if dep_data[i][1] == '###']
    # sent_boundaries = [(0, sent_index[0])] + [(sent_index[i], sent_index[i + 1]) for i in range(len(sent_index) - 1)]
    new_sen_index = (a + b for a, b in enumerate(sent_index))
    rel_candidate = deepcopy(ner_candidate)
    for i in new_sen_index:
        for j in range(len(ner_candidate) - 2):
            if not str(ner_candidate[j][2]).startswith('-1') and not str(ner_candidate[j + 2]).startswith('-1'):
                if int(ner_candidate[j][2]) < i < int(ner_candidate[j + 2][2]):
                    rel_candidate[j + 1: j + 1] = [('##Sent', '##Sent', -1)]
    root_tuple = [(i[1][0], i[1][2], i[0]) for i in enumerate(dep_tfidf) if
                  i[1][1] == 'ROOT' and i[1][2] != '0' and i[1][0] not in STOP_ROOTS]
    root_result = Counter(i[0] for i in root_tuple)
    return root_result


def batch_loading(dict_conf, crf_f, city_f, com_single_f, com_suffix_f, country_f, name_f, tfdf_f, tfidf_f, swtich):
    conf = load_yaml_conf(dict_conf)
    crf = jl.load(crf_f) if swtich == 'test' else None
    features = prepare_feature_dict(city_f, com_single_f, com_suffix_f, country_f, name_f, tfdf_f, tfidf_f)
    tfdf, tfidf, city, com_single, com_suffix, country, name = features
    return conf, crf, tfdf, tfidf, city, com_single, com_suffix, country, name


##############################################################################


# Streaming


def streaming_pos_crf(in_f, crf, conf, tfdf, tfidf, city, com_single, com_suffix, country, name):
    raw_df = pd.read_json(in_f, lines=True)
    raw_df['content'] = raw_df.result.to_dict()[0]['content']

    parsed_data = chain.from_iterable(spacy_batch_processing(raw_df, '', 'content', ['content'], 'crf'))
    prepared_data = [list(x[1]) for x in groupby(parsed_data, lambda x: x == ('##END', '###', 'O')) if not x[0]]
    test_sents = batch_add_features(prepared_data, tfdf, tfidf, city, com_single, com_suffix, country, name)

    X_test, y_test = feed_crf_trainer(test_sents, conf)
    crf_result = crf_predict(crf, prepared_data, X_test)
    ner_phrase = crf_result2dict(crf_result)

    raw_df.result.to_dict()[0]['ner_phrase'] = ner_phrase
    raw_df = raw_df.drop(['content'], axis=1)
    json_result = raw_df.to_json(orient='records', lines=True)

    return json_result


def streaming_pos_dep_crf(in_f, crf, conf, tfdf, tfidf, city, com_single, com_suffix, country, name):
    raw_df = pd.read_json(in_f, lines=True)
    raw_df['content'] = raw_df.result.to_dict()[0]['content']

    # parsed_data = chain.from_iterable(spacy_batch_processing(raw_df, '', 'content', ['content'], 'dep'))
    parsed_data = list(chain.from_iterable(spacy_batch_processing(raw_df, '', 'content', ['content'], 'crf')))

    prepared_data = [list(x[1]) for x in groupby(parsed_data, lambda x: x == ('##END', '###', 'O', 'O')) if not x[0]]
    test_sents = batch_add_features(prepared_data, tfdf, tfidf, city, com_single, com_suffix, country, name)

    X_test, y_test = feed_crf_trainer(test_sents, conf)
    crf_result = crf_predict(crf, prepared_data, X_test)
    dep_data = list(chain.from_iterable(spacy_batch_processing(raw_df, '', 'content', ['content'], 'dep')))

    ner_phrases, root_result = crf_dep_result2dict(crf_result, dep_data, tfidf)

    # raw_df.result.to_dict()[0]['ner_phrases'] = ner_phrases
    # raw_df = raw_df.drop(['content'], axis=1)
    # json_result = raw_df.to_json(orient='records', lines=True)

    return root_result


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


def pipeline_crf_train_evt(train_f, test_f, model_f, dict_conf, tfdf_f, tfidf_f, city_f, com_single_f, com_suffix_f,
                           country_f, name_f):
    train_data, test_data = process_annotated(train_f), process_annotated(test_f)
    loads = batch_loading(dict_conf, '', city_f, com_single_f, com_suffix_f, country_f, name_f, tfdf_f, tfidf_f,
                          'train')
    conf, crf, tfdf, tfidf, city, com_single, com_suffix, country, name = loads
    train_sents = batch_add_features(train_data, tfdf, tfidf, city, com_single, com_suffix, country, name)
    test_sents = batch_add_features(test_data, tfdf, tfidf, city, com_single, com_suffix, country, name)
    print(get_now(), 'converted')
    X_train, y_train = feed_crf_trainer_spfc(train_sents, conf, 'COM')
    X_test, y_test = feed_crf_trainer_spfc(test_sents, conf, 'COM')
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
    parsed_data = spacy_batch_processing(data, '', 'content', ['content'], 'crf')
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
    i = 0
    root_dic = defaultdict()
    for in_f in listdir(in_folder):
        ff = path.join(in_folder, in_f)
        json_result = streaming_pos_crf(ff, crf, conf, tfdf, tfidf, city, com_single, com_suffix, country, name)
        i += 1
        if modf(i / 100)[0] == 0.0:
            print(get_now(), i)
        with open(path.join(out_folder, str(in_f) + '.json'), 'w') as out:
            out.write(json_result)
        # root_result = streaming_pos_dep_crf(ff, crf, conf, tfdf, tfidf, city, com_single, com_suffix, country, name)
        # root_dic.update(root_result)
        i += 1
        if modf(i / 100)[0] == 0.0:
            print(get_now(), i)
    result = pd.DataFrame.from_dict(root_dic, orient='index').reset_index()
    result.columns = ['Token', 'Freq']
    result = result.sort_values(by='Freq', ascending=False)
    result.to_csv(out_folder, header=None, index=False)


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
