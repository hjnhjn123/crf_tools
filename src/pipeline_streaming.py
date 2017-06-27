# -*- coding: utf-8 -*-

from itertools import chain
from collections import defaultdict, Counter

import pandas as pd
import joblib as jl

from .arsenal_boto import sqs_get_msgs, sqs_send_msg, s3_get_file
from .arsenal_crf import batch_add_features, batch_loading, feed_crf_trainer, crf_predict, crf_result2json, df2crfsuite
from .arsenal_logging import basic_logging
from .arsenal_spacy import spacy_batch_processing
from .arsenal_stats import sort_dic
from .settings import *


##############################################################################


def streaming_pos_crf(in_f, crf, f_dics, feature_conf, hdf_key, window_size, col):
    raw_df = pd.read_json(in_f, lines=True)
    raw_df['content'] = raw_df[col].to_dict()[0]['content']
    parsed_data = chain.from_iterable(spacy_batch_processing(raw_df, '', 'content', ['content'], 'crf'))
    prepared_data = pd.DataFrame(list(parsed_data))
    test_df = batch_add_features(prepared_data, f_dics)
    test_sents = df2crfsuite(test_df)
    X_test, y_test = feed_crf_trainer(test_sents, feature_conf, hdf_key, window_size)
    crf_result = crf_predict(crf, test_sents, X_test)
    return crf_result, raw_df


def streaming_pos_crf_multi(in_f, f_dics, feature_conf, hdf_key, window_size, col, model_dics):
    raw_df = pd.read_json(in_f, lines=True)
    raw_df['content'] = raw_df[col].to_dict()[0]['content']
    parsed_data = chain.from_iterable(spacy_batch_processing(raw_df, '', 'content', ['content'], 'crf'))
    prepared_data = pd.DataFrame(list(parsed_data))
    test_df = batch_add_features(prepared_data, f_dics)
    test_sents = df2crfsuite(test_df)
    X_test, y_test = feed_crf_trainer(test_sents, feature_conf, hdf_key, window_size)
    crf_results = {name: crf_predict(model, test_sents, X_test) for name, model in model_dics.items()}
    return crf_results, raw_df


def voting(crf_results):
    crf_dfs = [pd.DataFrame(crf_list, columns=HEADER).add_suffix('_'+name) for name, crf_list in crf_results.items()]
    combined = pd.concat(crf_dfs, axis=1)
    cols = [i for i in combined.columns if i.startswith('NER')]
    # to_vote = combined[cols].apply(tuple, axis=1).tolist()  # convert a df to zipped list
    to_vote = sort_dic({col.split('_')[1]: combined[col].tolist() for col in cols})
    to_vote = sort_dic(to_vote)
    if len(cols) == 2:
        vote_result = merge_list_dic(to_vote)
    elif len(cols) > 2:
       specific = {name: lst for name, lst in to_vote.items() if name != 'NER_GEN'}
       pass # todo fix more than three models
    return list(zip(combined.iloc[:,0].tolist(), vote_result, combined.iloc[:,2].tolist(), ))


def merge_list_dic(list_dict):
    l1, l2 = list_dict.values()
    name1, name2 = list_dict.keys()
    return [l1[i] if l1[i].endswith(name1) else l2[i] for i in range(len(l1))]
    

def load_multi_models(model_fs):
    model_dics = {model.split('.')[0].split('_')[-1]: jl.load(model) for model in model_fs}
    return model_dics

##############################################################################


def pipeline_streaming_sqs(in_queue, out_queue, model_f, hdf_f, hdf_key, feature_conf, window_size, col):
    sqs_queues = sqs_get_msgs(in_queue)
    model = jl.load(model_f)
    f_dics = batch_loading(hdf_f, hdf_key)

    while True:
        for q in sqs_queues.receive_messages():
            json_input = q.body
            crf_result, raw_df = streaming_pos_crf(json_input, model, f_dics, feature_conf, hdf_key, window_size, col)
            json_result = crf_result2json(crf_result, raw_df, col)
            sqs_send_msg(json_result, queue_name=out_queue)
            basic_logging('Queue output')
            q.delete()


def pipeline_multi_streaming_sqs(in_queue, out_queue, hdf_f, hdf_key, feature_conf, window_size, col, *model_fs):
    sqs_queues = sqs_get_msgs(in_queue)
    # models = [jl.load(model) for model in model_fs]
    model_dics = load_multi_models(model_fs)
    f_dics = batch_loading(hdf_f, hdf_key)

    while True:
        for q in sqs_queues.receive_messages():
            json_input = q.body
            
            crf_results, raw_df = streaming_pos_crf_multi(json_input, f_dics, feature_conf, hdf_key, window_size,col, model_dics)
            final_result = voting(crf_results)
            json_result = crf_result2json(final_result, raw_df, col)

            sqs_send_msg(json_result, queue_name=out_queue)
            basic_logging('Queue output')
            q.delete()


##############################################################################

def main():
    s3_get_file(S3_BUCKET, MODEL_KEY, MODEL_FILE)
    s3_get_file(S3_BUCKET, HDF_FILE_KEY, HDF_FILE)
    basic_logging('Queue prepared')
    pipeline_streaming_sqs(IN_QUEUE, OUT_QUQUE, HDF_FILE, HDF_KEY, FEATURE_CONF, WINDOW_SIZE,
                               CONTENT_COL, MODEL_FS)

    # pipeline_streaming_sqs(IN_QUEUE, OUT_QUQUE, MODEL_FILE, HDF_FILE, HDF_KEY, FEATURE_CONF, WINDOW_SIZE,
    #                            CONTENT_COL)
