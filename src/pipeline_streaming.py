# -*- coding: utf-8 -*-

from itertools import chain

import joblib as jl
import pandas as pd

from .arsenal_boto import sqs_get_msgs, sqs_send_msg, s3_get_file
from .arsenal_crf import batch_add_features, batch_loading, feed_crf_trainer, crf_predict, crf_result2json, \
    df2crfsuite, voting, load_multi_models, prepare_remap
from .arsenal_logging import basic_logging
from .arsenal_spacy import spacy_batch_processing
from .arsenal_stats import get_now
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


##############################################################################


def pipeline_streaming_sqs(in_queue, out_queue, model_f, hdf_f, hdf_key, feature_conf, window_size, col, remap_f):
    sqs_queues = sqs_get_msgs(in_queue)
    model = jl.load(model_f)
    f_dics = batch_loading(hdf_f, hdf_key)
    remap_dic = prepare_remap(remap_f)


    while True:
        for q in sqs_queues.receive_messages(WaitTimeSeconds=10):
            json_input = q.body
            crf_result, raw_df = streaming_pos_crf(json_input, model, f_dics, feature_conf, hdf_key, window_size, col)
            json_result = crf_result2json(crf_result, raw_df, col, remap_dic)
            sqs_send_msg(json_result, queue_name=out_queue)
            basic_logging('Queue output')
            q.delete()


def pipeline_multi_streaming_sqs(in_queue, out_queue, hdf_f, hdf_key, feature_conf, window_size, col, remap_f, *model_fs):
    sqs_queues = sqs_get_msgs(in_queue)
    # models = [jl.load(model) for model in model_fs]
    model_dics = load_multi_models(model_fs)
    f_dics = batch_loading(hdf_f, hdf_key)
    remap_dic = prepare_remap(remap_f)


    while True:
        for q in sqs_queues.receive_messages(WaitTimeSeconds=10):
            json_input = q.body

            crf_results, raw_df = streaming_pos_crf_multi(json_input, f_dics, feature_conf, hdf_key, window_size, col,
                                                          model_dics)
            final_result = voting(crf_results)
            json_result = crf_result2json(final_result, raw_df, col, remap_dic)

            sqs_send_msg(json_result, queue_name=out_queue)
            basic_logging('Queue output')
            q.delete()


def pipeline_offline_single(in_file, out_file, hdf_f, hdf_key, feature_conf, window_size, col, model_f, remap_f):
    model = jl.load(model_f)
    f_dics = batch_loading(hdf_f, hdf_key)
    remap_dic = prepare_remap(remap_f)

    with open(in_file, 'r') as ff:
        count = 0
        out = open(out_file, 'w')
        for json_input in ff:
            crf_result, raw_df = streaming_pos_crf(json_input, model, f_dics, feature_conf, hdf_key, window_size, col)
            json_result = crf_result2json(crf_result, raw_df, col, remap_dic)
            out.write(json_result + '\n')
            if count % 100 == 0:
                print(get_now(), ': Processed ' + str(count) + ' lines')
            count += 1


##############################################################################

def main():
    s3_get_file(S3_BUCKET, MODEL_KEY, MODEL_FILE)
    s3_get_file(S3_BUCKET, HDF_FILE_KEY, HDF_FILE)
    basic_logging('Queue prepared')
    pipeline_multi_streaming_sqs(IN_QUEUE, OUT_QUQUE, HDF_FILE, HDF_KEY, FEATURE_CONF, WINDOW_SIZE, CONTENT_COL,
                                 REMAP_F, MODEL_FS)

    # pipeline_streaming_sqs(IN_QUEUE, OUT_QUQUE, MODEL_FILE, HDF_FILE, HDF_KEY, FEATURE_CONF, WINDOW_SIZE, CONTENT_COL)
