# -*- coding: utf-8 -*-

from itertools import chain

import pandas as pd
import joblib as jl

from .arsenal_boto import sqs_get_msgs, sqs_send_msg, s3_get_file
from .arsenal_crf import batch_add_features, batch_loading, feed_crf_trainer, crf_predict, crf_result2json, df2crfsuite
from .arsenal_logging import basic_logging
from .arsenal_spacy import spacy_batch_processing
from .arsenal_stats import combine_multi_df
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


def streaming_pos_crf_multi(in_f, f_dics, feature_conf, hdf_key, window_size, col, models):
    raw_df = pd.read_json(in_f, lines=True)
    raw_df['content'] = raw_df[col].to_dict()[0]['content']
    parsed_data = chain.from_iterable(spacy_batch_processing(raw_df, '', 'content', ['content'], 'crf'))
    prepared_data = pd.DataFrame(list(parsed_data))
    test_df = batch_add_features(prepared_data, f_dics)
    test_sents = df2crfsuite(test_df)
    X_test, y_test = feed_crf_trainer(test_sents, feature_conf, hdf_key, window_size)
    crf_results = [crf_predict(model, test_sents, X_test) for model in models]
    return crf_results, raw_df


def voting(crf_results):
    crf_dfs = [pd.DataFrame(crf_list, columns=HEADER) for crf_list in crf_results]
    combined = pd.concat(crf_dfs, axis=1)
    combined.columns = range(len(combined.columns))
    to_vote = combined[[i for i in combined.columns if (i + 1) % 3 == 0]].apply(tuple, axis=1).tolist()
    # convert a df to zipped list

    vote_result = []
    for i in to_vote:
        if i == tuple('O' for k in range(len(i))):
            vote_result.append('O')
            # if all items are 'O', just append it
        for j in range(len(i)):
            if i[j] != 'O':
                vote_result.append(i[j])
                break
            elif i[j] == 'O':
                j += 1
    return list(zip(combined[0].tolist(), combined[1].tolist(), vote_result))


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
    models = [jl.load(model) for model in model_fs]
    f_dics = batch_loading(hdf_f, hdf_key)

    while True:
        for q in sqs_queues.receive_messages():
            json_input = q.body
            crf_results, raw_df = streaming_pos_crf_multi(json_input, f_dics, feature_conf, hdf_key, window_size,col, models)
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
    pipeline_streaming_sqs(IN_QUEUE, OUT_QUQUE, MODEL_FILE, HDF_FILE, HDF_KEY, FEATURE_CONF, WINDOW_SIZE,
                               CONTENT_COL)
