# -*- coding: utf-8 -*-

import os
from itertools import chain, groupby

import pandas as pd

from .arsenal_boto import sqs_get_msgs, sqs_send_msg, s3_get_file
from .arsenal_crf import batch_add_features, batch_loading, feed_crf_trainer, crf_predict, crf_result2json
from .arsenal_spacy import spacy_batch_processing
from .settings import *


##############################################################################


def streaming_pos_crf(in_f, crf, f_dics, feature_conf, hdf_key, window_size):
    raw_df = pd.read_json(in_f, lines=True)
    raw_df['content'] = raw_df.result.to_dict()[0]['content']

    parsed_data = chain.from_iterable(spacy_batch_processing(raw_df, '', 'content', ['content'], 'crf'))
    prepared_data = [list(x[1]) for x in groupby(parsed_data, lambda x: x == ('##END', '###', 'O')) if not x[0]]
    test_sents = batch_add_features(prepared_data, f_dics)

    X_test, y_test = feed_crf_trainer(test_sents, feature_conf, hdf_key, window_size)
    crf_result = crf_predict(crf, prepared_data, X_test)
    return crf_result, raw_df


##############################################################################


def pipeline_streaming_sqs(in_bucket, out_bucket, model_f, hdf_f, hdf_key, feature_conf, window_size):
    sqs_queues = sqs_get_msgs(in_bucket)
    crf, f_dics = batch_loading(model_f, hdf_f, hdf_key)

    while True:
        for q in sqs_queues.receive_messages():
            json_input = q.body
            crf_result, raw_df = streaming_pos_crf(json_input, crf, f_dics, feature_conf, hdf_key, window_size)
            json_result = crf_result2json(crf_result, raw_df)
            sqs_send_msg(json_result, queue_name=out_bucket)
            q.delete()


##############################################################################

def main(argv):
    model_f = s3_get_file(BUCKET, MODEL_KEY, MODEL_FILE)
    hdf_f = s3_get_file(BUCKET, HDF_FILE_KEY, HDF_FILE)

    if len(argv) > 1 and argv[1] == 'aws':
        in_bucket, out_bucket = os.environ['NLP_QUEUE_IN'], os.environ['NLP_QUEUE_OUT']
        pipeline_streaming_sqs(in_bucket, out_bucket, model_f, hdf_f, HDF_KEY, FEATURE_CONF, WINDOW_SIZE)
