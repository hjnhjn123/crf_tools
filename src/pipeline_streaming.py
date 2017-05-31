# -*- coding: utf-8 -*-

from itertools import chain

import pandas as pd

from .arsenal_boto import sqs_get_msgs, sqs_send_msg, s3_get_file
from .arsenal_crf import batch_add_features, batch_loading, feed_crf_trainer, crf_predict, crf_result2json, df2crfsuite
from .arsenal_logging import basic_logging
from .arsenal_spacy import spacy_batch_processing
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


##############################################################################


def pipeline_streaming_sqs(in_queue, out_queue, model_f, hdf_f, hdf_key, feature_conf, window_size, col, online):
    sqs_queues = sqs_get_msgs(in_queue, online)
    crf, f_dics = batch_loading(model_f, hdf_f, hdf_key)

    while True:
        for q in sqs_queues.receive_messages():
            json_input = q.body
            crf_result, raw_df = streaming_pos_crf(json_input, crf, f_dics, feature_conf, hdf_key, window_size, col)
            json_result = crf_result2json(crf_result, raw_df, col)
            sqs_send_msg(json_result, queue_name=out_queue)
            basic_logging('Queue output')
            # q.delete()


##############################################################################

def main(argv):
    if len(argv) > 1 and argv[1] == 'aws':
        online = argv[1]
        s3_get_file(S3_BUCKET, MODEL_KEY, MODEL_FILE, online)
        s3_get_file(S3_BUCKET, HDF_FILE_KEY, HDF_FILE, online)
        basic_logging('Queue prepared')
        pipeline_streaming_sqs(IN_QUEUE, OUT_QUQUE, MODEL_FILE, HDF_FILE, HDF_KEY, FEATURE_CONF, WINDOW_SIZE,
                               CONTENT_COL, online)
