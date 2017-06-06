#!/usr/bin/env bash

export METRICS_URL=http://10.10.130.103:86

export NLP_S3_BUCKET=patsnap-360-npl
export NLP_SQS_QUEUE_INPUT=360_nlp_input
export NLP_SQS_QUEUE_OUTPUT=360_nlp_output

export NLP_MODEL_KEY=360-nlp/ner-models/crf_en_model_20170602_bp.joblib
export NLP_HDF_FILE_KEY=360-nlp/ner-dicts/features_20170425.h5

export NLP_MODE=dev
export NLP_DEV_S3_ENDPOINT_URL=http://192.168.27.190:4572
export NLP_DEV_SQS_ENDPOINT_URL=http://192.168.27.190:4576

export AWS_ACCESS_KEY_ID=
export AWS_SECRET_ACCESS_KEY=
export AWS_ACCESS_REGION=