# -*- coding: utf-8 -*-
import os

# FEATURES

# Set features to compute

FEATURE_CONF = {
    'current_original': lambda x: x,
    # 'current_lower': lambda x: x.lower(),
    # 'current_last3': lambda x: x[-3:],
    # 'current_last2': lambda x: x[-2:],
    # 'current_first3': lambda x: x[:3],
    # 'current_first2': lambda x: x[:2],
    # 'current_isupper': lambda x: x.isupper(),
    # 'current_istitle': lambda x: x.istitle(),
    # 'current_isdigit': lambda x: x.isdigit(),
    # 'current_islower': lambda x: x.islower(),
    # 'previous_lower': lambda x: x.lower(),
    # 'previous1_isupper': lambda x: x.isupper(),
    # 'previous1_istitle': lambda x: x.istitle(),
    # 'previous1_isdigit': lambda x: x.isdigit(),
    # 'previous1_islower': lambda x: x.islower(),
    # 'next1_lower': lambda x: x.lower(),
    # 'next1_isupper': lambda x: x.isupper(),
    # 'next1_istitle': lambda x: x.istitle(),
    # 'next1_isdigit': lambda x: x.isdigit(),
    # 'next1_islower': lambda x: x.islower(),
    # 'previous2_lower': lambda x: x.lower(),
    # 'previous2_isupper': lambda x: x.isupper(),
    # 'previous2_istitle': lambda x: x.istitle(),
    # 'previous2_isdigit': lambda x: x.isdigit(),
    # 'previous2_islower': lambda x: x.islower(),
    # 'next2_lower': lambda x: x.lower(),
    # 'next2_isupper': lambda x: x.isupper(),
    # 'next2_istitle': lambda x: x.istitle(),
    # 'next2_isdigit': lambda x: x.isdigit(),
    # 'next2_islower': lambda x: x.islower(),
    # 'previous3_lower': lambda x: x.lower(),
    # 'previous3_isupper': lambda x: x.isupper(),
    # 'previous3_istitle': lambda x: x.istitle(),
    # 'previous3_isdigit': lambda x: x.isdigit(),
    # 'previous3_islower': lambda x: x.islower(),
    # 'next3_lower': lambda x: x.lower(),
    # 'next3_isupper': lambda x: x.isupper(),
    # 'next3_istitle': lambda x: x.istitle(),
    # 'next3_isdigit': lambda x: x.isdigit(),
    # 'next3_islower': lambda x: x.islower()

}

##############################################################################

# FILE DIR
PATH=os.path.dirname(os.path.abspath('..'))
# Set train DIR
TRAIN_F = PATH+'/Data/token_data/orgin_data/new_train.txt'
# TRAIN_F = '/home/hujianan/data/token_data/process_data/new_train.txt'

# Set test DIR
TEST_F = PATH+'/Data/token_data/orgin_data/FixeCnData_2(1)'
# TEST_F = '/home/hujianan/data/token_data/process_data/FixeCnData_2(1)'

# Set validate DIR
VALIDATE_F = '/Users/acepor/Work/patsnap/data/pat360ner_data/annotated_data' \
             '/zdnet_test_20170518.csv'

# Set HDF5 DIR
# HDF_F = '/Users/acepor/Work/patsnap/data/pat360ner_data/dicts/features_20170619.h5'
HDF_F = PATH+'/Data/token_data/process_data/Chemical_dic.h5'

# Set model DIR
# MODEL_F = '/Users/acepor/Work/patsnap/data/pat360ner_data/model/crf_en_model_20170602.joblib'
MODEL_F = PATH+'/Data/token_data/process_data/crf_model_20170731.joblib'

MODEL_FS = ['/Users/acepor/Work/patsnap/data/pat360ner_data/model/crf_en_model_20170602.joblib',
            '/Users/acepor/Work/patsnap/data/pat360ner_data/model/crf_en_model_20170619.joblib']

# Set output DIR
OUT_F = ''

# Set result DIR
# RESULT_F = '/Users/acepor/Work/patsnap/code/pat360ner/log/train_result'
RESULT_F = PATH+'/Data/token_data/process_data'

REMAP_F = 'src/conf/remap_dic.csv'

PREDICT_FILE = PATH+'/Data/token_data/process_data/predict.csv'
##############################################################################

# AWS Conf

# Set S3 bucket name

S3_BUCKET = 'patsnap-360-npl' if not os.environ.get('NLP_S3_BUCKET') else os.environ.get('NLP_S3_BUCKET')

# Set sqs queue

IN_QUEUE = '360_nlp_input' if not os.environ.get('NLP_SQS_QUEUE_INPUT') else os.environ.get('NLP_SQS_QUEUE_INPUT')

OUT_QUQUE = '360_nlp_output' if not os.environ.get('NLP_SQS_QUEUE_OUTPUT') else os.environ.get('NLP_SQS_QUEUE_OUTPUT')

# Set model key

MODEL_KEY = '360-nlp/ner-models/crf_en_model_20170602_bp.joblib' if not os.environ.get('NLP_MODEL_KEY') \
    else os.environ.get('NLP_MODEL_KEY')

# Set model file

MODEL_FILE = './data/crf_en_model_20170602_bp.joblib'

# Set HDF5 file key

HDF_FILE_KEY = '360-nlp/ner-dicts/features_20170425.h5' if not os.environ.get('NLP_HDF_FILE_KEY') \
    else os.environ.get('NLP_HDF_FILE_KEY')

# Set HDF file

HDF_FILE = './data/features_20170425.h5'

# Set json attribute for content

CONTENT_COL = 'DETAIL'

##############################################################################

# MISC

# Set HDF5 key to extract feature dicts
# HDF_KEY = ['aca', 'com_single', 'com_suffix', 'location', 'name', 'ticker']
HDF_KEY = ['chemical_start', 'chemical_end']

# Set header for train/test/validate dataset
# HEADER = ['TOKEN', 'NER', 'POS']
HEADER = ['TOKEN', 'NER']

# Set report type
REPORT_TYPE = 'spc'

# Set cv scale
CV = 5

# Set iteration
ITERATION = 10

# Set Window Size
WINDOW_SIZE = 2
