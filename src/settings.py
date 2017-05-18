# -*- coding: utf-8 -*-

# FEATURES

# Set features to compute

FEATURE_FUNCTION = {
    'current_original': lambda x: x,
    'current_lower': lambda x: x.lower(),
    'current_last3': lambda x: x[-3:],
    'current_last2': lambda x: x[-2:],
    'current_first3': lambda x: x[:2],
    'current_first2': lambda x: x[-3:],
    'current_isupper': lambda x: x.isupper(),
    'current_istitle': lambda x: x.istitle(),
    'current_isdigit': lambda x: x.isdigit(),
    'current_islower': lambda x: x.islower(),
    'previous_lower': lambda x: x.lower(),
    'previous_isupper': lambda x: x.isupper(),
    'previous_istitle': lambda x: x.istitle(),
    'previous_isdigit': lambda x: x.isdigit(),
    'previous_islower': lambda x: x.islower(),
    'next_lower': lambda x: x.lower(),
    'next_isupper': lambda x: x.isupper(),
    'next_istitle': lambda x: x.istitle(),
    'next_isdigit': lambda x: x.isdigit(),
    'next_islower': lambda x: x.islower()
}

##############################################################################

# FILE DIR

# Set train DIR
TRAIN_F = '/Users/acepor/Work/patsnap/data/pat360ner_data/annotated_data' \
    '/annotate_train_20170425.csv'

# Set test DIR
TEST_F = '/Users/acepor/Work/patsnap/data/pat360ner_data/annotated_data' \
    '/annotate_test_20170425.csv'

# Set validate DIR
VALIDATE_F = '/Users/acepor/Work/patsnap/data/pat360ner_data/annotated_data' \
    '/zdnet_test_20170425.csv'

# Set HDF5 DIR
HDF_F = '/Users/acepor/Work/patsnap/data/pat360ner_data/dicts/features_20170425.h5'

# Set model DIR
MODEL_F = ''

# Set output DIR
OUT_F = ''

##############################################################################

# MISC

# Set HDF5 key to extract feature dicts
HDF_KEY = ['aca', 'com_single', 'com_suffix', 'location', 'name', 'ticker', 'tfidf']

# Set report type
REPORT_TYPE = 'spc'

# Set cv scale
CV = 5

# Set iteration
ITERATION = 10