# -*- coding: utf-8 -*-

# FEATURES

FEATURE_FUNCTION = {
    'current_original': lambda x: x,
    'current_lower': lambda x: x.lower(),
    'current_last3': lambda x: x[-3:],
    'current_last2': lambda x: x[-2:],
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

TRAIN_F = '/Users/acepor/Work/patsnap/data/pat360ner_data/annotated_data' \
    '/annotate_train_20170425.csv'

TEST_F = '/Users/acepor/Work/patsnap/data/pat360ner_data/annotated_data' \
    '/annotate_test_20170425.csv'

HDF_F = '/Users/acepor/Work/patsnap/data/pat360ner_data/dicts/features_20170425.h5'

MODEL_F = ''

OUT_F = ''

##############################################################################

# MISC


HDF_KEY = ['aca', 'com_single', 'com_suffix', 'location', 'name', 'ticker', 'tfdf',
           'tfidf']

REPORT_TYPE = 'spc'

CV = 5

ITERATION = 10