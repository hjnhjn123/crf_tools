FEATURE_FUNCTION = {
    'lower': lambda x: x.lower(),
    'last3': lambda x: x[-3:],
    'last2': lambda x: x[-2:],
    'isupper': lambda x: x.isupper(),
    'istitle': lambda x: x.istitle(),
    'isdigit': lambda x: x.isdigit(),
    'islower': lambda x: x.islower()
}

HDF_KEY = ['aca', 'com_single', 'com_suffix', 'location', 'name', 'ticker', 'tfdf',
           'tfidf']

f_hdf = '/Users/acepor/Work/patsnap/data/pat360ner_data/dicts/features_20170425.h5'

model_f = ''

train_f = '/Users/acepor/Work/patsnap/data/pat360ner_data/annotated_data/annotate_train_20170425.csv'

test_f = '/Users/acepor/Work/patsnap/data/pat360ner_data/annotated_data/annotate_test_20170425.csv'

report_type = 'spc'