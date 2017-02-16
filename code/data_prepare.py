# -*- coding: utf-8 -*-

from .arsenal_stats import *


HEADER_FS = ['fact', 'entity_proper_name', 'entity_type']
HEADER_SN = ['factset_entity_id', 'short_name']
HEADER_SN_TYPE = ['entity_type', 'short_name']
HEADER_SCHWEB = ['Language', 'Title', 'Type']

LABEL_COMPANY = ['PUB', 'EXT', 'SUB', 'PVT', 'MUT', 'UMB', 'PVF', 'HOL', 'MUC', 'TRU', 'OPD', 'PEF', 'FND', 'FNS',
                 'JVT', 'VEN', 'HED', 'UIT', 'MUE', 'ABS', 'GOV', 'ESP', 'PRO', 'FAF', 'SOV', 'COR',
                 'IDX', 'BAS', 'PRT', 'SHP']
LABEL_ANS = ['category', 'nname_en']


##############################################################################


def prepare_ans_dataset(in_file, out_file, col_list=LABEL_ANS):
    """
    It read ANS dataset
    :param in_file: an ANS json file
    :param col_list:
    :return: a df for gold parser to train
    """
    data = json2pd(in_file, col_list)
    data = rename_series(data, 'category', 'entity_types')
    data = rename_series(data, 'nname_en', 'entity_names')
    data['entity_names'] = data['entity_names'].str.title()
    data.to_csv(out_file, index=False)


def prepare_schweb_dataset(in_file, out_file):
    """
    :param in_file: schweb raw csv
    :param out_file: schweb csv
    """
    data = csv2pd(in_file, HEADER_SCHWEB, HEADER_SCHWEB, sep='\t')
    en_data = data[data.Language == 'en']
    result = en_data[en_data.Type.str.contains('Location|Personal|Organisation')]
    result['entity_type'] = np.where(result.Type.str.contains('Personal'), 'PERSON',
                                     np.where(result.Type.str.contains('Location'), 'GPE',
                                              np.where(result.Type.str.contains('Organisation'), 'ORG', 'MISC')))
    result = rename_series(result, 'Title', 'entity_name')
    result = result.drop(['Language', 'Type'], axis=1)
    result.to_csv(out_file, index=False)


##############################################################################


def output_factset_sn_type(type_file, sn_file, out_file):
    sn = quickest_read_csv(sn_file, HEADER_SN)
    ty = quickest_read_csv(type_file, HEADER_FS)
    result = pd.merge(ty, sn, on='factset_entity_id', how='inner')
    result = result.dropna()
    result.tocsv(out_file, index=False)


def remap_factset_sn_type(in_file, out_file):
    data = quickest_read_csv(in_file, HEADER_SN_TYPE)
    result = remap_series(data, 'entity_type', 'new_entity_type', LABEL_COMPANY, 'ORG')
    result = result.drop(['entity_type'], axis=1)
    result.to_csv(out_file, index=False)

def extract_factset_short_names(in_file, out_single, out_multi):
    data = quickest_read_csv(in_file, ['entity_proper_name', 'entity_type', 'factset_entity_id', 'short_name'])
    single_name = data[data.short_name.str.split(' ').apply(len) == 1]
    multi_name = data[data.short_name.str.split(' ').apply(len) > 1]
    single_name = single_name.drop(['entity_proper_name', 'entity_type', 'factset_entity_id'], axis=1)
    multi_name = multi_name.drop(['entity_proper_name', 'entity_type', 'factset_entity_id'], axis=1)
    single_name.to_csv(out_single, index=False)
    multi_name.to_csv(out_multi, index=False)


def split_city(in_file, single_file, multi_file):
    out_single, out_multi = open(single_file, 'w'), open(multi_file, 'w')
    with open(in_file, 'r') as data:
        for line in data:
            if len(line.split(' ')) == 1:
                out_single.write(line)
            else:
                out_multi.write(line)


def titlefy_names(in_file, out_file):
    out = open(out_file, 'w')
    with open(in_file, 'r') as data:
        result = [line for line in data if len(line) > 2]
        result = [line.title() for line in result]
        for line in result:
            out.write(line)

