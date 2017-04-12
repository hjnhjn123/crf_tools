# -*- coding: utf-8 -*-

from scipy import stats

from arsenal_nlp import *
from arsenal_spacy import *

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
    data = csv2pd(in_file, HEADER_SCHWEB, sep='\t')
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


##############################################################################


def output_tfidf(in_file, out_file, cols, col_name):
    out = open(out_file, 'w')
    data = json2pd(in_file, cols, lines=True)
    data = data[col_name].apply(remove_punc)
    tfidf = get_tfidf(data.tolist())
    for k, v in tfidf.items():
        if v > 1.0:
            out.write(k + ',' + str(v) + '\n')


def output_tfdf(in_file, out_file, cols, col_name):
    out = open(out_file, 'w')
    data = json2pd(in_file, cols, lines=True)
    data = data[col_name].apply(remove_punc)
    _, tfdf, _ = get_tfdf(data.tolist())
    for k, v in tfdf.items():
        if v > 1.0:
            out.write(k + ',' + str(v) + '\n')


def output_tfidf_zscore(in_file, out_file, cols, col_name):
    data = json2pd(in_file, cols, lines=True)
    data = data[col_name].apply(remove_punc)
    tfidf_dic = get_tfidf(data.tolist())
    tfidf_df = pd.DataFrame.from_dict(tfidf_dic, orient='index')
    tfidf_df.columns = ['tf_idf']
    tfidf_df['zscore'] = stats.mstats.zscore(tfidf_df['tf_idf'])
    tfidf_df['zvalue'] = tfidf_df['zscore'].apply(lambda x: 0 if x < 0 else 1)
    tfidf_df = tfidf_df.drop(['tf_idf', 'zscore'], axis=1)
    tfidf_df.to_csv(out_file, header=False)


def output_tfdf_zscore(in_file, out_file, cols, col_name):
    data = json2pd(in_file, cols, lines=True)
    data = data[col_name].apply(remove_punc)
    _, tfdf_dic, _ = get_tfdf(data.tolist())
    tfdf_df = pd.DataFrame.from_dict(tfdf_dic, orient='index')
    tfdf_df.columns = ['tf_idf']
    tfdf_df['zscore'] = stats.mstats.zscore(tfdf_df['tf_idf'])
    tfdf_df['zvalue'] = tfdf_df['zscore'].apply(lambda x: 0 if x < 0 else 1)
    tfdf_df = tfdf_df.drop(['tf_idf', 'zscore'], axis=1)
    tfdf_df.to_csv(out_file, header=False)


def titlefy_names(in_file, out_file):
    out = open(out_file, 'w')
    with open(in_file, 'r') as data:
        result = [line for line in data if len(line) > 2]
        result = [line.title() for line in result]
        for line in result:
            out.write(line)


def train_gold_parser(in_file, entity_col, tag_col, gold_parser_col, label):
    data = quickest_read_csv(in_file, HEADER_SN_TYPE)
    data = df2gold_parser(data, entity_col, tag_col)
    data = read_gold_parser_train_data(data, gold_parser_col, False)
    gold_parser(data, label)


def prepare_techcrunch(in_file, header, col):
    data = quickest_read_csv(in_file, header)
    data = data.dropna()
    data = clean_dataframe(data, [col], rpls={'\n': ' ', '\t': ' '})
    return data


def process_techcrunch(in_file, out_file, cols, pieces=10):
    data = json2pd(in_file, cols, lines=True)
    data = data.dropna()
    random_data = random_rows(data, pieces, 'content')
    parsed_data = spacy_batch_processing(random_data, '', 'content', ['content'], 'crf')
    parsed_data = reduce(add, parsed_data)
    pd.DataFrame(parsed_data, columns=['TOKEN', 'POS', 'NER']).to_csv(out_file, header=False, index=False)


##############################################################################


# Evaluation

def compare_difference(fixed_f, bug_f, out_f, fp_f, header, new_header):
    fixed = pd.read_csv(fixed_f, header=header, sep=',', engine='c', quoting=0)
    bug = pd.read_csv(bug_f, header=header, sep=',', engine='c', quoting=0)
    print('fixed: ', len(fixed), 'bug: ', len(bug))
    fixed.columns = new_header if header == None else header
    bug.columns = new_header if header == None else header
    merged = pd.concat([bug.Token, bug.NER, fixed.NER], axis=1)
    merged.columns = ['Token', 'bugged_NER', 'fixed_NER']
    difference = merged[merged['bugged_NER'] != merged['fixed_NER']]
    difference.to_csv(out_f, index=False)
    false_positive = difference[difference['fixed_NER'] == 'O']
    false_positive.to_csv(fp_f, index=False)

