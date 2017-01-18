# -*- coding: utf-8 -*-

from .arsenal_stats import *
import numpy as np
import spacy
import random
from spacy.gold import GoldParse
from spacy.pipeline import EntityRecognizer

HEADER_FS = ['fact', 'entity_proper_name', 'entity_type']
HEADER_SN = ['factset_entity_id', 'short_name']
HEADER_SN_TYPE = ['entity_type', 'short_name']
HEADER_TC = ['"ID"', '"TITLE"', '"CONTENT"', '"TIME"']

LABEL_NER = ('PERSON', 'NORP', 'ORG', 'GPE', 'PRODUCT', 'EVENT', 'MONEY')

LABEL_FACTSET = ['PUB', 'EXT', 'SUB', 'PVT', 'MUT', 'UMB', 'PVF', 'HOL', 'MUC', 'TRU', 'OPD', 'PEF', 'FND', 'FNS',
                 'JVT', 'VEN', 'NPO', 'HED', 'UIT', 'MUE', 'COL', 'ABS', 'GOV', 'ESP', 'PRO', 'FAF', 'SOV', 'COR',
                 'IDX', 'BAS', 'PRT', 'SHP']

LABEL_COMPANY = ['PUB', 'EXT', 'SUB', 'PVT', 'MUT', 'UMB', 'PVF', 'HOL', 'MUC', 'TRU', 'OPD', 'PEF', 'FND', 'FNS',
                 'JVT', 'VEN', 'HED', 'UIT', 'MUE', 'ABS', 'GOV', 'ESP', 'PRO', 'FAF', 'SOV', 'COR',
                 'IDX', 'BAS', 'PRT', 'SHP']

LABEL_COLLEGE = ['COL']

LABEL_REMAPPED = ['ORG', 'MISC']

LABEL_ANS = ['category', 'nname_en']

NLP = spacy.load('en')


def spacy_ner_recogniser(sent):
    """
    param: sent csv file
    return: {ner: ner type}
    """
    entity = NLP(sent).ents
    extracted = {i.text: (i.start, i.end, i.label_) for i in entity if i.label_ in LABEL_NER}
    return extracted


def df2gold_parser(df, entity_col='short_name', tag_col='entity_type'):
    """
    ('Who is Chaka Khan?', [(7, 17, 'PERSON')]),
    :param df: a df containing entity names and entity types
    :param entity_col: entity names
    :param tag_col: entity types
    :return: (string containing entities, [(start, end, type)])
    """
    length = pd.Series((df[entity_col].str.len())).astype(str)
    zeros, tag = pd.Series(['0' for i in range(len(df))]), df[tag_col]
    len_series = pd.Series(list(zip(df[entity_col], zeros, length, df.entity_type)))
    result = pd.DataFrame({'Gold_parser_format': len_series})
    return result


def read_gold_parser_train_data(input, col, file=True):
    """
    ('Who is Chaka Khan?', [(7, 17, 'PERSON')]),
    :param input:
    :param col:
    :param file: True for csv file, else dataframe
    :return: list of (string containing entities, [(start, end, type)])
    """
    data = quickest_read_csv(input, col) if file is True else input
    train_data = [(i[0], list((tuple((int(i[1]), int(i[2]), i[3])),))) for i in data[col].tolist()]
    return train_data


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


def gold_parser(train_data, label=LABEL_FACTSET):
    """
    https://spacy.io/docs/usage/entity-recognition#updating
    :param train_data: list of (string containing entities, [(start, end, type)])
    :param label: a list of entity types
    """
    ner = EntityRecognizer(NLP.vocab, entity_types=label)
    for itn in range(5):
        random.shuffle(train_data)
        for raw_text, entity_offsets in train_data:
            doc = NLP.make_doc(raw_text)
            gold = GoldParse(doc, entities=entity_offsets)
            NLP.tagger(doc)
            ner.update(doc, gold)
    ner.model.end_training()

##############################################################################################


def batch_processing(in_file, out_file, col='CONTENT'):
    data = quickest_read_csv(in_file, HEADER_TC)
    data = data.dropna()
    result = data[col].apply(spacy_ner_recogniser)
    result.to_csv(out_file)


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


def train_gold_parser(in_file, entity_col, tag_col, gold_parser_col, label=LABEL_REMAPPED):
    data = quickest_read_csv(in_file, HEADER_SN_TYPE)
    data = df2gold_parser(data, entity_col, tag_col)
    data = read_gold_parser_train_data(data, gold_parser_col, False)
    gold_parser(data, label)
