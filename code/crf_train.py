# -*- coding: utf-8 -*-

from .arsenal_stats import *
import spacy
import random
from spacy.gold import GoldParse
from spacy.pipeline import EntityRecognizer

HEADER_FS = ['fact', 'entity_proper_name', 'entity_type']
HEADER_SN = ['factset_entity_id', 'short_name']
HEADER_SN_TYPE = ['entity_type', 'short_name']
HEADER_TC = ['"ID"', '"TITLE"', '"CONTENT"', '"TIME"']
NER_LABEL = ('PERSON', 'NORP', 'ORG', 'GPE', 'PRODUCT', 'EVENT', 'MONEY')
FACTSET_LABEL = ['PUB', 'EXT', 'SUB', 'PVT', 'MUT', 'UMB', 'PVF', 'HOL', 'MUC', 'TRU', 'OPD', 'PEF', 'FND', 'FNS',
                 'JVT', 'VEN', 'NPO', 'HED', 'UIT', 'MUE', 'COL', 'ABS', 'GOV', 'ESP', 'PRO', 'FAF', 'SOV', 'COR',
                 'IDX', 'BAS', 'PRT', 'SHP']

NLP = spacy.load('en')


def spacy_ner(sent):
    """
    param: sent csv file
    return: {ner: ner type}
    """
    entity = NLP(sent).ents
    extracted = {i.text: (i.start, i.end, i.label_) for i in entity if i.label_ in NER_LABEL}
    # sentence = {k: i for k, i in enumerate(sent.split(' '))}
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


def read_spacy_ner_train_data(in_file, col):
    """
    ('Who is Chaka Khan?', [(7, 17, 'PERSON')]),
    :param in_file:
    :param col:
    :return: list of (string containing entities, [(start, end, type)])
    """
    data = quickest_read_csv(in_file, col)
    train_data = [(i[0], list((tuple((int(i[1]), int(i[2]), i[3])),))) for i in data[col].tolist()]
    return train_data


def train_gold_parser(train_data, label=FACTSET_LABEL):
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


def batch_processing(in_file, col='CONTENT'):
    data = quickest_read_csv(in_file, HEADER_TC)
    data = data.dropna()
    return data[col].apply(spacy_ner)


def get_factset_sn_type(type_file, sn_file):
    sn = quickest_read_csv(sn_file, HEADER_SN)
    ty = quickest_read_csv(type_file, HEADER_FS)
    result = pd.merge(ty, sn, on='factset_entity_id', how='inner')
    return result.dropna()
