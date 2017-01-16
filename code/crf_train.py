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


def read_faceset(in_file, column_names):
    """
    param: in_file: csv file
    """
    data = csv2pd(column_constant=get_header(in_file), column_names=column_names, engine='c',
                  in_file=in_file, quote=0, sep=',')
    return data


def read_techcrunch(in_file):
    """
    param: in_file: csv file
    """
    data = csv2pd(column_constant=get_header(in_file), column_names=HEADER_TC, engine='c',
                  in_file=in_file, quote=1, sep=',')
    data = data.dropna()
    return data


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
    length = pd.Series((df[entity_col].str.len())).astype(str)
    zeros, tag = pd.Series(['0' for i in range(len(df))]), df[tag_col]
    len_series = pd.Series(list(zip(df[entity_col], zeros, length, df.entity_type)))
    # len_series = len_series.apply(list)
    # result = pd.DataFrame({'Gold_parser_format': tuple(zip(df[entity_col], len_series))})
    result = pd.DataFrame({'Gold_parser_format': len_series})

    return result


def read_gold_train(in_file, col):
    data = read_faceset(in_file, col)
    train_data = [(i[0], list((tuple((int(i[1]), int(i[2]), i[3])),))) for i in data[col].tolist()]
    return train_data


def train_gold_parser(train_data, label=FACTSET_LABEL):
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
    data = read_techcrunch(in_file)
    return data[col].apply(spacy_ner)


def get_factset_sn_type(type_file, sn_file):
    sn = read_faceset(sn_file, HEADER_SN)
    ty = read_faceset(type_file, HEADER_FS)
    result = pd.merge(ty, sn, on='factset_entity_id', how='inner')
    return result.dropna()
