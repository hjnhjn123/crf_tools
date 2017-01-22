# -*- coding: utf-8 -*-

from .arsenal_stats import *
import random
import spacy
from collections import OrderedDict

NLP = spacy.load('en')


LABEL_FACTSET = ['PUB', 'EXT', 'SUB', 'PVT', 'MUT', 'UMB', 'PVF', 'HOL', 'MUC', 'TRU', 'OPD', 'PEF', 'FND', 'FNS',
                 'JVT', 'VEN', 'NPO', 'HED', 'UIT', 'MUE', 'COL', 'ABS', 'GOV', 'ESP', 'PRO', 'FAF', 'SOV', 'COR',
                 'IDX', 'BAS', 'PRT', 'SHP']
LABEL_NER = ('PERSON', 'NORP', 'ORG', 'GPE', 'PRODUCT', 'EVENT', 'MONEY')


HEADER_SN_TYPE = ['entity_type', 'short_name']
HEADER_TC = ['"ID"', '"TITLE"', '"CONTENT"', '"TIME"']

##############################################################################################


def df2gold_parser(df, entity_col='short_name', tag_col='entity_type'):
    """
    ('Who  is Chaka Khan?', [(7, 17, 'PERSON')]),
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


def spacy_chunker(doc):
    """
    :param doc:
    :return: a list of sentences
    """
    return [i for i in NLP(doc).sents]


def spacy_ner(sent):
    """
    param: sent csv file
    return: {ner: ner type}
    """
    entity = NLP(sent).ents
    extracted = OrderedDict([(i.text, (i.start, i.end, i.label_)) for i in entity if i.label_ in LABEL_NER])
    return extracted


def spacy_pos(sent):
    """
    param: sent csv file
    return: {word: pos}
    """
    doc = NLP(sent)
    extracted = OrderedDict([(i.text, i.pos_) for i in doc])
    return extracted


def gold_parser(train_data, label=LABEL_FACTSET):
    """
    https://spacy.io/docs/usage/entity-recognition#updating
    :param train_data: list of (string containing entities, [(start, end, type)])
    :param label: a list of entity types
    """
    ner = spacy.pipeline.EntityRecognizer(NLP.vocab, entity_types=label)
    for itn in range(5):
        random.shuffle(train_data)
        for raw_text, entity_offsets in train_data:
            doc = NLP.make_doc(raw_text)
            gold = spacy.gold.GoldParse(doc, entities=entity_offsets)
            NLP.tagger(doc)
            ner.update(doc, gold)
    ner.model.end_training()


##############################################################################################


DIC_SPACY = {'ner': spacy_ner,
             'pos': spacy_pos}


def spacy_batch_processing(in_file, out_file, switch, col='CONTENT', header=HEADER_TC):
    data = quickest_read_csv(in_file, header)
    data = data.dropna()
    result = data[col].apply(DIC_SPACY[switch])
    result.to_csv(out_file)


def train_gold_parser(in_file, entity_col, tag_col, gold_parser_col, label):
    data = quickest_read_csv(in_file, HEADER_SN_TYPE)
    data = df2gold_parser(data, entity_col, tag_col)
    data = read_gold_parser_train_data(data, gold_parser_col, False)
    gold_parser(data, label)