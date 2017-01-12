# -*- coding: utf-8 -*-

from .arsenal_stats import *
import spacy


HEADER_FS = ['"_id"', 'entity_name', 'entity_proper_name', 'entity_sub_type', 'entity_type']
HEADER_TC = ['"ID"', '"TITLE"', '"CONTENT"', '"TIME"']
NER_LABEL = ('PERSON', 'NORP', 'ORG', 'GPE', 'PRODUCT', 'EVENT', 'MONEY')
NLP = spacy.load('en')


def read_faceset(in_file):
    """
    param: in_file: csv file
    """
    data = csv2pd(column_constant=get_header(in_file), column_names=HEADER_FS, engine='c',
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


def batch_processing(in_file, col='CONTENT'):
    data = read_techcrunch(in_file)
    return data[col].apply(spacy_ner)
