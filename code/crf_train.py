# -*- coding: utf-8 -*-

from .arsenal_stats import *
import spacy


HEADER_FS = ['fact', 'entity_proper_name', 'entity_type']
HEADER_SN = ['factset_entity_id', 'short_name']
HEADER_TC = ['"ID"', '"TITLE"', '"CONTENT"', '"TIME"']
NER_LABEL = ('PERSON', 'NORP', 'ORG', 'GPE', 'PRODUCT', 'EVENT', 'MONEY')
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


# def df2gold_parser(df, entity_col='entity_proper_name', tag_col='entity_type'):
#     length = df[entity_col].str.len()
#     tag = df[tag_col]
#     zip(len_list)


##############################################################################################


def batch_processing(in_file, col='CONTENT'):
    data = read_techcrunch(in_file)
    return data[col].apply(spacy_ner)


def get_factset_sn_type(type_file, sn_file):
    sn = read_faceset(sn_file, HEADER_SN)
    ty = read_faceset(type_file, HEADER_FS)
    result = pd.merge(ty, sn, on='factset_entity_id', how='inner')
    return result.dropna()
