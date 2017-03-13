# -*- coding: utf-8 -*-

import random
from collections import OrderedDict
from itertools import chain

import pandas as pd
import spacy
from spacy import pipeline, gold

NLP = spacy.load('en')

LABEL_FACTSET = ['PUB', 'EXT', 'SUB', 'PVT', 'MUT', 'UMB', 'PVF', 'HOL', 'MUC', 'TRU', 'OPD', 'PEF', 'FND', 'FNS',
                 'JVT', 'VEN', 'NPO', 'HED', 'UIT', 'MUE', 'COL', 'ABS', 'GOV', 'ESP', 'PRO', 'FAF', 'SOV', 'COR',
                 'IDX', 'BAS', 'PRT', 'SHP']
LABEL_NER = ('PERSON', 'NORP', 'ORG', 'GPE', 'PRODUCT', 'EVENT', 'MONEY')

HEADER_SN_TYPE = ['entity_type', 'short_name']
HEADER_TC = ['"ID"', '"TITLE"', '"CONTENT"', '"TIME"']


##############################################################################################


# Applying


def spacy_parser(text, switches, label):
    """
    :param text: a sentence or a doc
    :param switch: a list of switches: chk for sentence chunking, pos for pos tagging, and ner for NER,
    crf for crf pre-processing
    :param label: filtering the NER labels
    :return:
    """
    nlp_result = NLP(text)
    spacy_dic = {'chk': [i.text for i in nlp_result.sents],
                 'pos': [(i.text, i.pos_) for i in nlp_result],
                 'crf': [(i.text, i.pos_, 'O') for i in nlp_result],
                 'ner': OrderedDict(
                     [(i.text, (i.start, i.end, i.label_)) for i in nlp_result.ents if i.label_ in label]),
                 'dep': [(i.text, i.dep_) for i in nlp_result]
                 }
    return spacy_dic[''.join(switches)] if len(switches) == 1 else [spacy_dic[i] for i in switches]


def spacy_pos_text_list(text_list):
    """
    | use spacy pos tagger to annotate each chunk
    | add end_label to the end of each chunk
    :param text_list: a list of sentences
    :return: a list of POS result
    """
    result = (spacy_parser(i, ['crf'], '') + [('##END', '###', 'O')] for i in text_list)
    result = (i for i in result if len(i) > 1)
    return chain.from_iterable(result)


def extract_ner_candidate(sents):
    """
    If a chunk contains more than two non-lower words.
    :param sents: chunks
    :return: ner candidates
    """
    k, result = 0, []
    for sent in sents:
        for word in sent.split(' '):
            if word.isalnum():
                # extract all numbers and alphabets
                if not word.islower():
                    # extract non-lower words
                    k += 1
        if k > 2:
            result.append(sent)
            k = 0
    return result


def spacy_batch_processing(data, switches, label, col, header):
    """
    :param switches: set switches for spacy_parser
    :param label: set label for spacy_parser
    :param col: set the needed column
    :param header: set the needed header
    :return:
    """
    data = data[data[col] != "\\N"]  # Only containing EOL = Empty line
    result = data[col].apply(spacy_parser, args=(switches, label))  # Chunking
    result = result.apply(extract_ner_candidate)
    result = result.apply(spacy_pos_text_list)  # POS tagging
    return result


##############################################################################################


# Training


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
    data = pd.read_csv(input, usecols=col, quoting=0, engine='c') if file else input
    train_data = [(i[0], list((tuple((int(i[1]), int(i[2]), i[3])),))) for i in data[col].tolist()]
    return train_data


def gold_parser(train_data, label=LABEL_FACTSET):
    """
    https://spacy.io/docs/usage/entity-recognition#updating
    :param train_data: list of (string containing entities, [(start, end, type)])
    :param label: a list of entity types
    """
    ner = pipeline.EntityRecognizer(NLP.vocab, entity_types=label)
    for itn in range(5):
        random.shuffle(train_data)
        for raw_text, entity_offsets in train_data:
            doc = NLP.make_doc(raw_text)
            gold = spacy.gold.GoldParse(doc, entities=entity_offsets)
            NLP.tagger(doc)
            ner.update(doc, gold)
    ner.model.end_training()
