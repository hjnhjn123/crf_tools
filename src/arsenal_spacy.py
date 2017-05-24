# -*- coding: utf-8 -*-

from itertools import chain

import spacy

NLP = spacy.load('en')

LABEL_FACTSET = ['PUB', 'EXT', 'SUB', 'PVT', 'MUT', 'UMB', 'PVF', 'HOL', 'MUC', 'TRU',
                 'OPD', 'PEF', 'FND', 'FNS',
                 'JVT', 'VEN', 'NPO', 'HED', 'UIT', 'MUE', 'COL', 'ABS', 'GOV', 'ESP',
                 'PRO', 'FAF', 'SOV', 'COR',
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
    spacy_result = NLP(text)
    spacy_dic = {'chk': [i.text for i in spacy_result.sents],
                 'txt': (i.text for i in spacy_result),
                 'pos': (i.pos_ for i in spacy_result),
                 'dep': (i.dep_ for i in spacy_result),
                 'vec': (i.vector for i in spacy_result)
                 }
    result = {'pos': spacy_dic['pos'],
              'chk': spacy_dic['chk'],
              'vec': spacy_dic['vec'],
              'txt': spacy_dic['txt'],
              'crf': ((a,) + ('O',) + (b,) for (a, b) in zip(spacy_dic['txt'], spacy_dic['pos'])),
              'dep': (i for i in zip(spacy_dic['txt'], spacy_dic['dep'])),
              'pos+dep': (i[:2] + ('O',) + (i[2],) for i in
                          zip(spacy_dic['txt'], spacy_dic['pos'], spacy_dic['dep'])),
              'txt+pos': (i for i in zip(spacy_dic['txt'], spacy_dic['pos'])),
              }
    return list(result[switches])


def spacy_pos_list(text_list):
    """
    | use spacy pos tagger to annotate each chunk
    | add end_label to the end of each chunk
    :param text_list: a list of sentences
    :return: a list of POS result
    """
    result = (spacy_parser(i, 'crf', '') + [('##END', '###', 'O')] for i in text_list)
    result = (i for i in result if len(i) > 1)
    return chain.from_iterable(result)


def spacy_dep_list(text_list):
    """
    | use spacy pos tagger to annotate each chunk
    | add end_label to the end of each chunk
    :param text_list: a list of sentences
    :return: a list of POS result
    """
    result = (spacy_parser(i, 'dep', '') + [('##END', '###', '###')] for i in text_list)
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


def spacy_batch_processing(data, label, col, header, switch):
    """
    :param switches: set switches for spacy_parser
    :param label: set label for spacy_parser
    :param col: set the needed column
    :param header: set the needed header
    :return:
    """
    data = data[data[col] != "\\N"]  # Only containing EOL = Empty line
    result = data[col].apply(spacy_parser, args=('chk', label))  # Chunking
    # result = result.apply(extract_ner_candidate)
    result_dic = {'crf': result.apply(spacy_pos_list),
                  'dep': result.apply(spacy_dep_list)
                  }

    return result_dic[switch]
