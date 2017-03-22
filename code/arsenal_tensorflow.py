# -*- coding: utf-8 -*-

import re

from .pipeline_crf import *

HEADER_ANNOTATION = ['TOKEN', 'POS', 'NER']

LEARNING_RATE = 0.001
TRAINING_ITERS = 100000
BATCH_SIZE = 128
DISPLAY_STEP = 10

EMBEDDING_SIZE = 311  # (300 for word2vec embeddings and 11 for extra features (POS,CHUNK,CAP))
MAX_DOCUMENT_LENGTH = 30
MAX_WORD_LENGTH = 15
num_classes = 5
BASE_DIR = "ner"  # path to coNLL data set


##############################################################################


def add_word_embedding(sent):
    """
    :param sent: [(word, pos, ner)]
    :param feature_set: {feature1, feature2}
    :return: [(word, pos, ner, other_features)]
    """
    embeddings = [spacy_parser(i[0], 'vec', '') for i in sent]
    return [(sent[i] + (embeddings[i],)) for i in range(len(list(sent)))]


def prepare_rnn_feature_dict(city_f, com_single_f, com_suffix_f, country_f, name_f):
    name, country = line_file2set(name_f), line_file2set(country_f)
    city, com_single = line_file2set(city_f), line_file2set(com_single_f)
    com_suffix = {i.title() for i in line_file2set(com_suffix_f)}
    return city, com_single, com_suffix, country, name


def batch_rnn_loading(dict_conf, city_f, com_single_f, com_suffix_f, country_f, name_f):
    conf = load_yaml_conf(dict_conf)
    features = prepare_rnn_feature_dict(city_f, com_single_f, com_suffix_f, country_f, name_f)
    city, com_single, com_suffix, country, name = features
    return conf, city, com_single, com_suffix, country, name


def add_features_rnn(token, city, com_single, com_suffix, country, name):
    onehot = np.zeros(5)
    if token in city:
        onehot[0] = 1
    elif token in com_single:
        onehot[1] = 1
    elif token in com_suffix:
        onehot[2] = 1
    elif token in country:
        onehot[3] = 1
    elif token in name:
        onehot[4] = 1
    # vector = get_embedding(token)
    vector = spacy_parser(token, 'vec', '')
    return np.append(onehot, vector)


def batch_add_features_rnn(sents, city, com_single, com_suffix, country, name):
    result = []
    for sent in sents:
        sent_vec = [add_features_rnn(token, city, com_single, com_suffix, country, name) for token, pos, ner in sent]
        pos_vec = [get_pos(pos) for token, pos, ner in sent]
        final_vec = np.asarray([np.asarray(np.append(m, n)) for m, n in zip(sent_vec, pos_vec)])
        result.append(final_vec)
    return np.asarray(result)


def process_annotated_rnn(in_file):
    """
    | following python-crfsuit, sklearn_crfsuit doesn't support pandas DF, so a feature dic is used instead
    | http://python-crfsuite.readthedocs.io/en/latest/pycrfsuite.html#pycrfsuite.ItemSequence
    :param in_file: CSV file: TOKEN, POS, NER
    :return: [[sent]]
    """
    data = pd.read_csv(in_file, header=None, engine='c', quoting=0)
    data.columns = HEADER_ANNOTATION
    data = data.dropna()
    sents = (tuple(i) for i in zip(data['TOKEN'].tolist(), data['POS'].tolist(), data['NER'].tolist()))
    sents = (list(x[1])[:-1] for x in groupby(sents, lambda x: x == ('##END', '###', 'O')) if not x[0])
    sents = [i for i in sents if i != []]
    return sents


##############################################################################


def get_pos(tag):
    onehot = np.zeros(6)
    if tag == 'NOUN':
        onehot[0] = 1
    elif tag == 'PRORN':
        onehot[1] = 1
    elif tag == 'SYM':
        onehot[2] = 1
    elif tag == 'NUM':
        onehot[3] = 1
    elif tag == 'X':
        onehot[4] = 1
    else:
        onehot[5] = 1
    return onehot


PRE_EMBEDDING = defaultdict()
for line in open('/Users/acepor/Work/patsnap/code/pat360ner/data/myvectors.txt'):
    l = line.strip().split()
    w = l[0]
    arr = l[1:]
    PRE_EMBEDDING[w] = arr

print("PRE_EMB size", len(PRE_EMBEDDING))


def get_embedding(token):
    randV = np.random.uniform(-0.25, 0.25, EMBEDDING_SIZE - 11)
    s = re.sub('[^0-9a-zA-Z]+', '', token)
    arr = []
    if token == "##END":
        arr = [0 for _ in range(EMBEDDING_SIZE)]
    elif token in PRE_EMBEDDING:
        arr = PRE_EMBEDDING[token]
    elif token.lower() in PRE_EMBEDDING:
        arr = PRE_EMBEDDING[token.lower()]
    elif s in PRE_EMBEDDING:
        arr = PRE_EMBEDDING[s]
    elif s.isdigit():
        arr = PRE_EMBEDDING["1"]

    return np.asarray(arr) if len(arr) > 0 else randV


def get_input(FILE_NAME):
    word = []
    sentence, sentence_tag = [], []

    # get max words in sentence
    max_sentence_length = MAX_DOCUMENT_LENGTH  # findMaxLenght(FILE_NAME)
    sentence_length = 0

    print("max sentence size is : " + str(max_sentence_length))

    for line in open(FILE_NAME, 'r'):
        if line in ['\n', '\r\n']:
            # print("aa"+str(sentence_length) )
            for _ in range(max_sentence_length - sentence_length):
                temp = get_embedding("~#~")
                word.append(temp)

            sentence.append(word)
            # print(len(word))
            sentence_tag.append(np.asarray(tag))

            sentence_length = 0
            word = []
            tag = []


        else:
            if sentence_length >= max_sentence_length:
                continue
            sentence_length += 1
            temp = get_embedding(line.split()[0])
            temp = np.append(temp, get_pos(line.split()[1]))  # adding pos embeddings
            word.append(temp)
            convert_tag(line)

    return np.asarray(sentence), sentence_tag


def convert_tag(line):
    """
    Eight classes: 0-None, 1-Company, 2-Date, 3-Event, 4-Location, 5-Money, 6-Product, 7-People
    :param line:
    :return:
    """
    tag = []
    ner = line.split()[2]
    if ner.endswith('O'):
        tag.append(np.asarray([1, 0, 0, 0, 0, 0, 0, 0]))
    elif ner.endswith('COM'):
        tag.append(np.asarray([0, 1, 0, 0, 0, 0, 0, 0]))
    elif ner.endswith('DAT'):
        tag.append(np.asarray([0, 0, 1, 0, 0, 0, 0, 0]))
    elif ner.endswith('EVT'):
        tag.append(np.asarray([0, 0, 0, 1, 0, 0, 0, 0]))
    elif ner.endswith('GPE'):
        tag.append(np.asarray([0, 0, 0, 0, 1, 0, 0, 0]))
    elif ner.endswith('MON'):
        tag.append(np.asarray([0, 0, 0, 0, 0, 1, 0, 0]))
    elif ner.endswith('PDT'):
        tag.append(np.asarray([0, 0, 0, 0, 0, 0, 1, 0]))
    elif ner.endswith('PPL'):
        tag.append(np.asarray([0, 0, 0, 0, 0, 0, 0, 1]))
    else:
        print("error in input" + str(ner))
    return tag


##############################################################################


def pipeline_rnn_train(train_f, test_f, model_f, dict_conf, tfdf_f, tfidf_f, city_f, com_single_f, com_suffix_f,
                       country_f, name_f):
    train_data, test_data = process_annotated(train_f), process_annotated(test_f)
    loads = batch_rnn_loading(dict_conf, city_f, com_single_f, com_suffix_f, country_f, name_f)
    conf, city, com_single, com_suffix, country, name = loads
    train_sents = batch_add_features_rnn(train_data, city, com_single, com_suffix, country, name)
    test_sents = batch_add_features_rnn(test_data, city, com_single, com_suffix, country, name)




    # result, details = test_crf_prediction(crf, X_test, y_test)
    # print(get_now(), 'predict')
    # jl.dump(crf, model_f)
    # return crf, result, details
