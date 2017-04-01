# -*- coding: utf-8 -*-

import re
import numpy as np
import tflearn
from .pipeline_crf import *
from tensorflow.contrib.rnn import MultiRNNCell, GRUCell, static_bidirectional_rnn
import tensorflow as tf
from copy import deepcopy


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
    vector = get_embedding(token)
    # vector = spacy_parser(token, 'vec', '')
    return np.append(onehot, vector)


def convert_pos(tag):
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


def convert_ner(ner):
    onehot = np.zeros(8)
    if ner.endswith('O'):
        onehot[0] = 1
    elif ner.endswith('COM'):
        onehot[1] = 1
    elif ner.endswith('DAT'):
        onehot[2] = 1
    elif ner.endswith('EVT'):
        onehot[3] = 1
    elif ner.endswith('GPE'):
        onehot[4] = 1
    elif ner.endswith('MON'):
        onehot[5] = 1
    elif ner.endswith('PDT'):
        onehot[6] = 1
    elif ner.endswith('PPL'):
        onehot[7] = 1
    return onehot


##############################################################################


def convert_length(target):
    used = tf.sign(tf.reduce_max(tf.abs(target), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length


def cost(prediction, target):
    target = tf.reshape(target, [-1, MAX_DOCUMENT_LENGTH, num_classes])
    prediction = tf.reshape(prediction, [-1, MAX_DOCUMENT_LENGTH, num_classes])
    cross_entropy = target * tf.log(prediction)
    cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
    mask = tf.sign(tf.reduce_max(tf.abs(target), reduction_indices=2))
    cross_entropy *= mask
    cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
    cross_entropy /= tf.cast(convert_length(target), tf.float32)
    return tf.reduce_mean(cross_entropy)


def build_rnn():
    net = tflearn.input_data([None, MAX_DOCUMENT_LENGTH, EMBEDDING_SIZE])
    net = static_bidirectional_rnn(MultiRNNCell([GRUCell(256)] * 3), MultiRNNCell([GRUCell(256)] * 3),
                                   tf.unstack(tf.transpose(net, perm=[1, 0, 2])),
                                   dtype=tf.float32)  # 256=num_hidden, 3=num_layers
    net = tflearn.dropout(net[0], 0.5)
    net = tf.transpose(tf.stack(net), perm=[1, 0, 2])

    net = tflearn.fully_connected(net, MAX_DOCUMENT_LENGTH * num_classes, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', loss=cost)

    model = tflearn.DNN(net, clip_gradients=0., tensorboard_verbose=0)
    return  model


def fill_zeros(vec, length, dim):
    result = deepcopy(vec)
    diff = length - len(result)
    print(diff)
    return result.append(np.zeros(dim) * diff) if diff > 0 else result


##############################################################################


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


##############################################################################


def batch_add_features_rnn(sents, city, com_single, com_suffix, country, name):
    feature_result, ner_result, token_result = [], [], []
    max_length = max(len(sent) for sent in sents)
    for sent in sents:
        diff = max_length - len(sent)  # get the max length for filing
        sent = sent + [('0', '0', '0') for i in range(diff)] if diff > 0 else sent  # filling according to max length
        sent_vec, pos_vec, ner_vec, feature_vec, token_vec = [], [], [], [], []
        for token, pos, ner in sent:
            # print(get_now(), 'start')
            token_vec.append(token)
            pos_vec.append(convert_pos(pos)), ner_vec.append(convert_ner(ner))
            # print(get_now(), 'end')
            sent_vec.append(add_features_rnn(token, city, com_single, com_suffix, country, name))
            # print(get_now(), 'final')
        final_vec = [np.asarray(np.append(m, n)) for m, n in zip(sent_vec, pos_vec)] # Merge two vectors
        feature_result.append(final_vec), ner_result.append(ner_vec)
        token_result.append(token_vec)
    return np.asarray(feature_result), np.asarray(ner_result), token_result


##############################################################################


def pipeline_rnn_train(train_f, test_f, model_f, dict_conf, tfdf_f, tfidf_f, city_f, com_single_f, com_suffix_f,
                       country_f, name_f):
    loads = batch_rnn_loading(dict_conf, city_f, com_single_f, com_suffix_f, country_f, name_f)
    conf, city, com_single, com_suffix, country, name = loads

    train_data, test_data = process_annotated(train_f), process_annotated(test_f)
    X_train, y_train, token_train = batch_add_features_rnn(train_data, city, com_single, com_suffix, country, name)
    X_test, y_test, token_test = batch_add_features_rnn(test_data, city, com_single, com_suffix, country, name)

