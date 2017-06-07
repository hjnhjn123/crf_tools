# -*- coding: utf-8 -*-

import re
from collections import Counter
from copy import deepcopy
from itertools import groupby

import pandas as pd
from sklearn_crfsuite import metrics

RE_WORDS = re.compile(r"[\w\d\.-]+")
HEADER_REPORT = ['tag', 'precision', 'recall', 'f1', 'support']


def extract_entity(ners_list):
    ner_index = (i for i in range(len(ners_list)) if ners_list[i][1][0] == 'U' or ners_list[i][1][0] == 'L')
    new_index = (a + b for a, b in enumerate(ner_index))
    pred_copy = deepcopy(ners_list)
    for i in new_index:
        pred_copy[i + 1:i + 1] = [('##split', '##split')]
    evaluate_list = [list(x[1]) for x in groupby(pred_copy, lambda x: x == ('##split', '##split'))]
    return evaluate_list


def cal_metrics(true_positive, all_positive, T):
    """
    compute overall precision, recall and f_score (default values are 0.0)
    """
    precision = true_positive / all_positive if all_positive else 0
    recall = true_positive / T if T else 0
    f_score = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return round(precision, 4), round(recall, 4), round(f_score, 4)


def evaluate_ner_result(y_pred, y_test):
    """
    :param y_pred: [y_pred] 
    :param y_test: [y_test]
    :return: {}
    """
    test_ners = [i for i in enumerate(y_test) if i[1] != 'O']
    pred_ners = [i for i in enumerate(y_pred) if i[1] != 'O']
    both_ners = [i for i in zip(y_pred, y_test) if i[1] != 'O']

    evaluate_list = extract_entity(both_ners)
    test_entities = extract_entity(test_ners)
    pred_entities = extract_entity(pred_ners)

    true_positive_list = [ner_can for ner_can in evaluate_list if
                          len([(a, b) for a, b in ner_can if a == b]) == len(ner_can) and ner_can != [
                              ('##split', '##split')]]
    test_total = [ner_can for ner_can in test_entities if ner_can != [('##split', '##split')]]
    pred_total = [ner_can for ner_can in pred_entities if ner_can != [('##split', '##split')]]

    true_positive_result = Counter(i[0][0].split('-')[1] for i in true_positive_list)

    relevant_elements = Counter(i[0][1].split('-')[1] for i in test_total)
    selected_elements = Counter(i[0][1].split('-')[1] for i in pred_total)

    final_result = {k: cal_metrics(true_positive_result[k], v, selected_elements[k]) + (v,) for (k, v) in
                    relevant_elements.items()}
    total_result = cal_metrics(sum(true_positive_result.values()), sum(relevant_elements.values()),
                               sum(selected_elements.values()))
    final_result.update({'Total': total_result + (sum(relevant_elements.values()),)})
    output = pd.DataFrame(final_result).T.reset_index()
    output.columns = ['Label', 'Precision', 'Recall', 'F1_score', 'Support']
    return output


##############################################################################


def convert_tags(data):
    converted = []
    for sent in data:
        test_result = []
        for tag in sent:
            if tag == 'O':
                test_result.append('0')
            else:
                test_result.append('1')
        converted.append(test_result)
    return converted


def export_test_result(labels, y_test, y_pred):
    details = metrics.flat_classification_report(y_test, y_pred, digits=3, labels=labels)
    details = [i for i in [re.findall(RE_WORDS, i) for i in details.split('\n')] if i !=
               []][1:-1]
    details = pd.DataFrame(details, columns=HEADER_REPORT)
    details = details.sort_values('f1', ascending=False)
    return details


def show_crf_label(crf, remove_list=['O', 'NER', '']):
    labels = list(crf.classes_)
    return [i for i in labels if i not in remove_list]


def test_crf_prediction(crf, y_pred, y_test, test_switch='spc'):
    """

    :param crf:
    :param y_pred:
    :param y_test:
    :param test_switch: 'spc' for specific labels, 'bin' for binary labels
    :return:
    """
    if test_switch == 'spc':
        labels = show_crf_label(crf)

        result = metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)
        details = export_test_result(labels, y_test, y_pred)
        return result, details

    elif test_switch == 'bin':

        y_pred_converted = convert_tags(y_pred)
        y_test_converted = convert_tags(y_test)
        labels = ['1']

        result = metrics.flat_f1_score(y_test_converted, y_pred_converted,
                                       average='weighted', labels=labels)
        y_test_flatten = ['0' if j == 'O' else '1' for i in y_test for j in i]
        details = export_test_result(labels, y_test_flatten, y_pred_converted)
        return result, details
