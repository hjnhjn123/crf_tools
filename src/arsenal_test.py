# -*- coding: utf-8 -*-


from __future__ import division, print_function, unicode_literals

import argparse
import sys
import pandas as pd
from collections import defaultdict


# sanity check
def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-r", "--raw",
        default=False, action="store_true",
        help="accept raw result tags"
    )
    argparser.add_argument(
        "-d", "--delimiter",
        default=None,
        help="alternative delimiter tag (default: single space)"
    )
    argparser.add_argument(
        "-o", "--oTag",
        default="O",
        help="alternative delimiter tag (default: O)"
    )
    args = argparser.parse_args()
    return args


"""
• IOB1: I is a token inside a chunk, O is a token outside a chunk and B is the
beginning of chunk immediately following another chunk of the same Named Entity.
• IOB2: It is same as IOB1, except that a B tag is given for every token, which exists at
the beginning of the chunk.
• IOE1: An E tag used to mark the last token of a chunk immediately preceding another
chunk of the same named entity.
• IOE2: It is same as IOE1, except that an E tag is given for every token, which exists at
the end of the chunk.
• START/END: This consists of the tags B, E, I, S or O where S is used to represent a
chunk containing a single token. Chunks of length greater than or equal to two always
start with the B tag and end with the E tag.
• IO: Here, only the I and O labels are used. This therefore cannot distinguish between
adjacent chunks of the same named entity.

"""


def chunk_ending(prev_tag, tag, prev_type, type):
    """
    # chunk_ending: checks if a chunk ended between the previous and current word
    # arguments:  previous and current chunk tags, previous and current types
    # note:       this code is capable of handling other chunk representations
    #             than the default CoNLL-2000 ones, see EACL'99 paper of Tjong
    #             Kim Sang and Veenstra http://xxx.lanl.gov/abs/cs.CL/9907006
    
    checks if a chunk ended between the previous and current word;
    arguments:  previous and current chunk tags, previous and current types
    
    # corrected 1998-12-22: these chunks are assumed to have length 1

    """
    return ((prev_tag == "B" and tag == "B") or
            (prev_tag == "B" and tag == "O") or
            (prev_tag == "I" and tag == "B") or
            (prev_tag == "I" and tag == "O") or

            (prev_tag == "E" and tag == "E") or
            (prev_tag == "E" and tag == "I") or
            (prev_tag == "E" and tag == "O") or
            (prev_tag == "I" and tag == "O") or

            (prev_tag != "O" and prev_tag != "." and prev_type != type) or
            (prev_tag == "]" or prev_tag == "["))


def chunk_beginning(prev_tag, tag, prev_type, type):
    """
    # chunk_beginning: checks if a chunk started between the previous and current word
    # arguments:    previous and current chunk tags, previous and current types
    # note:         this code is capable of handling other chunk representations
    #               than the default CoNLL-2000 ones, see EACL'99 paper of Tjong
    #               Kim Sang and Veenstra http://xxx.lanl.gov/abs/cs.CL/9907006

    checks if a chunk started between the previous and current word;
    arguments:  previous and current chunk tags, previous and current types
    
    # corrected 1998-12-22: these chunks are assumed to have length 1

    # print("chunk_beginning?", prev_tag, tag, prev_type, type)
    # print(chunk_beginning)
    """
    chunk_beginning = ((prev_tag == "B" and tag == "B") or
                       (prev_tag == "B" and tag == "B") or
                       (prev_tag == "I" and tag == "B") or
                       (prev_tag == "O" and tag == "B") or
                       (prev_tag == "O" and tag == "I") or

                       (prev_tag == "E" and tag == "E") or
                       (prev_tag == "E" and tag == "I") or
                       (prev_tag == "O" and tag == "E") or
                       (prev_tag == "O" and tag == "I") or

                       (tag != "O" and tag != "." and prev_type != type) or
                       (tag == "]" or tag == "["))

    return chunk_beginning


def cal_metrics(TP, P, T):
    """
    compute overall precision, recall and f_score (default values are 0.0)
    if percent is True, return 100 * original decimal value
    """
    precision = TP / P if P else 0
    recall = TP / T if T else 0
    f_score = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return 100 * precision, 100 * recall, 100 * f_score


def split_tag(chunk_tag, oTag="O", raw=False):
    """
    Split chunk tag into IOB tag and chunk type;
    return (iob_tag, chunk_type)
    """
    if chunk_tag == "O" or chunk_tag == oTag:
        tag, type = "O", None
    elif raw:
        tag, type = "B", chunk_tag
    else:
        try:
            # split on first hyphen, allowing hyphen in type
            tag, type = chunk_tag.split('-', 1)
        except ValueError:
            tag, type = chunk_tag, None
    return tag, type


def count_chunk(file_iter, delimiter, raw, o_tag, boundary="-X-"):
    """
    Process input in given format and count chunks using the last two columns;
    return correct_chunk, found_guessed, found_correct, correct_tag, token_count
    """

    correct_chunk = defaultdict(int)  # number of correctly identified chunks
    found_correct = defaultdict(int)  # number of chunks in corpus per type
    found_guessed = defaultdict(int)  # number of identified chunks per type

    token_count = 0  # token counter (ignores sentence breaks)
    correct_tag = 0  # number of correct chunk tags

    lastType = None  # temporary storage for detecting duplicates
    in_correct = False  # currently processed chunk is correct until now
    last_correct, last_correct_type = "O", None  # previous chunk tag in corpus
    last_guessed, last_guessed_type = "O", None  # previously identified chunk tag

    for line in file_iter:
        # each non-empty line must contain >= 3 columns
        features = line.strip().split(delimiter)
        if not features or features[0] == boundary:
            features = [boundary, "O", "O"]
        elif len(features) < 3:
            raise IOError("conlleval: unexpected number of features in line %s\n" % line)

        # extract tags from last 2 columns
        guessed, guessed_type = split_tag(features[-1], oTag=o_tag, raw=raw)
        correct, correct_type = split_tag(features[-2], oTag=o_tag, raw=raw)

        # 1999-06-26 sentence breaks should always be counted as out of chunk
        firstItem = features[0]
        if firstItem == boundary:
            guessed, guessed_type = "O", None

        # decide whether current chunk is correct until now
        if in_correct:
            guessed_end = chunk_ending(last_correct, correct, last_correct_type, correct_type)
            correct_end = chunk_ending(last_guessed, guessed, last_guessed_type, guessed_type)
            if (guessed_end and correct_end and last_guessed_type == last_correct_type):
                in_correct = False
                correct_chunk[last_correct_type] += 1
            elif (guessed_end != correct_end or guessed_type != correct_type):
                in_correct = False

        guessed_start = chunk_beginning(last_guessed, guessed, last_guessed_type, guessed_type)
        correct_start = chunk_beginning(last_correct, correct, last_correct_type, correct_type)
        if (correct_start and guessed_start and guessed_type == correct_type):
            in_correct = True
        if correct_start:
            found_correct[correct_type] += 1
        if guessed_start:
            found_guessed[guessed_type] += 1

        if firstItem != boundary:
            if correct == guessed and guessed_type == correct_type:
                correct_tag += 1
            token_count += 1

        last_guessed, last_guessed_type = guessed, guessed_type
        last_correct, last_correct_type = correct, correct_type

    if in_correct:
        correct_chunk[last_correct_type] += 1

    return correct_chunk, found_guessed, found_correct, correct_tag, token_count


def evaluate(correct_chunk, found_guessed, found_correct):
    # sum counts
    correct_chunk_sum = sum(correct_chunk.values())
    found_guessed_sum = sum(found_guessed.values())
    found_correct_sum = sum(found_correct.values())

    # sort chunk type names
    sorted_type = list(found_correct) + list(found_guessed)
    sorted_type = list(set(sorted_type))
    sorted_type.sort()

    # compute overall precision, recall and FB1 (default values are 0.0)
    precision, recall, FB1 = cal_metrics(correct_chunk_sum, found_guessed_sum, found_correct_sum)
    # print overall performance
    print("processed %i tokens with %i phrases; " % (token_count, found_correct_sum), end='')
    print("found: %i phrases; correct: %i.\n" % (found_guessed_sum, correct_chunk_sum), end='')
    if token_count:
        print("accuracy: %6.2f%%; " % (100 * correct_tag / token_count), end='')
        print("precision: %6.2f%%; recall: %6.2f%%; FB1: %6.2f" %
              (precision, recall, FB1))

    result = pd.DataFrame([cal_metrics(correct_chunk[i], found_guessed[i], found_correct[i]) for i in sorted_type])
    return result


if __name__ == "__main__":
    delimiter, raw, o_tag, boundary = parse_args()
    # process input and count chunks
    correct_chunk, found_guessed, found_correct, correct_tag, token_count = count_chunk(sys.stdin, delimiter, raw,
                                                                                        o_tag, boundary)

    # compute metrics and print
    evaluate(correct_chunk, found_guessed, found_correct)

    sys.exit(0)
