# -*- coding: utf-8 -*-


from __future__ import division, print_function, unicode_literals

import argparse
import sys
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


def chunk_ending(prevTag, tag, prevType, type):
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
    return ((prevTag == "B" and tag == "B") or
            (prevTag == "B" and tag == "O") or
            (prevTag == "I" and tag == "B") or
            (prevTag == "I" and tag == "O") or

            (prevTag == "E" and tag == "E") or
            (prevTag == "E" and tag == "I") or
            (prevTag == "E" and tag == "O") or
            (prevTag == "I" and tag == "O") or

            (prevTag != "O" and prevTag != "." and prevType != type) or
            (prevTag == "]" or prevTag == "["))


def chunk_beginning(prevTag, tag, prevType, type):
    """
    # chunk_beginning: checks if a chunk started between the previous and current word
    # arguments:    previous and current chunk tags, previous and current types
    # note:         this code is capable of handling other chunk representations
    #               than the default CoNLL-2000 ones, see EACL'99 paper of Tjong
    #               Kim Sang and Veenstra http://xxx.lanl.gov/abs/cs.CL/9907006

    checks if a chunk started between the previous and current word;
    arguments:  previous and current chunk tags, previous and current types
    
    # corrected 1998-12-22: these chunks are assumed to have length 1

    # print("chunk_beginning?", prevTag, tag, prevType, type)
    # print(chunk_beginning)
    """
    chunk_beginning = ((prevTag == "B" and tag == "B") or
                       (prevTag == "B" and tag == "B") or
                       (prevTag == "I" and tag == "B") or
                       (prevTag == "O" and tag == "B") or
                       (prevTag == "O" and tag == "I") or

                       (prevTag == "E" and tag == "E") or
                       (prevTag == "E" and tag == "I") or
                       (prevTag == "O" and tag == "E") or
                       (prevTag == "O" and tag == "I") or

                       (tag != "O" and tag != "." and prevType != type) or
                       (tag == "]" or tag == "["))

    return chunk_beginning


def cal_metrics(TP, P, T, percent=True):
    """
    compute overall precision, recall and FB1 (default values are 0.0)
    if percent is True, return 100 * original decimal value
    """
    precision = TP / P if P else 0
    recall = TP / T if T else 0
    FB1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    if percent:
        return 100 * precision, 100 * recall, 100 * FB1
    else:
        return precision, recall, FB1


def split_tag(chunkTag, oTag="O", raw=False):
    """
    Split chunk tag into IOB tag and chunk type;
    return (iob_tag, chunk_type)
    """
    if chunkTag == "O" or chunkTag == oTag:
        tag, type = "O", None
    elif raw:
        tag, type = "B", chunkTag
    else:
        try:
            # split on first hyphen, allowing hyphen in type
            tag, type = chunkTag.split('-', 1)
        except ValueError:
            tag, type = chunkTag, None
    return tag, type


def count_chunk(fileIterator, args):
    """
    Process input in given format and count chunks using the last two columns;
    return correct_chunk, found_guessed, found_correct, correct_tag, tokenCounter
    """
    boundary = "-X-"  # sentence boundary
    delimiter = args.delimiter
    raw = args.raw
    oTag = args.oTag

    correct_chunk = defaultdict(int)  # number of correctly identified chunks
    found_correct = defaultdict(int)  # number of chunks in corpus per type
    found_guessed = defaultdict(int)  # number of identified chunks per type

    tokenCounter = 0  # token counter (ignores sentence breaks)
    correct_tag = 0  # number of correct chunk tags

    lastType = None  # temporary storage for detecting duplicates
    inCorrect = False  # currently processed chunk is correct until now
    lastCorrect, lastCorrectType = "O", None  # previous chunk tag in corpus
    lastGuessed, lastGuessedType = "O", None  # previously identified chunk tag

    for line in fileIterator:
        # each non-empty line must contain >= 3 columns
        features = line.strip().split(delimiter)
        if not features or features[0] == boundary:
            features = [boundary, "O", "O"]
        elif len(features) < 3:
            raise IOError("conlleval: unexpected number of features in line %s\n" % line)

        # extract tags from last 2 columns
        guessed, guessedType = split_tag(features[-1], oTag=oTag, raw=raw)
        correct, correctType = split_tag(features[-2], oTag=oTag, raw=raw)

        # 1999-06-26 sentence breaks should always be counted as out of chunk
        firstItem = features[0]
        if firstItem == boundary:
            guessed, guessedType = "O", None

        # decide whether current chunk is correct until now
        if inCorrect:
            endOfGuessed = chunk_ending(lastCorrect, correct, lastCorrectType, correctType)
            endOfCorrect = chunk_ending(lastGuessed, guessed, lastGuessedType, guessedType)
            if (endOfGuessed and endOfCorrect and lastGuessedType == lastCorrectType):
                inCorrect = False
                correct_chunk[lastCorrectType] += 1
            elif (endOfGuessed != endOfCorrect or guessedType != correctType):
                inCorrect = False

        startOfGuessed = chunk_beginning(lastGuessed, guessed, lastGuessedType, guessedType)
        startOfCorrect = chunk_beginning(lastCorrect, correct, lastCorrectType, correctType)
        if (startOfCorrect and startOfGuessed and guessedType == correctType):
            inCorrect = True
        if startOfCorrect:
            found_correct[correctType] += 1
        if startOfGuessed:
            found_guessed[guessedType] += 1

        if firstItem != boundary:
            if correct == guessed and guessedType == correctType:
                correct_tag += 1
            tokenCounter += 1

        lastGuessed, lastGuessedType = guessed, guessedType
        lastCorrect, lastCorrectType = correct, correctType

    if inCorrect:
        correct_chunk[lastCorrectType] += 1

    return correct_chunk, found_guessed, found_correct, correct_tag, tokenCounter


def evaluate(correct_chunk, found_guessed, found_correct):
    # sum counts
    correct_chunkSum = sum(correct_chunk.values())
    found_guessedSum = sum(found_guessed.values())
    found_correctSum = sum(found_correct.values())

    # sort chunk type names
    sorted_type = list(found_correct) + list(found_guessed)
    sorted_type = list(set(sorted_type))
    sorted_type.sort()


    # compute overall precision, recall and FB1 (default values are 0.0)
    precision, recall, FB1 = cal_metrics(correct_chunkSum, found_guessedSum, found_correctSum)
    # print overall performance
    print("processed %i tokens with %i phrases; " % (tokenCounter, found_correctSum), end='')
    print("found: %i phrases; correct: %i.\n" % (found_guessedSum, correct_chunkSum), end='')
    if tokenCounter:
        print("accuracy: %6.2f%%; " % (100 * correct_tag / tokenCounter), end='')
        print("precision: %6.2f%%; recall: %6.2f%%; FB1: %6.2f" %
              (precision, recall, FB1))

    for i in sorted_type:
        precision, recall, FB1 = cal_metrics(correct_chunk[i], found_guessed[i], found_correct[i])
        print("%17s: " % i, end='')
        print("precision: %6.2f%%; recall: %6.2f%%; FB1: %6.2f" %
              (precision, recall, FB1), end='')
        print("  %d" % found_guessed[i])




if __name__ == "__main__":
    args = parse_args()
    # process input and count chunks
    correct_chunk, found_guessed, found_correct, correct_tag, tokenCounter = count_chunk(sys.stdin, args)

    # compute metrics and print
    evaluate(correct_chunk, found_guessed, found_correct)

    sys.exit(0)
