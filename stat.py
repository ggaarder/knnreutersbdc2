import math
import itertools
import operator
import pandas as pd
import reuters_util as util

def simple_exam(actual, expected):
    """return correct_cnt, all_question_cnt"""
    if len(actual) != len(expected): raise OverflowError()

    correct_cnt = len([None for i, j in zip(actual, expected)
        if i == j])
    quiz_sum = len(actual)

    return correct_cnt, quiz_sum

if __name__ == '__main__':
    expected = [i for i,_ in util.read_csv('test.csv')]

    with open(input('result> ')) as r:
        actual = [i for i in r.read().split('\n') if i]

    print('ACCURACY: {}'.format(
        operator.truediv(*simple_exam(actual, expected))))
    print('Macro, Micro F1: {}, {}'.format(
            *macro_micro_f1(actual, expected)))