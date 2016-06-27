import json
import sys
from svmutil import *


def svm_read_problem1(data_file_name):
    """
    svm_read_problem(data_file_name) -> [y, x]

    Read LIBSVM-format data from data_file_name and return labels y
    and data instances x.
    """
    prob_y = []
    prob_x = []
    for line in open(data_file_name):
        lineArr = line.split(' ')
        label = lineArr[-1]
        features = lineArr[0:-1]
        xi = {}
        ind = 0
        for e in features:
            ind += 1
            val = e
            xi[ind] = float(val) / 255
        prob_y += [float(label)]
        prob_x += [xi]

    return (prob_y, prob_x)


def svm_read_problem_json(data_file_name):
    """
    svm_read_problem(data_file_name) -> [y, x]

    Read LIBSVM-format data from data_file_name and return labels y
    and data instances x.
    """
    prob_y = []
    prob_x = []
    for line in open(data_file_name):
        lineArr = line
        label = lineArr[0]
        features = lineArr[1:]
        xi = {}
        f = json.loads(features)
        for (d, x) in f.items():
            xi[int(d)]=x
        prob_y += [float(label)]
        prob_x += [xi]

    return (prob_y, prob_x)

#
y, x = svm_read_problem_json('ptsModified.points')
# y, x = svm_read_problem1("sex_pic.txt")
# m = svm_train(y, x, '-c 7 -g 0.02')

m = svm_train(y, x, '-c 30 -g 0.82 -e 1 -v 5')
# svm_save_model('libsvm.model', m)
