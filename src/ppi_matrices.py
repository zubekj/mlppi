import os
import sys
import cPickle
import json
import sqlite3
import argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool
from random import sample, choice
from itertools import combinations, imap, chain
import affinity
from sklearn.externals import joblib
import math

from segment_datasets import *
from helpers.partial_complement import create_partial_complement
from db_mapping import up_parsing

sec_str_annotations = dict((e,i) for i,e in enumerate(['-', 'B', 'E', 'G', 'H', 'I', 'S', 'T']))

class TooShortSequenceError(Exception):
    pass

PROCESSES=8

# Class that processes sequences with predictor I and returns prediction matrices.
class DatasetProcessor(object):

    def __init__(self, window_classifier, encoder, window_length, window_shift=1):
        self.base_classifier = joblib.load(window_classifier)
        self.base_classifier.n_jobs = PROCESSES
        self.window_length = window_length
        self.window_shift = window_shift
        self.encoder = encoder

    def extract_window(self, window_center, sequence):
        prefix = []
        suffix = []
        begin = window_center - self.window_length/2 - 1
        end = window_center + self.window_length/2

        res = []
        if begin < 0:
            res += ['_'] * (-begin)
            begin = 0

        res += sequence[begin:end]

        if end > len(sequence):
            res += ['_'] * (end - len(sequence))

        return res

    def predict(self, p1_seq, p2_seq, p1_struct, p2_struct):

        if len(p1_seq) > len(p1_struct):
            p1_seq = p1_seq[:len(p1_struct)]
            #p1_struct += "-" * (len(p1_seq)-len(p1_struct))
        if len(p2_seq) > len(p2_struct):
            p2_seq = p2_seq[:len(p2_struct)]
            #p2_struct += "-" * (len(p2_seq)-len(p2_struct))

        #p1_struct = [sec_str_annotations[e] for e in p1_struct]
        #p2_struct = [sec_str_annotations[e] for e in p2_struct]

        nrow = int(math.ceil(float(len(p1_seq))/self.window_shift))
        ncol = int(math.ceil(float(len(p2_seq))/self.window_shift))
        vectors = []
        for i in xrange(nrow):
            for j in xrange(ncol):
                w1 = self.extract_window(i*self.window_shift, p1_seq)
                w2 = self.extract_window(j*self.window_shift, p2_seq)
                w1s = self.extract_window(i*self.window_shift, p1_struct)
                w2s = self.extract_window(j*self.window_shift, p2_struct)
                v = self.encoder.encode(w1, w2, w1s, w2s)
                #vectors.append(w1s+w2s)
                vectors.append(v)

        predictions = self.base_classifier.predict_proba(vectors)[:,1]
        probs = np.reshape(predictions, (nrow, ncol))
        return probs

    def prepare_data(self, pos_neg_data):
        positives, negatives = pos_neg_data

        for p in chain(positives, negatives):
            yield [self.predict(p[2], p[3], p[4], p[5])]

        yield np.r_[np.ones(len(positives)), np.zeros(len(negatives))]


    def close(self):
        self.encoder.close()

# Loads sequences from CSV file.
def load_csv_data(filename):
    positives = []
    negatives = []
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            if row[-1] == '1':
                positives.append((row[0], row[1]))
            else:
                negatives.append((row[0], row[1]))
    return (positives, negatives)


def save_dataset(filename, window_classifier, encoder, pos_neg_file, window_length):
    pos_neg_data = load_pos_neg_data(pos_neg_file)
    data_processor = DatasetProcessor(window_classifier, encoder, window_length)
    npfile = open(filename, 'wb')
    for arr in data_processor.prepare_data(pos_neg_data):
        np.save(npfile, arr)
    npfile.close()
    data_processor.close()


def load_pos_neg_data(filename):
    cfile = open(filename, 'rb')
    data = cPickle.load(cfile)
    cfile.close()
    return data


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generating data set of positive and negative protein pairs.')
    parser.add_argument('data_file', help='file in with prepared data will be saved')
    parser.add_argument('window_length', type=int, help='length of the window')
    parser.add_argument('window_predictor', type=str, help="file in with level I predictor is stored")
    parser.add_argument('pos_neg_file', type=str, help="generate predictions based on pos_neg file")

    args = parser.parse_args()

    save_dataset(args.data_file, args.window_predictor, StructEncoder(), args.pos_neg_file, args.window_length)
