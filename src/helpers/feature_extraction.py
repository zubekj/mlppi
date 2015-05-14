import os
import cPickle
import json
import sqlite3
from itertools import combinations, imap, chain, product
from random import sample, choice
import numpy as np
from scipy import ndimage
import math
from collections import defaultdict

import aaindex

from db_mapping.up_parsing import UPParser


hqi_db = "hqi.sqlite"

# liu_indices: hydrophobicity, hydrophicility, volumes of side chains of amino acids, polarity, polarizability, solvent-accessible surface area, net charge index
liu_indices = ["ARGP820101", "HOPT810101", "KRIW790101", "GRAR740102", "CHAM820101", "ROSG850101", "KLEP840101"]
hqi8_indices = ["BLAM930101", "BIOV880101", "MAXF760101", "TSAJ990101", "NAKH920108", "CEDJ970104", "LIFS790101", "MIYS990104"]

aaindex.init(path='.', index='1')
aai_recs = [aaindex.get(d) for d in liu_indices]
hqi8_recs = [aaindex.get(d) for d in hqi8_indices]

def get_aaindex_feature(amino_acid, aai_record):
    # B and Z are ambiguous amino acids.
    if amino_acid == "B":
        val = (aai_record.get("D") + aai_record.get("N")) / 2
    elif amino_acid == "Z":
        val = (aai_record.get("E") + aai_record.get("Q")) / 2
    elif amino_acid == "O":
        val = aai_record.get("K")
    elif amino_acid == "U":
        val = aai_record.get("C")
    elif amino_acid in "X*-":
        val = 0.0
    else:
        val = aai_record.get(amino_acid)
    # Checking for "None" type in case of an unspecified amino acid character.
    if type(val) != type(0.0):
        print("Unrecognised amino acid symbol " + amino_acid)
        exit(-1)
    return val

def encode_aaindex_liu_features(sequence):
    return np.array([[get_aaindex_feature(aa, r) for r in aai_recs]
        for aa in sequence], dtype=np.float_)

# Class for sequence-based features extraction.
class SequenceFeatures(object):

    def get_hqi8_vector(self, ac):
        return [get_aainex_feature(ac, r) for r in hqi8_recs]

    def calculate_quasi_residue_couples(self, seq, l):
        freqs = {(i, ''.join(k)): 0
                    for k in product(''.join(self.hqi8_data.keys()), repeat=2)
                    for i in xrange(l)}
        for d in xrange(l):
            for i in xrange(len(seq)-d):
                freqs[d, seq[i]+seq[i+d]] += 1
        for d, k in freqs:
            freqs[d, k] = float(freqs[d, k])/(len(seq)-d)
        return freqs

    def extract_quasi_residue_couples(self, seq_a, seq_b, l):
        qrc_a = self.calculate_quasi_residue_couples(seq_a, l)
        qrc_b = self.calculate_quasi_residue_couples(seq_b, l)

        return list(chain((qrc_a[k] for k in sorted(qrc_a.keys())),
                (qrc_b[k] for k in sorted(qrc_b.keys()))))

    def calculate_aac_freqs(self, seq):
        freqs = {k: 0 for k in self.hqi8_data.keys()}
        #freqs = {k: 0 for k in "GHITEBS-"}
        for a in seq:
            freqs[a] += 1
        for k in freqs:
            freqs[k] = float(freqs[k])/len(seq)
        return freqs

    def calculate_pseudo_aac(self, seq, l):
        v = np.array(map(self.get_hqi8_vector, seq))
        return np.array([np.mean((v[:-d,:]-v[d:,:])**2) for d in xrange(1,l)])

    def extract_pseudo_aac(self, seq_a, seq_b, l):
        dev_a = self.calculate_pseudo_aac(seq_a, l)
        dev_b = self.calculate_pseudo_aac(seq_b, l)

        return list(chain(self.extract_aac(seq_a, seq_b), dev_a.flat, dev_b.flat))

    def extract_aac(self, seq_a, seq_b):
        aac_a = self.calculate_aac_freqs(seq_a)
        aac_b = self.calculate_aac_freqs(seq_b)

        return list(chain((aac_a[k] for k in sorted(aac_a.keys())),
                (aac_b[k] for k in sorted(aac_b.keys()))))

    def extract_aac_hqi8(self, seq_a, seq_b):
        aac_a = self.calculate_aac_freqs(seq_a)
        aac_b = self.calculate_aac_freqs(seq_b)

        return list(chain(
                 np.array([np.array(self.get_hqi8_vector(k))*aac_a[k] for k in sorted(aac_a.keys())]).flat,
                 np.array([np.array(self.get_hqi8_vector(k))*aac_b[k] for k in sorted(aac_b.keys())]).flat))

    def calculate_2grams_freqs(self, seq):
        freqs = {''.join(k): 0 for k in product(''.join(self.hqi8_data.keys()), repeat=2)}
        #freqs = {''.join(k): 0 for k in product("GHITEBS-", repeat=2)}
        for i in xrange(len(seq)-1):
            freqs[seq[i]+seq[i+1]] += 1
        for k in freqs:
            freqs[k] = float(freqs[k])/(len(seq)+1)
        return freqs

    def extract_2grams(self, seq_a, seq_b):
        _2grams_a = self.calculate_2grams_freqs(seq_a)
        _2grams_b = self.calculate_2grams_freqs(seq_b)

        return list(chain((_2grams_a[k] for k in sorted(_2grams_a.keys())),
            (_2grams_b[k] for k in sorted(_2grams_b.keys()))))

    def extract_flat_features(self, seq_a, seq_b, l):
        va = encode_aaindex_liu_features(seq_a)
        vb = encode_aaindex_liu_features(seq_b)

        mean_a = np.mean(va * va, axis=0)
        dev_a = np.array([np.mean(va[:-d,:] * va[d:,:], axis=0) for d in xrange(1,l)])
        mean_b = np.mean(vb * vb, axis=0)
        dev_b = np.array([np.mean(vb[:-d,:] * vb[d:,:], axis=0) for d in xrange(1,l)])

        return list(chain(mean_a.flat, dev_a.flat, mean_b.flat, dev_b.flat))

# Class for extracting features using computer vision methods.
class ImageFeatures(object):

    def h_index(self, vec):
        vec = sorted(vec)[::-1]
        for i in xrange(1,len(vec)-1):
            if vec[i-1] < i:
                return i-1
        return len(vec)-1

    def g_index(self, vec):
        vec = sorted(vec)[::-1]
        s = 0
        g = 0
        for i in xrange(1,len(vec)-1):
            s += vec[i-1]
            if s >= i*i:
                g = i
        return g

    def h2_mindex(self, matrix):
        return self.h_index([self.h_index(row) for row in matrix])

    def g2_mindex(self, matrix):
        return self.h_index([self.g_index(row) for row in matrix])

    def pap_metrics(self, matrix):
        matrix = matrix.copy()
        matrix -= matrix.min(axis=1)[:, np.newaxis]
        matrix /= matrix.max(axis=1)[:, np.newaxis]
        matrix *= matrix.shape[1]
        return np.array([self.h2_mindex(matrix), self.h2_mindex(matrix.T),
            self.g2_mindex(matrix), self.g2_mindex(matrix.T),
            matrix.shape[0], matrix.shape[1]])

    def partition(self, matrix, depth=1):
        n,m = matrix.shape

        aa = matrix[n/2:,m/2:]
        ab = matrix[:n/2,m/2:]
        ba = matrix[n/2:,:m/2]
        bb = matrix[:n/2,:m/2]

        hr_mean1, vr_mean1 = ndimage.measurements.center_of_mass(matrix)
        mangle = math.atan2(vr_mean1, hr_mean1)
        if np.isnan(mangle):
            mangle = 0

        s = matrix.sum()
        if s == 0:
            s = 1
        #arr = [aa.sum()/s, ab.sum()/s, ba.sum()/s, bb.sum()/s]
        arr = [(aa.sum() + ab.sum() - ba.sum() - bb.sum())/s,
               (aa.sum() + ba.sum() - ab.sum() - bb.sum())/s,
               mangle]

        if depth > 0:
            return arr + self.partition(aa, depth-1) + \
                   self.partition(ab, depth-1) + self.partition(ba, depth-1) + \
                   self.partition(bb, depth-1)
        else:
            return arr

    def randomized_hline_detection(self, matrix, N=10):
        pvals = []
        for i in xrange(matrix.shape[0]):
            cmean = np.mean(matrix[i,:])
            count = 0
            for k in xrange(N):
                rline = []
                for j in xrange(matrix.shape[1]):
                    rline.append(np.random.choice(matrix[:,j]))
                m = np.mean(rline)
                if m >= cmean:
                   count += 1
            pvals.append(float(count)/N)
        return np.array(pvals)

    def randomized_vline_detection(self, matrix):
        return self.randomized_hline_detection(matrix.T)

    def find_lines_plus(self, matrix, t=0.1):

        hind = np.where(matrix.mean(axis=1) - matrix.mean() <= t)[0]
        vind = np.where(matrix.mean(axis=0) - matrix.mean() <= t)[0]

        m = matrix[hind[0]:hind[-1], vind[0]:vind[-1]]

        horizontal_lines = np.argsort(m.mean(axis=1))[::-1][:int(m.shape[0]*0.1)]
        vertical_lines = np.argsort(m.mean(axis=0))[::-1][:int(m.shape[1]*0.1)]
        horizontal_lines.sort()
        vertical_lines.sort()

        #horizontal_lines = np.argsort(m.mean(axis=1))[::-1][:10]
        #vertical_lines = np.argsort(m.mean(axis=0))[::-1][:10]

        m1 = np.zeros(m.shape)
        m1[horizontal_lines,:] = m[horizontal_lines,:]
        m1[:,vertical_lines] = m[:,vertical_lines]

        return (horizontal_lines, vertical_lines, m1)

    def find_lines_border(self, matrix, t=0.1):

        # Detecting lines
        horizontal_lines = np.where(matrix.mean(axis=1) - matrix.mean() > t)[0]
        vertical_lines = np.where(matrix.mean(axis=0) - matrix.mean() > t)[0]

        #th1 = 0.023
        #th = 0.9
        #horizontal_lines = ((matrix > th1).sum(1) > th*matrix.shape[1]).nonzero()[0]
        #vertical_lines = ((matrix > th1).sum(0) > th*matrix.shape[0]).nonzero()[0]

        # Remove border
        horizontal_lines = self.trim_leading_trailing(horizontal_lines)
        vertical_lines = self.trim_leading_trailing(vertical_lines)

        m = np.zeros(matrix.shape)
        m[horizontal_lines,:] = matrix[horizontal_lines,:]
        m[:,vertical_lines] = matrix[:,vertical_lines]

        return (horizontal_lines, vertical_lines, m)

    def find_lines(self, matrix, t=0.1):

        # Detecting lines
        horizontal_lines = np.where(matrix.mean(axis=1) - matrix.mean() > t)[0]
        vertical_lines = np.where(matrix.mean(axis=0) - matrix.mean() > t)[0]

        m = np.zeros(matrix.shape)
        m[horizontal_lines,:] = matrix[horizontal_lines,:]
        m[:,vertical_lines] = matrix[:,vertical_lines]

        return (horizontal_lines, vertical_lines, m)

    def find_lines_rand(self, matrix, t=0.1):

        hline_pvals = self.randomized_hline_detection(matrix)
        vline_pvals = self.randomized_vline_detection(matrix)

        # Remove border
        horizontal_lines = self.trim_leading_trailing(np.where(hline_pvals < t)[0])
        vertical_lines = self.trim_leading_trailing(np.where(vline_pvals < t)[0])

        m = np.zeros(matrix.shape)
        m[horizontal_lines,:] = matrix[horizontal_lines,:]
        m[:,vertical_lines] = matrix[:,vertical_lines]

        return (horizontal_lines, vertical_lines, m)

    # max, min, mean, std
    def mmms(self, m):
        m = np.array(m)
        if m.size == 0:
            return [0,0,0,0]
        vec = (np.max(m), np.min(m), np.mean(m), np.std(m))
        return [0 if np.isnan(v) else v for v in vec]

    def line_statistics(self, matrix, horizontal_lines, vertical_lines, m):

        hr_weight = np.sum(matrix[horizontal_lines,:])/np.sum(matrix)
        vr_weight = np.sum(matrix[:,vertical_lines])/np.sum(matrix)

        vr_std = [np.std(matrix[h,:]) for h in horizontal_lines]
        hr_std = [np.std(matrix[:,v]) for v in vertical_lines]

        var = self.mmms(vr_std) +\
              self.mmms(hr_std) +\
              self.mmms(matrix[horizontal_lines,:]) +\
              self.mmms(matrix[:,vertical_lines]) +\
              self.mmms([float(h)/matrix.shape[1] for h in horizontal_lines]) +\
              self.mmms([float(v)/matrix.shape[0] for v in vertical_lines])

        hr_count = float(len(horizontal_lines))/matrix.shape[1]
        vr_count = float(len(vertical_lines))/matrix.shape[0]

        hr_mean, vr_mean = ndimage.measurements.center_of_mass(m)
        hr_mean1, vr_mean1 = ndimage.measurements.center_of_mass(matrix)

        if np.isnan(hr_mean):
            hr_mean = 0.5*matrix.shape[0]
        if np.isnan(vr_mean):
            vr_mean = 0.5*matrix.shape[1]

        #mdist = math.sqrt((hr_mean - hr_mean1)**2 + (vr_mean - vr_mean1)**2)
        #mdist /= matrix.shape[0]*matrix.shape[1]
        #mangle = math.atan2(vr_mean1, hr_mean1)

        #hr_mean1 = abs(hr_mean - hr_mean1)/matrix.shape[0]
        #vr_mean1 = abs(vr_mean - vr_mean1)/matrix.shape[1]
        hr_mean1 = (hr_mean - hr_mean1)/matrix.shape[0]
        vr_mean1 = (vr_mean - vr_mean1)/matrix.shape[1]

        #consecutive_lines = 0
        #for i in xrange(len(horizontal_lines)-1):
        #    if horizontal_lines[i] + 1 == horizontal_lines[i+1]:
        #        consecutive_lines += 1
        #for i in xrange(len(vertical_lines)-1):
        #    if vertical_lines[i] + 1 == vertical_lines[i+1]:
        #        consecutive_lines += 1

        return [hr_weight + vr_weight, hr_count + vr_count]
    #hr_mean1, vr_mean1, matrix.shape[0], matrix.shape[1]]
        #return (hr_weight + vr_weight, hr_count + vr_count, dist_mean, matrix.shape[0], matrix.shape[1])
        #return (hr_weight, vr_weight, hr_count, vr_count, hr_mean1, vr_mean1, matrix.shape[0], matrix.shape[1])
        #return (hr_weight, vr_weight, hr_count, vr_count, hr_mean1, vr_mean1, matrix.shape[0], matrix.shape[1])
        #return (hr_weight, vr_weight, hr_count, vr_count, hr_mean, vr_mean, hr_mean1, vr_mean1, matrix.shape[0], matrix.shape[1])
        #return (hr_weight, vr_weight, vr_std, hr_std, hr_count, vr_count, hr_mean, vr_mean, hr_mean1, vr_mean1, matrix.shape[0], matrix.shape[1])
        #return [hr_mean, vr_mean, hr_mean1, vr_mean1]
        #return [hr_weight, vr_weight, hr_count, vr_count, hr_mean, vr_mean, hr_mean1, vr_mean1, matrix.shape[0], matrix.shape[1]] + var
        #return (hr_weight, vr_weight, vr_std, hr_std, hr_count, vr_count, hr_mstd, vr_mstd, hr_mean, vr_mean, hr_mean1, vr_mean1, matrix.shape[0], matrix.shape[1])
        #return (hr_weight, vr_weight, vr_std, hr_std, hr_count, vr_count, hr_lstd, vr_lstd, hr_mean, vr_mean, hr_mean1, vr_mean1, matrix.shape[0], matrix.shape[1])
        #return (hr_weight, vr_weight, vr_std, hr_std, hr_count, vr_count, hr_lstd, vr_lstd, hr_mean, vr_mean, hr_mean1, vr_mean1, matrix.shape[0], matrix.shape[1])
        #return (hr_weight + vr_weight, hr_count + vr_count, hr_mean, vr_mean, hr_mean1, vr_mean1, matrix.shape[0], matrix.shape[1])

    def trim_leading_trailing(self, l):
        l = sorted(l)
        i = 1
        while i < len(l) and l[i] == l[i-1] + 1:
            i += 1
        l = l[i:]

        j = -1
        while j > -len(l) and l[j] == l[j-1] + 1:
            j -= 1
        l = l[:j]
        return l

# Class for extracting feature vector from prediction matrix.
class FlatFeatures(object):

    def extract_shapes(self, arr):
        if arr.shape[0] < arr.shape[1]:
            r = float(arr.shape[0])/arr.shape[1]
        else:
            r = float(arr.shape[1])/arr.shape[0]
        #shapes = [r, float(arr.shape[0])/1365, float(arr.shape[1])/1365]
        return [r, float(arr.shape[0]+arr.shape[1])/3000]

    def extract_flat_pairs(self, arr, l):
        row_pairs = []
        col_pairs = []

        row_pairs.append(np.mean(arr))
        for d in xrange(1,l):
            a = arr[:,:-d]
            b = arr[:,d:]
            row_pairs.append(np.mean(a*b))
            a = arr[:-d,:]
            b = arr[d:,:]
            col_pairs.append(np.mean(a*b))

        return row_pairs + col_pairs

    def extract_flat_pairs_diag(self, arr, l):
        pairs = []

        for d in xrange(1,l):
            a = arr[:-d,:-d]
            b = arr[d:,d:]
            pairs.append(np.mean(a*b))

        return pairs

    def extract_sum_pairs(self, arr, l):
        row_vec = float(np.sum(arr, axis=0))/arr.shape[0]
        col_vec = float(np.sum(arr, axis=1))/arr.shape[1]
        row_pairs = []
        col_pairs = []

        row_pairs.append(np.mean(row_vec))
        col_pairs.append(np.mean(col_vec))
        for d in xrange(1,l):
            row_pairs.append(np.mean(np.multiply(row_vec[:-d],row_vec[d:])))
            col_pairs.append(np.mean(np.multiply(col_vec[:-d],col_vec[d:])))

        return row_pairs + col_pairs

    def extract_sum_cumulative(self, arr, l):
        row_vec = np.sum(arr, axis=0)/arr.shape[0]
        col_vec = np.sum(arr, axis=1)/arr.shape[1]
        row_pairs = []
        col_pairs = []

        r = row_vec
        c = col_vec
        for d in xrange(1,l):
            r = np.multiply(r[:-1], row_vec[d:])
            row_pairs.append(np.mean(r))
            c = np.multiply(c[:-1], col_vec[d:])
            col_pairs.append(np.mean(c))

        return row_pairs + col_pairs
