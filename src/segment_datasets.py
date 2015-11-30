import sys
import sqlite3
import argparse
from itertools import imap, izip, chain
import pandas as pd
import numpy as np
import random
from collections import defaultdict
from sklearn.cross_validation import train_test_split

from helpers import aaindex

hqi_db = "../../data/hqi.sqlite"

contact_aaindices = ["TANS760101", "TANS760102", "ROBB790102", "BRYS930101", "THOP960101"]

class AAIndexContactEncoder(object):

    def __init__(self, window_length, aaindices):
        self.variables = [n + str(w) for w in xrange(window_length) for n in aaindices]
        #self.variables = [n + str(w) for w in xrange(window_length) for n in aaindices] +\
        #                 [n + str(w) + "1" for w in xrange(window_length-3) for n in aaindices] +\
        #                 [n + str(w) + "2" for w in xrange(window_length-3) for n in aaindices]
        #self.variables = [n for n in aaindices]
        aaindex.init(path='.')
        self.aaindices = [aaindex.get(ind) for ind in aaindices]

    def encode(self, seq1, seq2, struct1, struct2):
        def getaa(a, b, aaind):
            r = aaind.get(a, b)
            if r is None:
                r = 0.0
            return r

        #return [np.mean([getaa(a, b, aaind) for a in seq1[7:15] for b in seq2[7:15]]) for aaind in self.aaindices]
        #return [getaa(a, b, aaind) for a, b in chain(izip(seq1, seq2), izip(seq1[3:], seq2),
        #                                             izip(seq1, seq2[3:]))
        #        for aaind in self.aaindices]
        return [getaa(a, b, aaind) for a, b in izip(seq1, seq2) for aaind in self.aaindices]

    def close(self):
        pass

class HQI8Encoder(object):

    def __init__(self, window_length):
        hqi_conn = sqlite3.connect(hqi_db)
        hqi_cur = hqi_conn.cursor()

        self.indices = {}
        for ac in "ACDEFGHIKLMNPQRSTVWY":
            hqi_cur.execute("SELECT ep,h,atp,pp,rp,c,bp,ip FROM hqi8 WHERE id=(?)", (ac,))
            self.indices[ac] = hqi_cur.fetchone()
        self.indices['_'] = [0.0] * 8
        self.indices['-'] = [0.0] * 8
        self.indices['*'] = [0.0] * 8
        self.indices['X'] = [0.0] * 8
        self.indices['B'] = [0.0] * 8
        self.indices['Z'] = [0.0] * 8
        self.indices['O'] = self.indices['K']
        self.indices['U'] = self.indices['C']

        hqi_conn.close()
        self.variables = [p + n + str(w) for p in ['p1_', 'p2_']
                          for w in xrange(window_length)
                          for n in ['ep', 'h', 'atp', 'pp', 'rp', 'c', 'bp', 'ip']]

    def encode(self, seq1, seq2, struct1, struct2):
        return [i for hqi8_vec in (self.indices[a] for a in chain(seq1, seq2)) for i in hqi8_vec]

    def close(self):
        pass

class StructEncoder(object):

    def __init__(self, window_length):
        self.variables = [p + "struct" + str(w) for p in ['p1_', 'p2_']
                for w in xrange(window_length)]

    def encode(self, seq1, seq2, struct1, struct2):
        sec_str_annotations = dict((e,i) for i,e in enumerate(['_', '-', 'B', 'E', 'G', 'H', 'I', 'S', 'T']))
        return [sec_str_annotations[e] for e in chain(struct1, struct2)]

    def close(self):
        pass

class Struct3LEncoder(object):
    """Encodes secondary structure limiting annotations to 3 PSIPRED symbols"""

    def __init__(self, window_length):
        self.variables = [p + "struct" + str(w) for p in ['p1_', 'p2_']
                for w in xrange(window_length)]

    def encode(self, seq1, seq2, struct1, struct2):
        def transform_symbol(s):
            if s in ['H', 'G','I']:
                return 0
            if s in ['E', 'B']:
                return 1
            if s == "-":
                return 2
            return 3

        return [transform_symbol(s) for s in chain(struct1, struct2)]

    def close(self):
        pass

class SimpleEncoder(object):

    def __init__(self, window_length):
        self.variables = [p + str(w) for p in ['p1_', 'p2_']
                          for w in xrange(window_length)]

    def encode(self, seq1, seq2, struct1, struct2):
        return [ord(c) for c in chain(seq1, seq2)]

    def close(self):
        pass

class CombinedEncoder(object):

    def __init__(self, encoders):
        self.encoders = encoders
        self.variables = ["{0}_{1}".format(i, v) for i, e in enumerate(encoders)
                for v in e.variables]

    def encode(self, seq1, seq2, struct1, struct2):
        return [str(v) for e in self.encoders for v in e.encode(seq1, seq2, struct1, struct2)]

    def close(self):
        for e in self.encoders:
            e.close()

def prepare_table(samples_cur, encoder, interaction_threshold, out_file, uid_filter=None):
    variables = encoder.variables + ["positive"]
    #interaction_threshold = 5
    #interaction_threshold = 1
    #interaction_threshold = 10
    #interaction_threshold = 15
    #interaction_threshold = 20

    with open(out_file, "w") as f:

        f.write(",".join(variables))
        f.write("\n")

        print("Generating positives")
        if uid_filter:
            query = '''SELECT p1_wseq, p2_wseq, p1_wstruct, p2_wstruct FROM pp_samples
                        JOIN interactions ON pp_samples.interaction_id = interactions.id
                        WHERE n_interactions > (?) AND interactions.p1_uni_id IN {0}
                            AND interactions.p2_uni_id IN {0}'''.format(uid_filter)
        else:
            query = "SELECT p1_wseq, p2_wseq, p1_wstruct, p2_wstruct FROM pp_samples WHERE n_interactions > (?)"

        n_positive = 0
        for row in samples_cur.execute(query, (interaction_threshold,)):
            v = encoder.encode(row[0], row[1], row[2], row[3]) + ["1"]
            f.write(",".join([str(e) for e in v]))
            f.write("\n")
            n_positive += 1

        print("Generating negatives")
        if uid_filter:
            query = '''SELECT p1_wseq, p2_wseq, p1_wstruct, p2_wstruct FROM pp_samples
                        JOIN interactions ON pp_samples.interaction_id == interactions.id
                        WHERE (n_interactions = 0 AND interactions.p1_uni_id IN {0}
                            AND interactions.p2_uni_id IN {0})
                        '''.format(uid_filter)
            n_query = '''SELECT count(1) FROM pp_samples
                        JOIN interactions ON pp_samples.interaction_id == interactions.id
                        WHERE (n_interactions = 0 AND interactions.p1_uni_id IN {0}
                            AND interactions.p2_uni_id IN {0})
                        '''.format(uid_filter)
        else:
            query = '''SELECT p1_wseq, p2_wseq, p1_wstruct, p2_wstruct FROM pp_samples WHERE n_interactions = 0'''
            n_query = '''SELECT count(1) FROM pp_samples WHERE n_interactions = 0'''

        samples_cur.execute(n_query)
        n_negative = samples_cur.fetchone()[0]

        random.seed(42)
        chosen_negatives = sorted(random.sample(xrange(n_negative), 3*n_positive))

        i = 0
        for row in samples_cur.execute(query):
            if len(chosen_negatives) > 0 and i == chosen_negatives[0]:
                v = encoder.encode(row[0], row[1], row[2], row[3]) + ["0"]
                f.write(",".join([str(e) for e in v]))
                f.write("\n")
                chosen_negatives = chosen_negatives[1:]
            i += 1

def split_train_test(samples_cur, train_filename, test_filename,
                     train_examples=250):

    # Load full interaction graph
    interactions = set()
    for row in samples_cur.execute("SELECT DISTINCT p1_uni_id, p2_uni_id FROM interactions"):
        if row[0] < row[1]:
            interactions.add((row[0], row[1]))
        else:
            interactions.add((row[1], row[0]))

    nodes = defaultdict(int)
    for pair in interactions:
        for n in pair:
            nodes[n] += 1
    protein_uids = sorted(nodes, key=nodes.__getitem__, reverse=True)

    if train_examples > 1:
        # Simulating old split method
        samples_cur.execute('''SELECT * FROM (SELECT p2_uni_id FROM interactions UNION
                               SELECT p1_uni_id from interactions)
                               LIMIT {0};'''.format(train_examples))
        train_uids = [r[0] for r in samples_cur.fetchall()]
        samples_cur.execute('''SELECT * FROM (SELECT p2_uni_id FROM interactions UNION
                               SELECT p1_uni_id from interactions)
                               LIMIT -1 OFFSET {0};'''.format(train_examples))
        test_uids = [r[0] for r in samples_cur.fetchall()]
    else:
        train_uids, test_uids = train_test_split(protein_uids,
                                                 train_size=train_examples,
                                                 random_state=42)
        train_uids = set(train_uids)
        test_uids = set(test_uids)

    samples_cur.execute('''CREATE TEMP TABLE train_uids(uid CHAR(10) PRIMARY KEY NOT NULL);''')
    for uid in train_uids:
        samples_cur.execute("INSERT INTO train_uids VALUES (?);", (uid,))

    samples_cur.execute('''CREATE TEMP TABLE test_uids(uid CHAR(10) PRIMARY KEY NOT NULL);''')
    for uid in test_uids:
        samples_cur.execute("INSERT INTO test_uids VALUES (?);", (uid,))

    train_interactions = pd.DataFrame([(p1, p2) for p1, p2 in interactions
        if p1 in train_uids and p2 in train_uids])
    try:
        train_interactions.columns = ["p1_uid", "p2_uid"]
    except ValueError:
        pass
    train_interactions.to_csv(train_filename, index=False)

    test_interactions = pd.DataFrame([(p1, p2) for p1, p2 in interactions
        if p1 in test_uids and p2 in test_uids])
    try:
        test_interactions.columns = ["p1_uid", "p2_uid"]
    except ValueError:
        pass
    test_interactions.to_csv(test_filename, index=False)


def create_data_table(samples_db, train_ids_file, test_ids_file, train_file,
                      test_file, encoder, interaction_threshold,
                      filter_uids=False, train_examples=250):
    samples_conn = sqlite3.connect(samples_db)
    samples_cur = samples_conn.cursor()

    if filter_uids:
        split_train_test(samples_cur, train_ids_file, test_ids_file, train_examples)
        prepare_table(samples_cur, encoder, interaction_threshold, train_file, "train_uids")
        prepare_table(samples_cur, encoder, interaction_threshold, test_file, "test_uids")
    else:
        prepare_table(samples_cur, encoder, interaction_threshold, train_file)

    encoder.close()
    samples_conn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generating data table from window samples stored in database.')
    parser.add_argument('samples_db', help='database with extracted pp_samples')
    parser.add_argument('out_file', help='file in with data table will be saved')
    parser.add_argument('-s', '--split', action="store_true", help='split into training and testing with respect to protein uid')
    parser.add_argument('-p', '--psipred_labels', action="store_true", help='use only 3 PSIPRED labels in secondary structure')
    args = parser.parse_args()

    samples_conn = sqlite3.connect(args.samples_db)
    samples_cur = samples_conn.cursor()
    samples_cur.execute("SELECT p1_wseq FROM pp_samples LIMIT 1")
    window_length = len(samples_cur.fetchone()[0])
    samples_conn.close()

    if not args.psipred_labels:
        encoders = (AAIndexContactEncoder(window_length, contact_aaindices),
                    HQI8Encoder(window_length),
                    StructEncoder(window_length),
                    SimpleEncoder(window_length))
    else:
        encoders = (AAIndexContactEncoder(window_length, contact_aaindices),
                    HQI8Encoder(window_length),
                    Struct3LEncoder(window_length),
                    SimpleEncoder(window_length))
    feature_lens = [len(e.variables) for e in encoders]
    feature_ranges = ", ".join("{0}: {1}-{2}".format(e.__class__.__name__, sum(feature_lens[:i]),
                               sum(feature_lens[:i+1])) for i,e in enumerate(encoders))
    encoder = CombinedEncoder(encoders)

    with open(filename + ".features", "w") as f:
        f.write(feature_ranges + "\n")

    create_data_table(args.samples_db, args.out_file+".train_interactions",
            args.out_file+".test_interactions", args.out_file+".train.csv",
            args.out_file+".test.csv", encoder, 15, args.split)
