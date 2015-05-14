import cPickle
import sqlite3
import argparse
import numpy as np
import pandas as pd

from segment_datasets import *
from helpers.partial_complement import create_partial_complement
from db_mapping import up_parsing

hqi_db = "../../data/hqi.sqlite"
#window_classifier = "gbc_2000_10_w27.pkl"

pdb_db = "../../data/interactions.sqlite"

protein_list = "../../data/uniprot_bacteria.list"
ppi_list = "../../data/pdb_interactions.list"

up_sequences_file = "../../data/all_bacteria_sequences.json"
up_interactions_file = "../../data/up_interactions.pkl"

up_sequences = {}

MIN_SEQ_LEN = 50

class TooShortSequenceError(Exception):
    pass

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

def get_sequence_structure_pdb(uid1, uid2, pdb_cur, up_parser):

    # If it is a positive pair, we should use it as the source of the structure.
    pdb_cur.execute("""SELECT p1_seq, p1_struct, p2_seq, p2_struct
                    FROM `protein-protein`
                    WHERE p1_uni_id = (?) AND p2_uni_id = (?)""",
                    (uid1, uid2))
    r = pdb_cur.fetchone()
    if r:
        return r

    pdb_cur.execute("""SELECT p2_seq, p2_struct, p1_seq, p1_struct
                    FROM `protein-protein`
                    WHERE p2_uni_id = (?) AND p1_uni_id = (?)""",
                    (uid1, uid2))
    r = pdb_cur.fetchone()
    if r:
        return r

    def get_single_uid(uid):
        pdb_cur.execute("SELECT p1_seq, p1_struct FROM `protein-protein` WHERE p1_uni_id = (?)",
                        (uid,))
        r = pdb_cur.fetchone()
        if r:
            return r

        pdb_cur.execute("SELECT p2_seq, p2_struct FROM `protein-protein` WHERE p2_uni_id = (?)",
                        (uid,))
        r = pdb_cur.fetchone()
        if r:
            return r

        print("No database entry for {0}, downloading from Uniprot".format(uid))
        entry = up_parser.get_uniprot_entry(uid)
        if entry:
            return (entry['sequence'], ''.join('-'*len(entry['sequence'])))
        else:
            print("Skipping entry " + up_id)
            return None

    ss1 = get_single_uid(uid1)
    ss2 = get_single_uid(uid2)

    return list(ss1)+list(ss2)


def load_data(ppi_list, pdb_db, uid2secstr_file=None):

    positives_set = set()
    negatives_set = set()
    positives = []
    negatives = []

    if uid2secstr_file:
        uid2secstr = pd.read_csv(uid2secstr_file, header=None, index_col=0)
        def get_sequence_structure(uid1, uid2):
            return tuple(list(uid2secstr.loc[uid1]) +
                         list(uid2secstr.loc[uid2]))
    else:
        def get_sequence_structure(uid1, uid2):
            return get_sequence_structure_pdb(uid1, uid2, pdb_cur, up_parser)

    with open(up_interactions_file) as f:
        up_interactions = cPickle.load(f)

    up_parser = up_parsing.UPParser()

    pdb_conn = sqlite3.connect(pdb_db)
    pdb_cur = pdb_conn.cursor()

    interactions = pd.read_csv(ppi_list)
    for r in interactions.iterrows():
        a, b = r[1]['p1_uid'], r[1]['p2_uid']
        if a == b:
            next

        # Assert that pair is in lexicographic order
        if a > b:
            a, b = b, a

        a_seq, a_struct, b_seq, b_struct = get_sequence_structure(a, b)
        if (a_seq and b_seq and len(a_seq) >= MIN_SEQ_LEN and\
            len(b_seq) >= MIN_SEQ_LEN and (a,b) not in positives_set):
            positives.append((a, b, a_seq, b_seq, a_struct, b_struct))
            positives_set.add((a,b))


    # Negatives from partial complement graph. !!! Unbiased but... not
    # publishable !!!
    neg_interactions = create_partial_complement(interactions.values)
    #
    for a,b in neg_interactions:
    #.........

    # Negatives only from randomly paired proteins from positive set.
    #up_ids = (list(interactions['p1_uid']) + list(interactions['p2_uid']))
    #up_ids = list(set(list(interactions['p1_uid']) + list(interactions['p2_uid'])))

    #random.seed(41)

    #neg_interactions = []
    #for i in xrange(len(positives)):
    #    a = choice(up_ids)
    #    b = choice(up_ids)

        if a == b:
            next

        # Assert that pair is in lexicographic order
        if a > b:
            a, b = b, a

        if (a,b) in up_interactions or (b,a) in up_interactions:
            next
        if (a,b) in positives_set or (b,a) in positives_set:
            next

        a_seq, a_struct, b_seq, b_struct = get_sequence_structure(a, b)
        if (a_seq and b_seq and len(a_seq) >= MIN_SEQ_LEN and\
            len(b_seq) >= MIN_SEQ_LEN and (a,b) not in negatives_set):
            negatives.append((a, b, a_seq, b_seq, a_struct, b_struct))
            negatives_set.add((a,b))
    #        neg_interactions.append((a,b))

    pdb_conn.close()

    return (positives, negatives), np.vstack((interactions.values, np.array(neg_interactions)))

def save_pos_neg_data(data, interactions, filename, interactions_filename):
    with open(filename, 'wb') as cfile:
        cPickle.dump(data, cfile)
    with open(interactions_filename, 'wb') as cfile:
        cPickle.dump(interactions, cfile)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generating data set of positive and negative protein pairs.')
    parser.add_argument('data_file', help='file in with prepared data will be saved')
    parser.add_argument('interactions_file', help='file in with interaction data will be saved')
    parser.add_argument('--ppi_list', type=str, help='use different protein interactions list')
    parser.add_argument('--ppi_csv', type=str, help='load protein interacting and non-interacting sequences from CSV file')
    parser.add_argument('--all_proteins', type=str, help='use custom list of all proteins')
    parser.add_argument('--all_sequences', type=str, help='use custom list of all protein sequences')
    parser.add_argument('--all_interactions', type=str, help='use custom list of all protein interacations')
    parser.add_argument('--pdb_db', type=str, help='database with protein interactions from PDB complexes')

    args = parser.parse_args()

    if args.pdb_db:
        pdb_db = args.pdb_db

    data = None
    if args.ppi_csv:
        data = load_csv_data(args.ppi_csv)
    else:
        if args.ppi_list:
            ppi_list = args.ppi_list
        if args.all_proteins:
            protein_list = args.all_proteins
        if args.all_sequences:
            up_sequences_file = args.all_sequences
        if args.all_interactions:
            up_interactions_file = args.all_interactions
        data, interactions = load_data(ppi_list, pdb_db)
    save_pos_neg_data(data, interactions, args.data_file, args.interactions_file)
