import sys
import random
import sqlite3
import simplejson as json
import pandas as pd
from bisect import bisect_left as bisect

# Limits number of negative samples from one protein pair.
NEGATIVE_SAMPLES_LIMIT = 100

# Shortest window that will be used
MIN_WINDOW_LENGTH = 21


class SequenceStructureLenghtMismatchError(Exception):
    """Base class for exceptions in this module."""
    pass


def split_pair_list(pair_list):
    l1 = []
    l2 = []
    for p in pair_list:
        l1.append(p[0])
        l2.append(p[1])
    return l1, l2

def create_pp_samples_table(cursor):
    cursor.execute('''CREATE TABLE `pp_samples`
         (interaction_id INTEGER, p1_resnum int,
          p2_resnum int, p1_wseq text, p2_wseq text,
          p1_wstruct text, p2_wstruct text, n_interactions int,
          PRIMARY KEY(interaction_id, p1_resnum, p2_resnum))''')

def create_interactions_table(cursor):
    cursor.execute('''CREATE TABLE `interactions`
         (id INTEGER PRIMARY KEY, pdb_id text,
          p1_uni_id text, p2_uni_id text,
          UNIQUE (pdb_id, p1_uni_id, p2_uni_id))''')

def extract_window(window_center, sequence, window_length):
    # NOTE: sequence is indexed from 0, but window_center
    #       is in range [1, len(sequence)].
    prefix = ""
    suffix = ""
    if window_center < 0:
        print(window_center)
    begin = window_center - window_length/2 - 1
    end = window_center + window_length/2
    if begin < 0:
        prefix = "_" * (-begin)
        begin = 0
    if end > len(sequence):
        suffix = "_" * (end - len(sequence))
        end = len(sequence)
    return prefix + sequence[begin:end] + suffix

def find_gaps(ilist, seq_length):
    gaps = []
    ilist = [-MIN_WINDOW_LENGTH/2] + ilist + [seq_length+MIN_WINDOW_LENGTH]
    for i in xrange(len(ilist)-1):
        gap_space = ilist[i+1] - ilist[i] - MIN_WINDOW_LENGTH
        gaps += [ilist[i]+j+MIN_WINDOW_LENGTH/2+1 for j in xrange(gap_space)]
    return gaps

# TODO: Refactor this ugly monster.
def assert_interaction_quality(pair, res1, res2, sform1, sform2):

    # Assert that windows are inside sequences' ranges.
    if (pair[0] + MIN_WINDOW_LENGTH/2) < 0 or (pair[0] - MIN_WINDOW_LENGTH/2) > len(sform1):
        return False
    if (pair[1] + MIN_WINDOW_LENGTH/2) < 0 or (pair[1] - MIN_WINDOW_LENGTH/2) > len(sform2):
        return False

    def find_window_interactions(idx, res):
        count = 0
        cntr = res[idx]
        i = idx
        while i < len(res) and res[i] <= cntr + MIN_WINDOW_LENGTH/2:
            count += 1
            i += 1
        i = idx-1
        while i >= 0 and res[i] >= cntr - MIN_WINDOW_LENGTH/2:
            count += 1
            i -= 1
        return count

    count1 = find_window_interactions(bisect(res1, pair[0]), res1)
    count2 = find_window_interactions(bisect(res2, pair[1]), res2)
    return (count1, count2)

def get_interaction_id(ocur, pdb, u1, u2):
    try:
        ocur.execute("INSERT INTO `interactions` VALUES (NULL,?,?,?)",
                     (pdb, u1, u2))
    except sqlite3.IntegrityError:
        pass
    ocur.execute('''SELECT id FROM `interactions` WHERE pdb_id=(?)
                    AND p1_uni_id=(?) AND p2_uni_id=(?)''', (pdb, u1, u2))
    return ocur.fetchone()[0]

def seq_interact(ocur, s):
    ocur.execute('''SELECT interaction_id FROM `pp_samples` WHERE p1_wseq = (?)
                    OR p2_wseq = (?)''', (s, s))
    return bool(ocur.fetchone())

def generate_positive_samples(icur, ocur, oconn, window_length,
                              uid2secstr=None):
    print("Generating positives")

    positive_fragments = set()

    for row in icur.execute('''SELECT pdb_id, p1_uni_id, p1_seq,
                p2_uni_id, p2_seq, p1_struct, p2_struct, interacting_residues
                FROM `protein-protein` WHERE p1_uni_id != p2_uni_id'''):
        interaction_id = get_interaction_id(ocur, row[0], row[1], row[3])
        interacting_residues = json.loads(row[7])
        interacting_residues = set([tuple(t) for t in interacting_residues])
        res1, res2 = split_pair_list(interacting_residues)
        res1.sort()
        res2.sort()

        if uid2secstr is not None:
            try:
                p1_struct = uid2secstr.loc[row[1], 2]
                p2_struct = uid2secstr.loc[row[3], 2]
            except (KeyError, AttributeError) as _:
                continue
        else:
            p1_struct = row[5]
            p2_struct = row[6]

        #if len(row[2]) > len(p1_struct) or len(row[4]) > len(p2_struct):
            #raise SequenceStructureLenghtMismatchError

        if len(row[2]) > len(p1_struct):
            #print(row[2], p1_struct)
            p1_struct += "_" * (len(row[2])-len(p1_struct))
        if len(row[4]) > len(p2_struct):
            p2_struct += "_" * (len(row[4])-len(p2_struct))

        for pair in interacting_residues:
            count1, count2 = assert_interaction_quality(pair, res1, res2, p1_struct, p2_struct)
            w1 = extract_window(pair[0], row[2], window_length)
            w2 = extract_window(pair[1], row[4], window_length)
            ws1 = extract_window(pair[0], p1_struct, window_length)
            ws2 = extract_window(pair[1], p2_struct, window_length)
            try:
                ocur.execute('''INSERT INTO `pp_samples` VALUES
                                (?,?,?,?,?,?,?,?)''', (interaction_id,
                             pair[0], pair[1], w1, w2, ws1, ws2, min(count1,count2)))
                min_offset = (window_length-MIN_WINDOW_LENGTH)/2
                if min_offset > 0:
                    positive_fragments.add((w1[min_offset:-min_offset], w2[min_offset:-min_offset]))
                else:
                    positive_fragments.add((w1, w2))
            except sqlite3.IntegrityError:
                continue
    oconn.commit()

    return positive_fragments


def generate_negative_samples(icur, ocur, oconn, window_length,
                              positive_fragments, uid2secstr=None):
    print("Generating negatives")
    for row in icur.execute('''SELECT pdb_id, p1_uni_id, p1_seq,
                p2_uni_id, p2_seq, p1_struct, p2_struct, interacting_residues
                FROM `protein-protein` WHERE p1_uni_id != p2_uni_id'''):
        interaction_id = get_interaction_id(ocur, row[0], row[1], row[3])
        interacting_residues = json.loads(row[7])
        interacting_residues_p1 = sorted(set(t[0] for t in interacting_residues))
        interacting_residues_p2 = sorted(set(t[1] for t in interacting_residues))

        # Negative samples (only as many as positive ones)
        gaps1 = find_gaps(interacting_residues_p1, len(row[2]))
        gaps2 = find_gaps(interacting_residues_p2, len(row[4]))

        # Results need to be uniform across all window sizes.
        random.seed(42)

        if uid2secstr is not None:
            try:
                p1_struct = uid2secstr.loc[row[1], 2]
                p2_struct = uid2secstr.loc[row[3], 2]
            except (KeyError, AttributeError) as _:
                continue
        else:
            p1_struct = row[5]
            p2_struct = row[6]

#        if len(row[2]) > len(p1_struct) or len(row[4]) > len(p2_struct):
#            raise SequenceStructureLenghtMismatchError

        if len(row[2]) > len(p1_struct):
            p1_struct += "_" * (len(row[2])-len(p1_struct))
        if len(row[4]) > len(p2_struct):
            p2_struct += "_" * (len(row[4])-len(p2_struct))

        # Half-negatives (active side paired with a gap) are required.
        for src1, src2 in [(gaps1, gaps2),
                           (gaps1, interacting_residues_p2),
                           (interacting_residues_p1, gaps2)]:
            if len(gaps1) == 0 or len(gaps2) == 0:
                continue
            for i in xrange(NEGATIVE_SAMPLES_LIMIT):
                g1 = random.choice(src1)
                g2 = random.choice(src2)

                # Assert that windows are inside sequences' ranges.
                if (g1 + MIN_WINDOW_LENGTH/2) < 0 or (g1 - MIN_WINDOW_LENGTH/2) > len(row[2]):
                    return False
                if (g2 + MIN_WINDOW_LENGTH/2) < 0 or (g2 - MIN_WINDOW_LENGTH/2) > len(row[4]):
                    return False

                w1 = extract_window(g1, row[2], window_length)
                w2 = extract_window(g2, row[4], window_length)
                ws1 = extract_window(g1, p1_struct, window_length)
                ws2 = extract_window(g2, p2_struct, window_length)

                min_offset = (window_length-MIN_WINDOW_LENGTH)/2
                wpair = (w1[min_offset:-min_offset], w2[min_offset:-min_offset]) if min_offset > 0 else (w1, w2)
                if wpair in positive_fragments:
                    continue

                try:
                    ocur.execute('''INSERT INTO `pp_samples` VALUES
                                    (?,?,?,?,?,?,?,?)''', (interaction_id,
                                 g1, g2, w1, w2, ws1, ws2, 0))
                except sqlite3.IntegrityError:
                    continue
    oconn.commit()

def create_pp_database(dbfile, samples_dbfile, window_length,
                       uid2secstr_file=None):
    iconn = sqlite3.connect(dbfile)
    icur = iconn.cursor()

    oconn = sqlite3.connect(samples_dbfile)
    ocur = oconn.cursor()

    ocur.execute('''SELECT name FROM sqlite_master WHERE type="table" AND
                    name="pp_samples"''')
    if not ocur.fetchone():
        create_pp_samples_table(ocur)
    ocur.execute('''SELECT name FROM sqlite_master WHERE type="table" AND
                    name="interactions"''')
    if not ocur.fetchone():
        create_interactions_table(ocur)

    uid2secstr = None
    if uid2secstr_file:
        uid2secstr = pd.read_csv(uid2secstr_file, header=None, index_col=0)

    positives = generate_positive_samples(icur, ocur, oconn, window_length,
                                          uid2secstr)
    generate_negative_samples(icur, ocur, oconn, window_length, positives,
                              uid2secstr)

    iconn.close()
    oconn.close()

if __name__ == '__main__':

    if len(sys.argv) != 5:
        print("Usage: {0} dbname samples_dbname window_length".format(sys.argv[0]))
        exit(1)

    dbfile = sys.argv[1]
    samples_dbfile = sys.argv[2]
    window_length = int(sys.argv[3])

    create_pp_database(dbfile, samples_dbfile, window_length)
