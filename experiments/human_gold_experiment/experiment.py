import sys
import os
import tempfile
import shutil
import cPickle
import numpy as np
from ruffus import *

sys.path.append("../../src/")

import ppi_database
import segment_database
import segment_datasets
import segment_learning
import ppi_datasets
import ppi_matrices
import ppi_learning

window_length = 21
interaction_threshold = 15
pdb_id_file = "pdb_human.id_list"
ppi_db_file = "pdb_human.ppi.sqlite"
train_uid_file = "human_train.uid"
test_uid_file = "human_test.uid"
train_csv_file = "human_train.csv"
train_ppi_uid_file = "human_gold_nooverlap.csv"
test_csv_file = "human_test.csv"
ppi_train_file = "ppi_human_pos_neg_train.pkl"
ppi_train_interactions_file = "ppi_human_interactions_train.pkl"
segment_classifier_file = "levelI_clf.zip"
ppi_matrices_train_file = "ppi_human_xy_train.pkl"
pdb_data_folder = os.path.dirname(os.path.abspath(__file__))+"/pdb_data"

segment_feature_encoder = segment_datasets.Struct3LEncoder(window_length)
psipred_secstr_file = "uid2secstr_human_gold.csv"

ppi_global_clf_result_file = "res_global.txt"
ppi_matrix_clf_result_file = "res_matrix.txt"

@transform(pdb_id_file, suffix(".id_list"), ".ppi.sqlite")
def create_ppi_database(input_file, output_file):
    ppi_database.create_protein_database(input_file, output_file)

@transform(create_ppi_database, suffix(".ppi.sqlite"), ".segment.sqlite")
def create_segment_database(input_file, output_file):
    segment_database.create_pp_database(input_file, output_file, window_length)

@split(create_segment_database, [train_uid_file, test_uid_file, train_csv_file,
    test_csv_file])
def create_segment_datasets(input_file, output_files):
    segment_datasets.create_data_table(input_file, train_uid_file, test_uid_file,
            train_csv_file, test_csv_file, segment_feature_encoder,
            interaction_threshold, True, 0.5)

@merge(create_segment_datasets, segment_classifier_file)
def train_segment_classifier(input_files, output_file):
    X, y = segment_learning.load_data(train_csv_file)
    X_test, y_test = segment_learning.load_data(test_csv_file)
    segment_learning.train_test_experiment(X, y, X_test, y_test, output_file)

@merge([create_ppi_database, create_segment_datasets], [ppi_train_file,
    ppi_train_interactions_file])
def create_ppi_datasets(input_files, output_files):
    np.random.seed(42)
    data, interactions = ppi_datasets.load_data(train_ppi_uid_file, None,
                                                psipred_secstr_file)
    ppi_datasets.save_pos_neg_data(data, interactions, ppi_train_file,
            ppi_train_interactions_file)

@merge([train_segment_classifier, create_ppi_datasets], ppi_matrices_train_file)
def create_ppi_matrices(input_files, output_file):
    ppi_matrices.save_dataset(output_file, segment_classifier_file,
            segment_feature_encoder, ppi_train_file, window_length)

@merge(create_ppi_datasets, ppi_global_clf_result_file)
def train_ppi_global_classifier(input_files, result_file):
    X, Y = ppi_learning.sequence_data(ppi_train_file)
    with open(ppi_train_interactions_file, "rb") as f:
        labels = cPickle.load(f)
    np.random.seed(42)
    ppi_learning.cv_experiment(X, Y, labels, result_file)

@merge([create_ppi_matrices, create_ppi_datasets], ppi_matrix_clf_result_file)
def train_ppi_matrix_classifier(input_files, result_file):
    with open(ppi_train_interactions_file, "rb") as f:
        labels = cPickle.load(f)
    X, Y = ppi_learning.predictorI_aggregated_data([ppi_matrices_train_file],
            labels)
    np.random.seed(42)
    ppi_learning.cv_experiment(X, Y, labels, result_file)

pipeline_run()
#pipeline_run(touch_files_only=True)
#pipeline_printout_graph ('flowchart.svg', 'svg',
#                         [train_ppi_matrix_classifier, train_ppi_global_classifier],
#                         no_key_legend = False)
