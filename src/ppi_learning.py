import cPickle
from itertools import chain
import argparse
import numpy as np
import pandas as pd
from sklearn import cross_validation, svm, metrics, ensemble
from sklearn.externals import joblib

from helpers import feature_extraction
from helpers.matrix_processing import *
from helpers.train_test_split import cv_split_tit_for_tat, random_cv_split_filtered

#from scipy.misc import imresize

# Class for sequence-based features extraction.
class SequenceFeatureExtractor(object):

    def __init__(self):
        self.sf = feature_extraction.SequenceFeatures()

    def process_dataset(self, data):

        #return np.array([self.sf.extract_2grams(d[4],d[5]) for d in data])
        #return np.array([self.sf.extract_aac(d[4],d[5]) for d in data])

        data1 = []
        for d in data:
            seq1 = d[2]
            #if len(seq1) > len(d[4]):
            #    seq1 = seq1[:len(d[4])]
            seq2 = d[3]
            #if len(seq2) > len(d[5]):
            #    seq2 = seq2[:len(d[5])]
            data1.append((seq1,seq2))

        return np.array([
            #self.sf.extract_aac(seq1,seq2)
            #self.sf.extract_pseudo_aac(seq1,seq2, 9)
            #self.sf.extract_2grams(seq1,seq2)
            #self.sf.extract_quasi_residue_couples(seq1,seq2, 9)
            self.sf.extract_flat_features(seq1,seq2,9)
            #self.sf.extract_aac_hqi8(seq1,seq2)
            for seq1,seq2 in data1])

class MatrixFeatureExtractor(object):

    def __init__(self):
        self.fs = feature_extraction.FlatFeatures()
        self.imgf = feature_extraction.ImageFeatures()

    def process_datasets(self, X_datas):
        Xavg = X_datas[0]
        for X in X_datas[1:]:
            for i in xrange(len(X_datas[0])):
                Xavg[i] += X[i]
        Xavg = [x/len(X_datas) for x in Xavg]

        return np.array([self.aggregated_features(x) for x in Xavg])

    def aggregated_features(self, mat):
        # Optimal set for yeast:

        def counts(mat):
            c1 = count_along_axis(mat)
            c2 = count_along_axis(mat, axis=1)
            return c1+c2 if sum(c1) > sum(c2) else c2+c1

        def counts_diag(mat):
            c1 = count_diagonals(mat)
            c2 = count_diagonals(mat.T)
            return c1+c2 if sum(c1) > sum(c2) else c2+c1

        return to_vector(mat,
                counts,
                lambda mat: largest_connected(mat, num=3, num_connected=3),
                np.mean,
                np.var,
                counts_diag,
                count_intersections,
                score_distribution,
                #mass_center,
                #lambda mat: list(rotate_mass(imresize(mat, (20, 20), mode='F')).ravel())
                )

        # Optimal set for bacteria (TODO)
        #return to_vector(mat,
        #        np.sum,
        #        np.var,
        #        count_along_axis,
        #        lambda mat: count_along_axis(mat, axis=1),
        #        count_diagonals,
        #        lambda mat: count_diagonals(mat[::-1]),
        #        count_intersections,
        #        lambda mat: count_along_axis(mat > 0.4),
        #        lambda mat: count_along_axis(mat > 0.4, axis=1),
        #        #lambda mat: count_squares(mat),
        #        #lambda mat: count_larger(mat, 0.42)/float(mat.size),
        #        score_distribution,
        #        best_interactions,
        #        lambda mat: count_larger(mat, 0.4)/float(mat.size),
        #        lambda mat: count_larger(mat, 0.35)/float(mat.size),
        #        lambda mat: count_larger(mat, 0.3)/float(mat.size),
        #        lambda mat: largest_connected(mat),
        #        lambda mat: largest_connected(mat, min_strength = 0.4),
        #        #lambda mat: largest_connected(mat > 0.42),
        #        #lambda mat: largest_connected(mat > 0.37),
        #        ##lambda mat: largest_connected(mat > 0.35),
        #        ##lambda mat: rotate_mass(imresize(mat, (20, 20), mode='F')).ravel(),
        #        lambda mat: mat.size,
        #        )

# Class for extracting features using computer vision methods.
class ImageFeatureExtractor(object):

    def __init__(self):
        self.imgf = feature_extraction.ImageFeatures()
        self.matf = MatrixFeatureExtractor()

    def process_datasets(self, X_datas):
        Xavg = X_datas[0]
        for X in X_datas[1:]:
            for i in xrange(len(X_datas[0])):
                Xavg[i] += X[i]
        Xavg = [x/len(X_datas) for x in Xavg]

        #Xs = X_datas[0]
        #for X in X_datas[1:-1]:
        #    for i in xrange(len(X_datas[0])):
        #        Xs[i] += (Xavg[i] - X[i])**2
        #Xs = [np.sqrt(x) for x in Xs]

        #Xcd = [Xavg[i]/Xs[i] for i in xrange(len(X_datas[0]))]

        res = []
        for x in Xavg:
            #hlines, vlines, m = self.imgf.find_lines_plus(x)
            #hlines, vlines, m = self.imgf.find_lines_border(x)
            #hlines, vlines, m = self.imgf.find_lines(x, t=0.01)
            #res.append(self.imgf.line_statistics(x, hlines, vlines, m) + list(self.matf.aggregated_features(x)))
            res.append(self.matf.aggregated_features(x))
            #res.append(self.imgf.line_statistics(x, hlines, vlines, m))
            #res.append(self.imgf.partition(x))
            #res.append(m[hlines,vlines].flatten())
            #res.append(self.imgf.pap_metrics(m))
            #res.append(self.imgf.pap_metrics(x))
            #res.append(np.hstack((self.imgf.pap_metrics(x),
            #    self.imgf.line_statistics(x, hlines, vlines, m))))
            #res.append(self.imgf.resize_matrix(x, 32, 32).flatten())
            #res.append(np.hstack((self.imgf.resize_matrix(x, 10, 10).flatten(),
            #    self.imgf.line_statistics(x, hlines, vlines, m))))
            #res.append(np.hstack((self.imgf.resize_matrix(x, 10, 10).flatten(),
            #    self.imgf.partition(m))))
            #res.append(np.hstack((self.imgf.partition(m),
            #    self.imgf.line_statistics(x, hlines, vlines, m))))

        return np.array(res)

# Class for extracting feature vector from prediction matrix.
class FlatFeatureExtractor(object):

    def __init__(self):
        self.fs = feature_extraction.FlatFeatures()

    def process_dataset(self, data, max_d=6):
        return np.array([
                self.fs.extract_flat_pairs(x, max_d) +
                self.fs.extract_flat_pairs_diag(x, max_d) +
                self.fs.extract_shapes(x)
                for x in data])

#DATA_FILE = "ppi_data/ppi_xy_ws3_w29.bin"
#DATA_FILE1 = "ppi_data/ppi_xy_ws3.bin"
#POS_NEG_DATA_FILE = "ppi_data/pos_neg.pkl"

def load_dataset(filename):
    arrays = []
    with open(filename, 'rb') as npfile:
        try:
            while True:
                arrays.append(np.load(npfile))
        except IOError:
            pass
    Y = arrays[-1]
    X = []
    for arr in arrays[:-1]:
        X += list(arr)
    return (X, Y)

def sequence_data(pos_neg_file):
    with open(pos_neg_file, 'rb') as cfile:
        positives, negatives = cPickle.load(cfile)

    Y = np.r_[np.ones(len(positives)), np.zeros(len(negatives))]
    X_data = chain(positives, negatives)

    #X_data = [(d[0],d[1],d[2],d[3],d[4],d[5]) if d[0]>d[1] else (d[1],d[0],d[3],d[2],d[5],d[4]) for d in X_data]

    feature_extractor = SequenceFeatureExtractor()
    X = feature_extractor.process_dataset(X_data)

    return X,Y

def predictorI_data(data_file):
    Xb, Y = load_dataset(data_file)

    feature_extractor = ImageFeatureExtractor()
    X = feature_extractor.process_dataset(Xb, 8)
    return (X,Y)

def predictorI_aggregated_data(data_files, labels):
    Y = None
    X_datas = []
    for d in data_files:
        X, Y = load_dataset(d)
        X_datas.append(X)

    feature_extractor = ImageFeatureExtractor()
    X = feature_extractor.process_datasets(X_datas)

    #for i in xrange(X.shape[0]):
    #    if X[i,0] <= X[i,1]:
    #        X[i,0], X[i,1] = X[i,1], X[i,0]

    #mask = labels[:,0] <= labels[:,1]
    #X[mask] = np.hstack((np.vstack((X[mask,1], X[mask,0])).T, X[mask,2:]))

    #X = np.vstack((X, np.hstack((np.vstack((X[:,1], X[:,0])).T, X[:,2:]))))
    #Y = np.hstack((Y,Y))

    return (X,Y)

def merge_data(vectors):
    X = np.hstack((v[0] for v in vectors))
    Y = vectors[0][1]
    return (X,Y)

def visualise_matrix(matrix):
    c = P.pcolor(matrix,cmap=P.cm.binary)
    P.colorbar(c,orientation='horizontal')
    P.show()

def cv_experiment_two_halves(X, Y, labels):
    #clf = ensemble.RandomForestClassifier(n_estimators=300, max_depth=10,
    #        min_samples_split=10)
    clf = ensemble.RandomForestClassifier(n_estimators=300, max_depth=7, random_state=42)

    Y_score = []
    Y_true = []

    #kf = cross_validation.KFold(len(labels), n_folds=30, random_state=42)
    #kf = cross_validation.KFold(len(labels), n_folds=30)
    kf = cross_validation.LeaveOneOut(len(labels))
    for train, test in kf:
        test_mask = (np.in1d(labels[:,0], labels[test].flatten()) & np.in1d(labels[:,1], labels[test].flatten()))
        train_mask = (np.invert(np.in1d(labels[:,0], labels[test].flatten())) & np.invert(np.in1d(labels[:,1], labels[test].flatten())))

        print(sum(train_mask), sum(test_mask))

        # This monster below is class stratification
        #n_pos, n_neg = sum(Y[train_mask] == 1), sum(Y[train_mask] == 0)
        #min_class = min(n_pos, n_neg)
        #train_mask[np.random.choice(np.where(train_mask & (Y == 1))[0], n_pos-min_class, replace=False)] = False
        #train_mask[np.random.choice(np.where(train_mask & (Y == 0))[0], n_neg-min_class, replace=False)] = False
        #
        #n_pos, n_neg = sum(Y[test_mask] == 1), sum(Y[test_mask] == 0)
        #min_class = min(n_pos, n_neg)
        #test_mask[np.random.choice(np.where(test_mask & (Y == 1))[0], n_pos-min_class, replace=False)] = False
        #test_mask[np.random.choice(np.where(test_mask & (Y == 0))[0], n_neg-min_class, replace=False)] = False

        train_matrix = np.vstack((X[train_mask][:,:X.shape[1]/2], X[train_mask][:,X.shape[1]/2:]))
        train_labels = np.hstack((Y[train_mask], Y[train_mask]))

        clf.fit(train_matrix, train_labels)

        Y_score1 = clf.predict_proba(X[test_mask][:,:X.shape[1]/2])[:,1]
        #Y_score1 = clf.predict_proba(X[test][:,:X.shape[1]/2])[:,0]
        Y_score2 = clf.predict_proba(X[test_mask][:,X.shape[1]/2:])[:,1]
        #Y_score2 = clf.predict_proba(X[test][:,X.shape[1]/2:])[:,0]
        Y_score.append(Y_score1*Y_score2)
        Y_true.append(Y[test_mask])

    Y_score = np.hstack(Y_score)
    Y_true = np.hstack(Y_true)

    Y_pred = np.array(Y_score > 0.25, dtype=int)

    accuracy = [metrics.accuracy_score(Y_true, Y_pred)]
    precision = [metrics.precision_score(Y_true, Y_pred)]
    recall = [metrics.recall_score(Y_true, Y_pred)]
    auc = [metrics.roc_auc_score(Y_true, Y_score)]

    accuracy = np.array(accuracy)
    precision = np.array(precision)
    recall = np.array(recall)
    auc = np.array(auc)

    print("| Accuracy  | Precision | Recall    | AUC       |")
    print("| "+ " | ".join(["{0:.2f} {1:.2f}".format(a.mean(), a.std())
        for a in [accuracy, precision, recall, auc]]) + " |")


def random_2level_train_test_split(Y, labels):

    train = np.repeat(False, len(Y))
    test = np.repeat(True, len(Y))

    while sum(test) > sum(train):
        idx = np.random.choice(np.arange(len(Y))[~train])
        train[idx] = True
        train = (np.in1d(labels[:, 0], labels[train].flatten()) &
                 np.in1d(labels[:, 1], labels[train].flatten()))
        test = ~(np.in1d(labels[:, 0], labels[train].flatten()) |
                 np.in1d(labels[:, 1], labels[train].flatten()))

    return train, test


def dietterich_5_2cv_experiment(X, Y, labels, result_name):

    labels = pd.DataFrame(labels)

    print(X.shape, Y.shape)

    clf = ensemble.RandomForestClassifier(n_estimators=300, max_depth=7,
                                          random_state=42)

    Y_score = []
    Y_true = []
    cv_fold = []

    np.random.seed(42)
    for i in xrange(5):

        suff = "a"
        for train_mask, test_mask in random_cv_split_filtered(labels, n_folds=2):

            train_mask = train_mask.values
            test_mask = test_mask.values

            print(sum(train_mask), sum(Y[train_mask]),
                  sum(test_mask), sum(Y[test_mask]),
                  len(set(labels.values.flatten())) -
                  len(set(labels.values[train_mask].flatten()) |
                      set(labels.values[test_mask].flatten())))

            clf.fit(X[train_mask], Y[train_mask])
            Y_score.append(clf.predict_proba(X[test_mask])[:, 1])
            Y_true.append(Y[test_mask])

            cv_fold.append(["{0}{1}".format(i, suff)] * sum(test_mask))
            suff = "b"

    Y_score = np.hstack(Y_score)
    Y_true = np.hstack(Y_true)
    cv_fold = np.hstack(cv_fold)

    pd.DataFrame({"Y_true": Y_true, "Y_score": Y_score,
                  "cv_fold": cv_fold}).to_csv("yscores_5_2cv_{0}.csv".format(result_name),
                                              index=False)

    Y_pred = np.array(Y_score > 0.5, dtype=int)

    accuracy = [metrics.accuracy_score(Y_true, Y_pred)]
    precision = [metrics.precision_score(Y_true, Y_pred)]
    recall = [metrics.recall_score(Y_true, Y_pred)]
    auc = [metrics.roc_auc_score(Y_true, Y_score)]

    TP = sum((Y_pred == 1)[Y_true == 1])
    FP = sum((Y_pred == 1)[Y_true == 0])
    TN = sum((Y_pred == 0)[Y_true == 0])
    FN = sum((Y_pred == 0)[Y_true == 1])

    accuracy = np.array(accuracy)
    precision = np.array(precision)
    recall = np.array(recall)
    auc = np.array(auc)

    with open("res_{0}.txt".format(result_name), "w") as f:
        f.write("& " + " & ".join(["{0:.2f}".format(a.mean(), a.std())
                for a in [accuracy, precision, recall, auc]]) + "\\\\\n")
        f.write("TP={0} FP={1} TN={2} FN={3}".format(TP, FP, TN, FN))


def cv_experiment(X, Y, labels, result_name):

    labels = pd.DataFrame(labels)

    print(X.shape, Y.shape)

    clf = ensemble.RandomForestClassifier(n_estimators=300, max_depth=7,
                                          random_state=42)

    Y_score = []
    Y_true = []
    cv_fold = []

    np.random.seed(42)
    for i, masks in enumerate(cv_split_tit_for_tat(labels, n_folds=10)):

        train_mask = masks[0].values
        test_mask = masks[1].values

        print(sum(train_mask), sum(Y[train_mask]),
                sum(test_mask), sum(Y[test_mask]),
                len(set(labels.values.flatten())) -
                len(set(labels.values[train_mask].flatten()) |
                    set(labels.values[test_mask].flatten())))

        clf.fit(X[train_mask], Y[train_mask])
        Y_score.append(clf.predict_proba(X[test_mask])[:, 1])
        Y_true.append(Y[test_mask])

        cv_fold.append(["{0}".format(i)] * sum(test_mask))

    Y_score = np.hstack(Y_score)
    Y_true = np.hstack(Y_true)
    cv_fold = np.hstack(cv_fold)

    pd.DataFrame({"Y_true": Y_true, "Y_score": Y_score,
                  "cv_fold": cv_fold}).to_csv("yscores_cv_{0}.csv".format(result_name),
                                              index=False)

    Y_pred = np.array(Y_score > 0.5, dtype=int)

    accuracy = [metrics.accuracy_score(Y_true, Y_pred)]
    precision = [metrics.precision_score(Y_true, Y_pred)]
    recall = [metrics.recall_score(Y_true, Y_pred)]
    auc = [metrics.roc_auc_score(Y_true, Y_score)]

    TP = sum((Y_pred == 1)[Y_true == 1])
    FP = sum((Y_pred == 1)[Y_true == 0])
    TN = sum((Y_pred == 0)[Y_true == 0])
    FN = sum((Y_pred == 0)[Y_true == 1])

    accuracy = np.array(accuracy)
    precision = np.array(precision)
    recall = np.array(recall)
    auc = np.array(auc)

    with open("res_{0}.txt".format(result_name), "w") as f:
        f.write("& " + " & ".join(["{0:.2f}".format(a.mean(), a.std())
                for a in [accuracy, precision, recall, auc]]) + "\\\\\n")
        f.write("TP={0} FP={1} TN={2} FN={3}".format(TP, FP, TN, FN))


def cv_experiment_svm(X, Y, labels, result_name):

    labels = pd.DataFrame(labels)

    print(X.shape, Y.shape)

    clf = svm.SVC()

    Y_score = []
    Y_true = []
    cv_fold = []

    np.random.seed(42)
    for i, masks in enumerate(cv_split_tit_for_tat(labels, n_folds=10)):

        train_mask = masks[0].values
        test_mask = masks[1].values

        print(sum(train_mask), sum(Y[train_mask]),
                sum(test_mask), sum(Y[test_mask]),
                len(set(labels.values.flatten())) -
                len(set(labels.values[train_mask].flatten()) |
                    set(labels.values[test_mask].flatten())))

        clf.fit(X[train_mask], Y[train_mask])
        Y_score.append(clf.decision_function(X[test_mask]))
        Y_true.append(Y[test_mask])

        cv_fold.append(["{0}".format(i)] * sum(test_mask))

    Y_score = np.hstack(Y_score)
    Y_true = np.hstack(Y_true)
    cv_fold = np.hstack(cv_fold)

    pd.DataFrame({"Y_true": Y_true, "Y_score": Y_score,
                  "cv_fold": cv_fold}).to_csv("yscores_cv_{0}_svm.csv".format(result_name),
                                              index=False)

    Y_pred = np.array(Y_score > 0.5, dtype=int)

    accuracy = [metrics.accuracy_score(Y_true, Y_pred)]
    precision = [metrics.precision_score(Y_true, Y_pred)]
    recall = [metrics.recall_score(Y_true, Y_pred)]
    auc = [metrics.roc_auc_score(Y_true, Y_score)]

    TP = sum((Y_pred == 1)[Y_true == 1])
    FP = sum((Y_pred == 1)[Y_true == 0])
    TN = sum((Y_pred == 0)[Y_true == 0])
    FN = sum((Y_pred == 0)[Y_true == 1])

    accuracy = np.array(accuracy)
    precision = np.array(precision)
    recall = np.array(recall)
    auc = np.array(auc)

    with open("res_svm_{0}.txt".format(result_name), "w") as f:
        f.write("& " + " & ".join(["{0:.2f}".format(a.mean(), a.std())
                for a in [accuracy, precision, recall, auc]]) + "\\\\\n")
        f.write("TP={0} FP={1} TN={2} FN={3}".format(TP, FP, TN, FN))


def grid_search_cv_experiment(X, Y, labels):

    #for C in xrange(10,100,10):
    for C in [0.1, 1, 10]:
        for gamma in [0.1*i for i in xrange(1,10)]:
        #for gamma in [5.0]:
            clf = svm.SVC(C=C, gamma=gamma, kernel='rbf')

            Y_score = []
            Y_true = []

            kf = cross_validation.StratifiedKFold(Y, n_folds=30, random_state=42)
            for train, test in kf:
                test_mask = (np.in1d(labels[:,0], labels[test].flatten()) & np.in1d(labels[:,1], labels[test].flatten()))
                train_mask = (np.invert(np.in1d(labels[:,0], labels[test].flatten())) & np.invert(np.in1d(labels[:,1], labels[test].flatten())))

                clf.fit(X[train_mask],Y[train_mask])
                Y_score.append(clf.decision_function(X[test_mask])[:,0])
                Y_true.append(Y[test_mask])

            Y_score = np.hstack(Y_score)
            Y_true = np.hstack(Y_true)

            Y_pred = np.array(Y_score > 0.0, dtype=int)

            auc = metrics.roc_auc_score(Y_true, Y_score)

            print("C={0} gamma={1} AUC: {2}".format(C, gamma, auc))

def train_test_experiment_two_halves(X_train, Y_train, X_test, Y_test):

    clf = ensemble.RandomForestClassifier(n_estimators=300, max_depth=10,
            min_samples_split=10, random_state=42)

    train_matrix = np.vstack((X_train[:,:X_train.shape[1]/2], X_train[:,X_train.shape[1]/2:]))
    train_labels = np.hstack((Y_train, Y_train))

    clf.fit(train_matrix, train_labels)

    Y_score1 = clf.predict_proba(X_test[:,:X_test.shape[1]/2])[:,1]
    Y_score2 = clf.predict_proba(X_test[:,X_test.shape[1]/2:])[:,1]

    Y_score = Y_score1*Y_score2
    Y_pred = np.array(Y_score > 0.25, dtype=int)

    print("| Accuracy  | Precision | Recall    | AUC     |")
    print("| "+ "      | ".join(["{0:.2f}".format(a.mean())
        for a in [metrics.accuracy_score(Y_test, Y_pred),
                  metrics.precision_score(Y_test, Y_pred),
                  metrics.recall_score(Y_test, Y_pred),
                  metrics.roc_auc_score(Y_test, Y_pred)]]) + "    |")


def train_test_experiment(X_train, Y_train, X_test, Y_test):

    clf = ensemble.RandomForestClassifier(n_estimators=300, max_depth=10,
            min_samples_split=10, random_state=42)

    clf.fit(X_train, Y_train)
    Y_score = clf.predict(X_test)

    Y_pred = np.array(Y_score > 0.5, dtype=int)

    print("| Accuracy  | Precision | Recall    | AUC     |")
    print("| "+ "      | ".join(["{0:.2f}".format(a.mean())
        for a in [metrics.accuracy_score(Y_test, Y_pred),
                  metrics.precision_score(Y_test, Y_pred),
                  metrics.recall_score(Y_test, Y_pred),
                  metrics.roc_auc_score(Y_test, Y_pred)]]) + "    |")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training and testing classifiers on ppi_data. All data sources are concatenated into one input vector.')
    parser.add_argument('labels_file', help='file with protein pairs UIDs')
    parser.add_argument('data_files', nargs='+', help='files with predictor I outputs or pos_neg data')
    parser.add_argument('-f', '--flat_features', action='store_true', help='use flat features')
    parser.add_argument('-t', '--train_data', nargs='+', help='use separate training data')

    args = parser.parse_args()

    with open(args.labels_file, "rb") as f:
        labels = cPickle.load(f)

    data_vectors = []
    predictorI_files = []
    for f in args.data_files:
        if f.split('.')[1] == 'pkl':
            data_vectors.append(sequence_data(f))
        else:
            predictorI_files.append(f)

    if len(predictorI_files) > 0:
        if args.flat_features:
            for f in predictorI_files:
                data_vectors.append(predictorI_data(f))
        else:
            data_vectors.append(predictorI_aggregated_data(predictorI_files,
                                                           labels))

    if len(data_vectors) == 0:
        print("You must specify at least one data source.")
        exit(0)

    print("Data sources: " + " ".join(args.data_files))

    X, Y = merge_data(data_vectors)
    labels = np.array(labels)

    if args.train_data:
        data_vectors = []
        predictorI_files = []
        for f in args.train_data:
            if f.split('.')[1] == 'pkl':
                data_vectors.append(sequence_data(f))
            else:
                predictorI_files.append(f)

        if len(predictorI_files) > 0:
            if args.flat_features:
                for f in predictorI_files:
                    data_vectors.append(predictorI_data(f))
            else:
                data_vectors.append(predictorI_aggregated_data(predictorI_files, labels))

        X_train, Y_train = merge_data(data_vectors)
        train_test_experiment(X_train, Y_train, X, Y)
        #train_test_experiment_two_halves(X_train, Y_train, X, Y)
    else:
        #cv_experiment(X, Y, labels)
        cv_experiment_svm(X, Y, labels)
        #grid_search_cv_experiment(X, Y, labels)
        #cv_experiment_two_halves(X, Y, labels)
