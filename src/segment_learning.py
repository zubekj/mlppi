import sys
import numpy as np
import pandas as pd
from sklearn import linear_model, metrics
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from matplotlib import pyplot as plt

from helpers.feature_extraction import *

sec_str_annotations = dict((e,i) for i,e in enumerate(['-', 'B', 'E', 'G', 'H', 'I', 'S', 'T']))

def load_data(filename):
    df = pd.read_csv(filename)
    X = df.iloc[:,:-1].values
    y = df.iloc[:,-1].values

    return np.array(X, dtype=float), np.array(y, dtype=float)

#def secondary_structure(X):
#    return np.array([[sec_str_annotations[e] for e in row] for row in X])

#def simple(X):
#    vals = dict((e, i) for i, e in enumerate(np.unique(X)))
#    return np.array([[vals[e] for e in row] for row in X])

def train_test_experiment(X, y, X_test, y_test, save_clf=None):
    clf = RandomForestClassifier(max_depth=15, min_samples_split=10, n_estimators=300, n_jobs=-1, random_state=1)
    #clf = KNeighborsClassifier(n_neighbors=5)
    #clf = SVC(gamma=2)
    clf.fit(X, y)
    if save_clf:
        joblib.dump(clf, save_clf, compress=3)

    y_score = clf.predict_proba(X_test)[:,1]
    print("ROC AUC: {0}".format(roc_auc_score(y_test, y_score)))

def train_test_halves_experiment(X, y, X_test, y_test):
    clf = RandomForestClassifier(max_depth=15, min_samples_split=10, n_estimators=300, n_jobs=-1, random_state=1)
    #clf = KNeighborsClassifier(n_neighbors=5)
    train_matrix = np.vstack((X[:, :X.shape[1]/2], X[:, X.shape[1]/2:]))
    train_labels = np.hstack((y, y))

    clf.fit(train_matrix, train_labels)

    y_score1 = clf.predict_proba(X_test[:, :X.shape[1]/2])[:,1]
    y_score2 = clf.predict_proba(X_test[:, X.shape[1]/2:])[:,1]
    y_score = y_score1*y_score2
    print("w{0}: {1}".format(window_length, roc_auc_score(y_test, y_score)))

def test_different_sizes(X, y, X_test, y_test):
    for i in xrange(1,window_length/2+1):
        indices = range(window_length/2-i, window_length/2+i+1) +\
                  range(window_length+window_length/2-i, window_length+window_length/2+i+1)
        print(len(indices)/2)
        train_test_experiment(X[:, indices], y, X_test[:, indices], y_test)

def train_classifier(train_file, test_file, clf_out_file):
    X, y = load_data(train_file)
    X_test, y_test = load_data(test_file)
    train_test_experiment(X, y, X_test, y_test, clf_out_file)

if __name__ == "__main__":

    train_file = sys.argv[1]
    test_file = sys.argv[2]
    clf_out_file = sys.argv[3]

    X, y = load_data(train_file)
    X_test, y_test = load_data(test_file)

    #train_set = pd.DataFrame(np.hstack((X, y[:,None])))
    #test_set = pd.DataFrame(np.hstack((X_test, y_test[:,None])))
    #train_set.to_csv("yeast_pp_w21_secstr.train.csv", header=False, index=False)
    #test_set.to_csv("yeast_pp_w21_secstr.test.csv", header=False, index=False)
    #exit(0)

    #X, y = load_data(train_file, hqi8)
    #X_test, y_test = load_data(test_file, hqi8)
    #X, y = load_data(train_file, simple)
    #X_test, y_test = load_data(test_file, simple)
    #X, y = load_data(train_file, aaindex_contact)
    #X_test, y_test = load_data(test_file, aaindex_contact)

    #X, y = load_data(train_file, secondary_structure_plus_simple)
    #X_test, y_test = load_data(test_file, secondary_structure_plus_simple)
    #X, y = load_data(train_file, secondary_structure_plus_hqi8)
    #X_test, y_test = load_data(test_file, secondary_structure_plus_hqi8)

    #train_test_experiment(X, y, X_test, y_test)
    #test_different_sizes(X, y, X_test, y_test)
    train_test_experiment(X, y, X_test, y_test, clf_out_file)
    #train_test_halves_experiment(X, y, X_test, y_test)
