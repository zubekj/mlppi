from itertools import chain
import numpy as np
import pandas as pd
from operator import itemgetter


def random_cv_split(pairs, n_folds=2):
    rand_pairs = list(pairs.values)
    np.random.shuffle(rand_pairs)

    t = int(len(pairs) / n_folds)

    for i in range(n_folds):
        test = rand_pairs[(i*t):((i+1)*t)]
        train = rand_pairs[:(i*t)] + rand_pairs[((i+1)*t):]

        yield (pd.DataFrame(train, columns=pairs.columns),
               pd.DataFrame(test, columns=pairs.columns))


def random_cv_split_filtered(pairs, n_folds=2, mask=True):
    rand_pairs = list(pairs.values)
    np.random.shuffle(rand_pairs)

    t = int(len(pairs) / n_folds)

    for i in range(n_folds):
        test = rand_pairs[(i*t):((i+1)*t)]
        train = rand_pairs[:(i*t)] + rand_pairs[((i+1)*t):]

        train_proteins = set(chain(*((p[0], p[1]) for p in train)))
        test_proteins = set(chain(*((p[0], p[1]) for p in test)))

        filtered_proteins = train_proteins - test_proteins

        train_mask = (pairs.iloc[:, 0].isin(filtered_proteins) &
                      pairs.iloc[:, 1].isin(filtered_proteins))
        test_mask = (pairs.iloc[:, 0].isin(test_proteins) &
                     pairs.iloc[:, 1].isin(test_proteins))

        if mask:
            yield (train_mask, test_mask)
        else:
            yield (pd.DataFrame(pairs[train_mask], columns=pairs.columns),
                   pd.DataFrame(pairs[test_mask], columns=pairs.columns))


def random_cv_split_object_level(pairs, n_folds=2):
    objects = list(set(chain(*pairs.values[:, :2])))
    np.random.shuffle(objects)

    t = int(len(objects) / n_folds)

    for i in range(n_folds):
        if i == n_folds-1:
            test_s = objects[(i*t):]
            train_s = objects[:(i*t)]
        else:
            test_s = objects[(i*t):((i+1)*t)]
            train_s = objects[:(i*t)] + objects[((i+1)*t):]

        test = pairs[pairs.iloc[:, 0].isin(test_s) &
                     pairs.iloc[:, 1].isin(test_s)]
        train = pairs[pairs.iloc[:, 0].isin(train_s) &
                      pairs.iloc[:, 1].isin(train_s)]

        yield (pd.DataFrame(train, columns=pairs.columns),
               pd.DataFrame(test, columns=pairs.columns))


def cv_split_balanced(pairs, n_folds=2):
    node_degrees = pd.Series(list(chain(*pairs.values[:,:2]))).value_counts().\
        sort_values().index.tolist()

    i = 0
    s = [[] for _ in range(n_folds)]
    while len(node_degrees) > 0:
        s[i].append(node_degrees.pop())
        i = (i + 1) % n_folds

    for i in range(n_folds):
        test_s = s[i]
        train_s = list(chain(*(s[j] for j in range(n_folds) if j != i)))

        test = pairs[pairs.iloc[:, 0].isin(test_s) &
                     pairs.iloc[:, 1].isin(test_s)]
        train = pairs[pairs.iloc[:, 0].isin(train_s) &
                      pairs.iloc[:, 1].isin(train_s)]

        yield (pd.DataFrame(train, columns=pairs.columns),
               pd.DataFrame(test, columns=pairs.columns))


def cv_split_tit_for_tat(pairs, n_folds=2, mask=True):

    node_neighbors = {}
    for p in pairs.values:
        a, b = p[0], p[1]
        if a not in node_neighbors:
            node_neighbors[a] = set()
        if b not in node_neighbors:
            node_neighbors[b] = set()
        node_neighbors[a].add(b)
        node_neighbors[b].add(a)

    node_degrees = pd.Series(list(chain(*pairs.values[:,:2]))).value_counts().\
        sort_values().index.tolist()

    if n_folds > len(node_degrees):
        n_folds = len(node_degrees)

    i = 0
    s = [set() for _ in range(n_folds)]
    while len(node_degrees) > 0:
        items = (len(s[i] & node_neighbors[n]) for n in node_degrees)
        index, element = max(enumerate(items), key=itemgetter(1))
        s[i].add(node_degrees[index])
        del node_degrees[index]
        i = (i + 1) % n_folds

    for i in range(n_folds):
        test_s = s[i]
        train_s = list(chain(*(s[j] for j in range(n_folds) if j != i)))

        test_mask = (pairs.iloc[:, 0].isin(test_s) &
                     pairs.iloc[:, 1].isin(test_s))
        train_mask = (pairs.iloc[:, 0].isin(train_s) &
                      pairs.iloc[:, 1].isin(train_s))

        if mask:
            yield (train_mask, test_mask)
        else:
            yield (pd.DataFrame(pairs[train_mask], columns=pairs.columns),
                   pd.DataFrame(pairs[test_mask], columns=pairs.columns))
