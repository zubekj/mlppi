# This script contains code to run full PPI predictor on new data using
# trained classifiers.

import argparse
import pandas as pd
import numpy as np
from sklearn.externals import joblib

from segment_database import extract_window
from segment_datasets import Struct3LEncoder

from ppi_learning import MatrixFeatureExtractor


def predict_contact_matrices(clfI, data):
    window_len = len(clfI.feature_importances_)/2
    encoder = Struct3LEncoder(window_len)

    matrices = []

    for index, row in data.iterrows():
        n = len(row["seqA"])
        m = len(row["seqB"])

        to_predict = []
        for i in xrange(n):
            s1 = extract_window(i+1, row["seqA"], window_len)
            ss1 = extract_window(i+1, row["structA"], window_len)
            for j in xrange(m):
                s2 = extract_window(j+1, row["seqB"], window_len)
                ss2 = extract_window(j+1, row["structB"], window_len)
                to_predict.append(encoder.encode(s1, s2, ss1, ss2))

        matrices.append(np.reshape(clfI.predict_proba(to_predict)[:,1], (n, m)))

    return matrices


def predict_ppi(clfII, matrices):
    extractor = MatrixFeatureExtractor()
    processed_matrices = [extractor.aggregated_features(m) for m in matrices]
    return clfII.predict_proba(processed_matrices)[:,1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Full PPI prediction for new proteins.')
    parser.add_argument('levelI_predictor', help='Trained level I predictor to use.')
    parser.add_argument('levelII_predictor', help='Trained level II predictor to use.')
    parser.add_argument('data_file', help='CSV file with protein pairs for prediction.')
    args = parser.parse_args()

    clfI = joblib.load(args.levelI_predictor)
    clfII = joblib.load(args.levelII_predictor)
    data = pd.read_csv(args.data_file, header=None)

    data.columns = ["idA", "idB", "seqA", "seqB", "structA", "structB"]

    matrices = predict_contact_matrices(clfI, data)
    results = predict_ppi(clfII, matrices)

    print(results)
