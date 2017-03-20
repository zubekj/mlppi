# Multi-level machine learning prediction of protein-protein interactions #

Julian Zubek, Marcin Tatjewski, Adam Boniecki, Maciej Mnich, Subhadip Basu,
Dariusz Plewczynski∗

* Corresponding author: d.plewczynski@cent.uw.edu.pl

Aim of this project is to predict PPI using multiscale features. Accurate
predictors could be then used for recovering protein interaction network.

The starting point for this experiment is extracting complexes of proteins from PDB.
Their sequences are sliced into segments using sliding
window technique. Pairs of interacting segments are used as a dataset for
training level-I classifier. Level-II classifier uses the predictions of
level-I classifier as an input and decides whether two proteins with known
sequences interact or not.

## Requirements ##

Experimental pipeline is implemented in Python 2.

External software:
* DSSP

Binary DSSP packages can be downloaded from *ftp://ftp.cmbi.ru.nl/pub/software/dssp/*
and placed in any directory listed in the *PATH* variable of the environment.

Python 2 packages:
* prody
* ruffus
* pandas
* numpy
* scipy
* scikit-learn
* affinity
* sqlite3
* urllib
* lxml

## Project structure ##

The *src* directory contains various scripts used in experimental pipeline.
The scripts are generally independent, working in batch mode: they feed on
input files and produce the output. Ruffus pipelining library is used to
glue all the parts together. *experiments* directory should contain
subdirectories corresponding to different experiments. Each experimental
directory contains *experiment.py* file with Ruffus pipeline configuration,
and all necessary input files, intermediate files and results files.

## Running the experiment ##

If all the dependencies are satisfied, you can run full experimental pipeline
with:

    > python experiment.py

If you are interested in repeating just one specific experimental step,
analyze the structure of *experiment.py* to learn the API.

## Supplied data ##

You can find training and testing data sets as well as trained classifiers
in the *experiments/yeast_experiment* directory:

* pdb_yeast.id_list -- list of PDB complexes used in the experiment.
* pdb_yeast.ppi.sqlite -- sqlite3 database with protein-protein interactions.
* yeast_train.csv -- training set for level-I predictor.
* yeast_train.uid -- Uniprot ids of protein pairs in the train set.
* yeast_test.csv -- testing set for level-I predictor.
* yeast_test.uid -- Uniprot ids of protein pairs in the test set.
* uid2secstr_psipred.csv -- sequences and PSIPRED-predicted secondary structures for the proteins used in the experiment.

The following files can be easily recreated by running *experiment.py* script:

* pdb_yeast.segment.sqlite -- sqlite3 database with segment interactions.
* ppi_yeast_pos_neg_psipred_train.pkl -- pickled list of positive and negative pairs for training level-II predictor.
* ppi_yeast_xy_psipred_train.pkl -- pickled list of positive and negative predicted contact maps for training level-II predictor.

## Using the trained classifiers ##

If you are just interested in running the trained classifiers to predict
interactions for supplied protein pairs, check the script *ppi_predict.py*
in *src* directory.

    > usage: ppi_predict.py [-h] levelI_predictor levelII_predictor data_file
    >
    > Full PPI prediction for new proteins.
    >
    > positional arguments:
    >  levelI_predictor   Trained level I predictor to use.
    >  levelII_predictor  Trained level II predictor to use.
    >  data_file          CSV file with protein pairs for prediction.
    >
    > optional arguments:
    >  -h, --help         show this help message and exit

*levelI_predictor* and *levelII_predictor* should be pickled classifiers
trained during full experimental pipeline. *data_file* should be a CSV
(comma separated values) file with no header containing the following
columns:

1. Protein A ID
2. Protein B ID
3. Protein A sequence
4. Protein B sequence
5. Protein A secondary structure
6. Protein B secondary structure

## Citing ##

Zubek, J., Tatjewski, M., Boniecki, A., Mnich, M., Basu, S., & Plewczynski, D. (2015). Multi-level machine learning prediction of protein–protein interactions in Saccharomyces cerevisiae. PeerJ, 3, e1041.

Zubek, J., Tatjewski, M., Basu, S., & Plewczynski, D. (2015). Evaluating Multi-level Machine Learning Prediction of Protein-protein Interactions. In Computational Methods in Data Analysis ITRIA 2015 (pp. 199–211). Institute of Computer Science, Polish Academy of Sciences.
