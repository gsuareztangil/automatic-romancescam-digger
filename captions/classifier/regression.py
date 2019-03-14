#!/usr/bin/env python

# Author: Guillermo Suarez-Tangil

import os, sys
from optparse import OptionParser
import settings
import numpy as np
from timeit import default_timer
import traceback

from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, train_test_split, cross_val_score, cross_val_predict, PredefinedSplit

from classifier import prepare_data_crossvalidation_given_split, prepare_data_crossvalidation, feature_selection

from samples_ensambled_probs import fetch_samples, get_index_relevant_class

option_1 = { 'name' : ('-i', '--input'), 'help' : 'path : path to folder containing input files', 'nargs' : 1 }
option_2 = { 'name' : ('-d', '--debug'), 'help' : 'debug', 'action': 'store_true', 'default': False }
option_3 = { 'name' : ('-s', '--given_split'), 'help' : 'Uses holdout validation with a given split', 'action': 'store_true', 'default': False }
options = [option_1, option_2, option_3]

def spawn_estimator(estimator_name=settings.estimator_name, processors=settings.processors):
    print 'Baseline estimator:', estimator_name
    if estimator_name == 'LSVR':
        global LinearSVR
        from sklearn.svm import LinearSVR
        estimator = LinearSVR(max_iter=350, tol=0.002)
    elif estimator_name == 'SVR':
        global SVR
        from sklearn.svm import SVR
        estimator = SVR(max_iter=350, tol=0.002)
    elif estimator_name == 'LR':
        global LogisticRegression
        from sklearn.linear_model import LogisticRegression
        estimator = LogisticRegression()
    elif estimator_name == 'IR':
        global IsotonicRegression
        from sklearn.isotonic import IsotonicRegression
        estimator = IsotonicRegression()
    else:
        raise Exception('Baseline classifer "{}" not supported. Supported classifiers are: {}'.format(classifier_name, str(settings.classifiers.keys())))
    return estimator

def main_regression(options, arguments):

    # ------ Read input options and overwrite settings when redundant 
    path_samples = options.input
    if not path_samples:
        raise AttributeError('Invalid --input (-i)')

    # ------ init
    X, Y, class_names, fvector_labels, samples = prepare_data_crossvalidation(path_samples)

    pipe = []

    scaler = preprocessing.RobustScaler()
    pipe.append(('scaler', scaler))

    fselection_model = feature_selection(settings.feature_selection, settings.processors)
    if fselection_model: 
        pipe.append(('feature_selection', fselection_model))

    # ------ Build estimator
    estimator = spawn_estimator()
    pipe.append(('regression', estimator))

    estimator = Pipeline(pipe)

    # ------ Fit
    estimator.fit(X, Y)
    print estimator.named_steps['regression'].coef_
    print estimator.named_steps['regression'].intercept_



def main_isotonic_regression(options, arguments):

    # ------ Read input options and overwrite settings when redundant 
    path_samples = options.input
    if not path_samples:
        raise AttributeError('Invalid --input (-i)')

    # ------ init
    X, Y, class_names, fvector_labels, samples = prepare_data_crossvalidation(path_samples)

    for i in range(X.shape[1]):

        X_train = X[:,i]
        X_train = X_train.reshape(-1, 1)

        pipe = []

        scaler = preprocessing.RobustScaler()
        pipe.append(('scaler', scaler))

        fselection_model = feature_selection(settings.feature_selection, settings.processors)
        if fselection_model: 
            pipe.append(('feature_selection', fselection_model))

        # ------ Build estimator
        estimator = spawn_estimator('IR')
        pipe.append(('regression', estimator))

        estimator = Pipeline(pipe)

        # ------ Fit
        estimator.fit(X_train, Y)
        print estimator.named_steps['regression'].f_


def debug(options, arguments):
    
    print '--- DEBUG ----'

if __name__ == "__main__" :
    parser = OptionParser()
    for option in options:
        param = option['name']
        del option['name']
        parser.add_option(*param, **option)

    options, arguments = parser.parse_args()
    sys.argv[:] = arguments
    if options.debug:
        debug(options, arguments)
    else:
        main_regression(options, arguments)
        #main_isotonic_regression(options, arguments)

