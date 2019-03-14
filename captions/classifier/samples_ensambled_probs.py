#!/usr/bin/env python

import os, sys
import re
import json
import numpy as np
import itertools
import pprint
import operator

def get_index_relevant_class():
    return 0

def scale(OldValue):
    if OldValue == 'NA':
        NewValue = 0.0
    else:
        OldValue = float(OldValue)
        OldMin = 0.0
        OldMax = 1.0
        NewMin = -1.0
        NewMax = 1.0
        NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
    return NewValue

def fetch_samples_probability_esamble(probabilities_path):

    import csv

    samples = []
    print 'Fetching samples at', probabilities_path

    with open(probabilities_path, 'rb') as csvread:

        freader = csv.DictReader(csvread, delimiter=',', quotechar='|')
        predictors = ['description_prob', 'demographics_prob', 'captions_prob']

        for row in freader:
            sample = {}

            profileID = row['file']
            truth = row['truth']
            #description_label = row['description_label']
            description_prob = row['descriptions_prob']
            #demographics_label = row['demographics_label']
            demographics_prob = row['demographics_prob']
            #captions_label = row['captions_label']
            captions_prob = row['captions_prob']

            sample['features'] = {}
            sample['features']['description_prob'] = scale(description_prob)
            sample['features']['demographics_prob'] = scale(demographics_prob)
            sample['features']['captions_prob'] = scale(captions_prob)
            #sample['features']['description_label'] = scale(description_label)
            #sample['features']['demographics_label'] = scale(demographics_label)
            #sample['features']['captions_label'] = scale(captions_label)
            #sample['features'] = []
            #sample['features'].append(scale(description_prob))
            #sample['features'].append(scale(demographics_prob))
            #sample['features'].append(scale(captions_prob))

            sample['class'] = scale(truth)
            sample['id'] = profileID
            samples.append(sample)

    print '#samples =', len(samples)
    return samples

def fetch_samples(path_samples):
    return fetch_samples_probability_esamble(path_samples)



def ___normalize(row):
    row_normalized = []
    for item in row:
        #if 'NA' in item:
        #    item = '0'
        row_normalized.append(item.replace('"', ''))
    return row_normalized


def ___get_esambled_prediction(): 

    import csv, collections

    predicted = []
    Y = []

    ties = 0

    #with open('../results/testlabels.csv', 'rb') as csvread:
    with open('../results/testlabels-all.csv', 'rb') as csvread:

        freader = csv.reader(csvread, delimiter=',', quotechar='|')

        header = True
        predictors = []
        num_predictors = 0
        for row in freader:

            #del row[2] # remove 'demographics'
            #del row[3] # remove 'description'
            #del row[4] # remove 'captions'

            row = ___normalize(row)
            if header:
                header = False
                predictors = row[2:]
                num_predictors = len(predictors)
                print 'Predictors:', predictors
            else:
                profileID = row[0]
                truth = row[1]
                
                counter = collections.Counter(row[2:])
                del counter['NA']

                tie = False
                if len(counter) > 1:
                    cmm = counter.most_common(2)
                    if cmm[0][1] == cmm[1][1]:
                        ties += 1
                        tie = True
                    #else: print cmm
                Y.append(truth)
                if tie:
                    #predicted.append('0')
                    predicted.append(row[4])
                else:
                    predicted.append(counter.most_common(1)[0][0])

            
    print '# Ties {}'.format(ties)

    import classifier
    classifier.print_metrics(Y, predicted, ['0', '1'], ['1'])


def ___get_esambled_prediction_weighted_voting(): 

    import csv, collections

    predicted = []
    Y = []

    predictions = []
    ties = 0

    profiles = []
    probabilities = []

    #with open('../results/testlabels.csv', 'rb') as csvread:
    with open('../results/testlabels-all-final.csv', 'rb') as csvread:

        freader = csv.reader(csvread, delimiter=',', quotechar='|')

        header = True
        predictors = []
        num_predictors = 0
        for row in freader:

            votes = []

            row = ___normalize(row)
            if header:
                header = False
                predictors = row[2:]
                num_predictors = len(predictors)
                print 'Predictors:', predictors
            else:
                profileID = row[0]
                truth = row[1]
                Y.append(truth)

                prediction = {}
                i = 2
                for predictor in predictors:
                    prediction[predictor] = row[i]
                    i += 1
                predictions.append(prediction)  

                profiles.append(profileID)
              
    
    for prediction in predictions:

        all_weighted_votes = []

        if prediction['description'] != 'NA':
            all_weighted_votes.extend([prediction['description']]*4)
        all_weighted_votes.extend([prediction['demographics']]*2)
        all_weighted_votes.extend([prediction['captions']]*3)


        counter = collections.Counter(all_weighted_votes)
        del counter['NA']
        
        tie = False
        if len(counter) > 1:
            cmm = counter.most_common(2)
            if cmm[0][1] == cmm[1][1]:
                ties += 1
                tie = True

        if tie:
            predicted.append(prediction['description'])
            #predicted.append(row[4])
        else:
            predicted.append(counter.most_common(1)[0][0])

        probabilities.append(counter.most_common(1)[0][1]/9)  

        #print all_weighted_votes, predicted[-1]

    if len(Y) != len(predicted):    
        raise Exception('WTF?')

    print '# Ties {}'.format(ties)

    import classifier
    classifier.print_metrics(Y, predicted, ['0', '1'], ['1'])

    import classifier
    classifier.print_metrics(Y, predicted, ['0', '1'], ['1'])

    missclassified = {}
    for i in range(len(predicted)):
        print Y[i], predicted[i]
        if (Y[i] == '0' and predicted[i] != '0') \
        or (Y[i] == '1' and predicted[i] != '1'):
            missclassified[profiles[i]] = probabilities[i]

    print sorted(missclassified.items(), key=operator.itemgetter(1), reverse=True)[0:10]




def ___get_esambled_prediction_weighted_voting_with_threshold(): 

    import csv, collections

    predicted = []
    Y = []

    predictions = []
    ties = 0

    #with open('../results/testlabels.csv', 'rb') as csvread:
    with open('../results/testlabels-all-final.csv', 'rb') as csvread:

        freader = csv.reader(csvread, delimiter=',', quotechar='|')

        header = True
        predictors = []
        num_predictors = 0
        for row in freader:

            votes = []

            row = ___normalize(row)
            if header:
                header = False
                predictors = row[2:]
                num_predictors = len(predictors)
                print 'Predictors:', predictors
            else:
                profileID = row[0]
                truth = row[1]
                Y.append(truth)

                prediction = {}
                i = 2
                for predictor in predictors:
                    prediction[predictor] = row[i]
                    i += 1

                predictions.append(prediction)   


    for prediction in predictions:

        all_weighted_votes = []

        if prediction['description'] == 'NA':
            #all_weighted_votes.extend([int(prediction['description'])]*3)
            all_weighted_votes.extend([int(prediction['captions'])]*4)
        else:
            all_weighted_votes.extend([int(prediction['description'])]*4)

        all_weighted_votes.extend([int(prediction['demographics'])]*2)
        all_weighted_votes.extend([int(prediction['captions'])]*3)

        score = sum(all_weighted_votes)

        if score >= int(len(all_weighted_votes)/2) + 1:
            predicted.append('1')
        else:
            predicted.append('0')

    if len(Y) != len(predicted):    
        raise Exception('WTF?')

    print '# Ties {}'.format(ties)

    import classifier
    classifier.print_metrics(Y, predicted, ['0', '1'], ['1'])



def ___get_esambled_prediction_partially_boost_captions(): 

    import csv, collections

    print 'Partially boost negative predictions from captions'

    Y = []
    predicted = []

    predictions = []

    #with open('../results/testlabels.csv', 'rb') as csvread:
    with open('../results/testlabels-all.csv', 'rb') as csvread:

        freader = csv.reader(csvread, delimiter=',', quotechar='|')


        header = True
        predictors = []
        num_predictors = 0
        for row in freader:
            row = ___normalize(row)
            if header:
                header = False
                predictors = row[2:]
                num_predictors = len(predictors)
                print 'Predictors:', predictors
            else:
                profileID = row[0]
                truth = row[1]
                Y.append(truth)
                
                prediction = {}
                i = 2
                for predictor in predictors:
                    prediction[predictor] = row[i]
                    i += 1

                predictions.append(prediction)


    for prediction in predictions:

        if int(prediction['captions']) == 0:
            predicted.append(prediction['captions'])
        else:
            del prediction['captions']
            counter = collections.Counter(prediction.values())
            del counter['NA']
            predicted.append(counter.most_common(1)[0][0])

    if len(Y) != len(predicted):    
        raise Exception('WTF?')

    import classifier
    classifier.print_metrics(Y, predicted, ['0', '1'], ['1'])


def ___get_esambled_prediction_weighted_voting_with_probabilities(): 

    import csv, collections

    predicted = []
    Y = []

    probabilities = []
    profiles = []

    with open('../results/validationprobs-all_probsonly.csv', 'rb') as csvread:

        freader = csv.DictReader(csvread, delimiter=',', quotechar='|')

        predictors = ['description_prob', 'demographics_prob', 'captions_prob']

        for row in freader:

            profileID = row['file']
            truth = row['truth']
            #description_label = row['descriptions_label']
            description_prob = row['descriptions_prob']
            #demographics_label = row['demographics_label']
            demographics_prob = row['demographics_prob']
            #captions_label = row['captions_label']
            captions_prob = row['captions_prob']

            prediction = 1

            if description_prob == 'NA' and demographics_prob == 'NA' and captions_prob == 'NA':
                raise Exception('Not a Number!')

            # ['captions_prob', 'demographics_prob', 'description_prob']
            #weights = [1.83570469, 2.66747052, 2.81212375]
            #intercept = -4.41998518 

            # ['captions_prob', 'demographics_prob', 'description_prob']
            weights = [1.83570469, 2.66747052, 2.81212375]
            intercept = -4.41998518 

            # ['captions_prob', 'demographics_prob', 'description_prob']
            #weights = [1.10475506, 1.54044513, 1.73504838]
            #intercept = -2.65668318

            if description_prob != 'NA':
                prediction += float(description_prob) * weights[2]

            if demographics_prob != 'NA':
                prediction += float(demographics_prob) * weights[1]
            if captions_prob != 'NA':
                prediction += float(captions_prob) * weights[0] 

            prediction /= len(weights)
            prediction += intercept
            maxv = (weights[0] + weights[1] + weights[2]) + intercept
            minv = intercept
            prediction = ___scale(prediction, minv, maxv)

            if prediction >= 0.5:
                predicted.append('1')
            else:
                predicted.append('0')

            probabilities.append(prediction)
            profiles.append(profileID)

            Y.append(truth)
        
    import classifier
    classifier.print_metrics(Y, predicted, ['0', '1'], ['1'])

    missclassified = {}
    for i in range(len(predicted)):
        #print predicted[i], probabilities[i]
        if Y[i] == '0' and probabilities[i]  >= 0.5 \
        or Y[i] == '1' and probabilities[i]  <= 0.5: 
            missclassified[profileID] = probabilities[i]

    print sorted(missclassified.items(), key=operator.itemgetter(1), reverse=True)




def ___min_maxDescriptions():

    import csv
    
    csvread = open('../results/testprobs-all-old.csv', 'rb')

    freader = csv.DictReader(csvread, delimiter=',', quotechar='|')
    first = True 
    for row in freader:
        if row['"description_prob"'] == 'NA':
            continue
        if first:
            minv = float(row['"description_prob"'])
            maxv = float(row['"description_prob"'])
            first = False
        else:
            current = float(row['"description_prob"'])
            if current < minv:
                minv = current
            if current > maxv:
                maxv = current
    print minv, maxv
    return minv, maxv

def ___scale(OldValue, minv, maxv):
    if OldValue == 'NA':
        NewValue = 0.0
    else:
        OldValue = float(OldValue)
        OldMin = minv
        OldMax = maxv
        NewMin = 0.0
        NewMax = 1.0
        NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
    return NewValue


def ___scale_descriptions(OldValue):
    if OldValue == 'NA':
        NewValue = 0.0
    else:
        OldValue = float(OldValue)
        OldMin = -4.21573466793
        OldMax = 2.93917709493
        NewMin = 0
        NewMax = 1.0
        NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
    return NewValue


def ___mergeCSVs():

    import csv
    
    csvread = open('../results/testprobs-all-old.csv', 'rb')
    #csvread2 = open('../results/testprobs-descriptions.csv', 'rb')

    freader = csv.DictReader(csvread, delimiter=',', quotechar='|')
    #freader2 = csv.DictReader(csvread2, delimiter=',', quotechar='|')

    samples = {}
    for row in freader:
        samples[int(row['"file"'])] = row

    #for row in freader2:
    #    samples[int(row['file'])]['"description_prob"'] = row['prob_1']

    with open('../results/testprobs-all.csv', 'w') as output:

        header = ['file', 'truth', 'description_label', 'description_prob', 'demographics_label', 'demographics_prob', 'captions_label', 'captions_prob']

        output.write(','.join(header) + '\n')

        keys = sorted(samples.keys())
        for key in keys: 
            row = []
            sample = samples[key]
            for key in header:
                row.append(sample['"' + key + '"'])
            output.write(','.join(row) + '\n')

def ___mergeCSVs_and_scale():

    import csv

    
    csvread = open('../results/testprobs-all-old.csv', 'rb')
    #csvread2 = open('../results/testprobs-descriptions.csv', 'rb')

    freader = csv.DictReader(csvread, delimiter=',', quotechar='|')
    #freader2 = csv.DictReader(csvread2, delimiter=',', quotechar='|')

    samples = {}
    for row in freader:
        samples[int(row['"file"'])] = row

    #for row in freader2:
    #    samples[int(row['file'])]['"description_prob"'] = row['prob_1']

    with open('../results/testprobs-all.csv', 'w') as output:

        header = ['file', 'truth', 'description_label', 'description_prob', 'demographics_label', 'demographics_prob', 'captions_label', 'captions_prob']

        output.write(','.join(header) + '\n')

        keys = sorted(samples.keys())
        for key in keys: 
            row = []
            sample = samples[key]
            for key in header:
                if key == 'description_prob':
                    value = sample['"' + key + '"']
                    if value != 'NA':
                        value = str(___scale_descriptions(value))
                    row.append(value)
                else:
                    row.append(sample['"' + key + '"'])
            output.write(','.join(row) + '\n')


if __name__ == "__main__" :
    #___get_esambled_prediction_partially_boost_captions()
    #___get_esambled_prediction()
    ___get_esambled_prediction_weighted_voting()
    #___get_esambled_prediction_weighted_voting_with_threshold()
    #___min_maxDescriptions()
    #___mergeCSVs()
    #___mergeCSVs_and_scale()
    #___get_esambled_prediction_weighted_voting_with_probabilities()
