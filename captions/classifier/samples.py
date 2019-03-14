#!/usr/bin/env python
import os, sys
import re
import nltk
import json
import numpy as np
import itertools
import pprint
import shutil

import csv
import pandas

def get_index_relevant_class():
    return 1

def process_file(cfile):
    #print 'Processing', cfile
    fileid = None
    captions = []
    probabilities = []
    with open(cfile) as f: 
        for line in f.readlines():
            if not line or len(line.strip()) == 0: 
                continue
            if line.startswith('Captions for image '):
                fileid = line[len('Captions for image '):-2]
            else: 
                caption_expr = re.split("\(|\)", line)
                caption = caption_expr[1].replace('.', '').strip()
                probability = float(caption_expr[2][2:].strip())
                captions.append(caption)
                probabilities.append(probability)
    return fileid, captions, probabilities

def get_groundtruth(sample_id):
    class_label = None
    path_to_dir = sample_id
    if 'scam' in path_to_dir and 'real' in path_to_dir:
        raise Exception('Invalid name for directory ' + path_to_dir)
    elif 'real' in path_to_dir:
        class_label = 'real'
    elif 'scam' in path_to_dir:
        class_label = 'scam'
    return class_label   

def get_features_tokens(raw_sample, stemming=False):
    features = {}

    if stemming:
        stemmer = nltk.stem.snowball.SnowballStemmer("english")

    captions, probabilities = raw_sample
    if len(captions) > 1:
        captions = [captions[0]]
    for caption in captions:
        tokens = nltk.word_tokenize(caption)
        tagged_tokens = nltk.pos_tag(tokens)
        for token, tag in tagged_tokens:
            # We only retain Nouns, Verbs, Adverbs and Adjectives ATM
            if 'NN' in tag or 'VB' in tag or 'JJ' in tag or 'RB' in tag:
                if stemming:
                    token = stemmer.stem(token)
                if token not in features:
                    features[token] = 0
                features[token] = 1 #+= 1

    return features

def get_features_ngrams(raw_sample):
    features = {}

    captions, probabilities = raw_sample

    if len(captions) > 1:
        captions = [captions[0]]

    for caption in captions:
        ngrams = nltk.ngrams(caption.split(), 2)
        for grams in ngrams:
            token = '-'.join(grams)
            if token not in features:
                features[token] = 0 
            features[token] += 1          

    return features

def get_features_selected_ngrams(raw_sample):
    features = {}

    captions, probabilities = raw_sample

    if len(captions) > 1:
        captions = [captions[0]]
        
    for caption in captions:
        tokens = nltk.word_tokenize(caption)
        tagged_tokens = nltk.pos_tag(tokens)
        selected_tokens = []
        for token, tag in tagged_tokens:
            # We only retain Nouns, Verbs, Adverbs and Adjectives ATM
            if 'NN' in tag or 'VB' in tag or 'JJ' in tag or 'RB' in tag:
                selected_tokens.append(token)

        ngrams = nltk.ngrams(selected_tokens, 2)
        for grams in ngrams:
            #print grams
            token = '-'.join(grams)
            if token not in features:
                features[token] = 0 
            features[token] += 1          

    return features

def get_features(raw_sample):
    return get_features_tokens(raw_sample)
    #return get_features_ngrams(raw_sample)
    #return get_features_selected_ngrams(raw_sample)



def fetch_samples_as_images(path_samples):
    samples = []
    print 'Fetching samples at', path_samples
    for root, dirs, files in os.walk(path_samples):
        for cfile in files:
            if cfile.endswith(".txt"):
                fileid, captions, probabilities = process_file(os.path.join(root, cfile))
                if fileid and probabilities and captions:
                    raw_sample = (captions, probabilities)

                    sample = {}
                    sample['features'] = get_features(raw_sample)
                    sample['class'] = get_groundtruth(root)
                    sample['id'] = fileid
                    samples.append(sample)
    print '#samples =', len(samples)
    return samples


def fetch_samples_as_profiles(profiles_path):

    #profiles_path = '../profiles-split/train'
    images_path = '../im2txt-all'

    samples = []

    profiles_filenames = [f for f in os.listdir(profiles_path) if (f.lower().endswith(".json"))]
    for profile in profiles_filenames:
        #print 'Processing', profile
        with open(os.path.join(profiles_path, profile)) as data_file:    
            
            data = json.load(data_file)

            sample = {}
            sample['features'] = {}

            #all_features = []
            valid_image = False
            for image in data['images']:

                if not image: 
                    continue

                # This image is the default avatar listed when the user doesn't have an picture 
                if 'fd6d0914b482e6c4aef6aa42df5eaf62' in image:
                    continue   

                valid_image = True

                image_path = os.path.join(images_path, os.path.basename(image) + '.txt')
                fileid, captions, probabilities = process_file(image_path)
                raw_sample = (captions, probabilities) 
                features_image = get_features(raw_sample)
                #all_features.append(features_image.keys())
                for f in features_image:
                    if f in sample['features']:
                        sample['features'][f] += features_image[f]
                    else:
                        sample['features'][f] = features_image[f]

            if data['scam'] == 1: 
                sample_class = 'scam'
            elif data['scam'] == 0: 
                sample_class = 'real'
            sample['class'] = sample_class
            sample['id'] = profile #data['username']
            if 'fold' in data:
                sample['fold'] = data['fold']
            samples.append(sample)

    return samples

def fetch_samples(path_samples):
    #return fetch_samples_as_images(path_samples)
    return fetch_samples_as_profiles(path_samples)


def __validation_per_profile(profiles_path, classifier, fvector_labels, std_scale):
    

    #profiles_path = '../profiles-split/test'
    images_path = '../im2txt-all'

    empty_profile_count = 0
    profile_count = 0
    profile_true_positives = 0
    profile_true_negatives = 0
    profile_false_positives = 0
    profile_false_negatives = 0

    profiles_filenames = [f for f in os.listdir(profiles_path) if (f.lower().endswith(".json"))]
    for profile in profiles_filenames:
        #print 'Processing', profile
        with open(os.path.join(profiles_path, profile)) as data_file:    
            
            data = json.load(data_file)

            scam = False # If the profile is predicted as SCAM
            active_profile = False # If there is an image in the profile
            
            D = []

            for image in data['images']:

                if not image: 
                    continue

                # This image is the default avatar listed when the user doesn't have an picture 
                if 'fd6d0914b482e6c4aef6aa42df5eaf62' in image:
                    continue   

                image_path = os.path.join(images_path, os.path.basename(image) + '.txt')
                fileid, captions, probabilities = process_file(image_path)
                raw_sample = (captions, probabilities) 
                features = get_features(raw_sample)

                for f in fvector_labels: 
                    if f in features:
                        D.append(features[f])
                    else:
                        D.append(0)

            if len(D) == 0:
                D = [0] * len(fvector_labels)
            X_test = [D] # Only one vector
            X_test = np.asarray(X_test)
            X_test = std_scale.transform(X_test)

            predicted_class = classifier.predict(X_test)[0]
            #print 'truth =', data['scam'], '\t', 'prediction =', predicted_class

            profile_count += 1

            '''
            # PREDICTION:   scam
            # TRUTH:        data['scam']
            '''

            if scam and data['scam'] == 1:
                # True Positive, predicted as scam and it is an actual scam
                profile_true_positives += 1
            elif not scam and data['scam'] == 0:
                # True Negative
                profile_true_negatives += 1
            elif scam and data['scam'] == 0:
                # False Positive, predicted as scam but it is not a scam 
                profile_false_positives += 1
            elif not scam and data['scam'] == 1:
                # False Negative
                profile_false_negatives += 1

    print 'Stats'
    print 'profile_count',              profile_count
    print 'profile_true_positives',     profile_true_positives
    print 'profile_true_negatives',     profile_true_negatives
    print 'profile_false_positives',    profile_false_positives
    print 'profile_false_negatives',    profile_false_negatives
    print 'TPR (recall)', float(profile_true_positives)/float(profile_true_positives + profile_false_negatives) # sum(TP)/sum(condition positive)
    print 'FPR (1-spec)', float(profile_false_positives)/float(profile_false_positives + profile_true_negatives) # sum(FP)/sum(condition negative)
    print 'ACCURACY\t', float(profile_true_positives + profile_true_negatives)/float(profile_count) #sum(TP) + sum(TN) / sum(total population)
    print
    print 'Confusion Matrix:'
    print '\t\t predicted condition'
    print '\t\t SCAM, REAL', '\t(cond. positive/negative)'
    print 'true condition'
    print 'SCAM\t\t ', profile_true_positives, profile_false_negatives, '\t({})'.format(profile_true_positives + profile_false_negatives)
    print 'REAL\t\t ', profile_false_positives, profile_true_negatives, '\t({})'.format(profile_false_positives + profile_true_negatives)


def __validation_per_profile_per_image(profiles_path, classifier, fvector_labels, std_scale):
    

    #profiles_path = '../profiles-split/test'
    images_path = '../im2txt-all'

    empty_profile_count = 0
    profile_count = 0
    profile_true_positives = 0
    profile_true_negatives = 0
    profile_false_positives = 0
    profile_false_negatives = 0

    profiles_filenames = [f for f in os.listdir(profiles_path) if (f.lower().endswith(".json"))]
    for profile in profiles_filenames:
        #print 'Processing', profile
        with open(os.path.join(profiles_path, profile)) as data_file:    
            
            data = json.load(data_file)

            scam = False # If the profile is predicted as SCAM
            active_profile = False # If there is an image in the profile

            for image in data['images']:

                if not image: 
                    continue

                # This image is the default avatar listed when the user doesn't have an picture 
                if 'fd6d0914b482e6c4aef6aa42df5eaf62' in image:
                    continue   

                active_profile = True # If there is an image

                image_path = os.path.join(images_path, os.path.basename(image) + '.txt')
                fileid, captions, probabilities = process_file(image_path)
                raw_sample = (captions, probabilities) 
                features = get_features(raw_sample)

                D = []
                for f in fvector_labels: 
                    if f in features:
                        D.append(features[f])
                    else:
                        D.append(0)

                X_test = [D] # Only one vector
                X_test = np.asarray(X_test)
                X_test = std_scale.transform(X_test)

                predicted_class = classifier.predict(X_test)[0]
                #print 'truth =', data['scam'], '\t', 'prediction =', predicted_class

                '''
                # SCAM: class == 0 
                # REAL: class == 1
                '''
                if predicted_class == 'scam':
                    scam = True
                    break # If one image is flagged as SCAM, the entire profile is considered malicious

            if active_profile: 

                profile_count += 1

                '''
                # PREDICTION:   scam
                # TRUTH:        data['scam']
                '''

                if scam and data['scam'] == 1:
                    # True Positive, predicted as scam and it is an actual scam
                    profile_true_positives += 1
                elif not scam and data['scam'] == 0:
                    # True Negative
                    profile_true_negatives += 1
                elif scam and data['scam'] == 0:
                    # False Positive, predicted as scam but it is not a scam 
                    profile_false_positives += 1
                elif not scam and data['scam'] == 1:
                    # False Negative
                    profile_false_negatives += 1

            else: 
                #print 'IGNORING EMPTY PROFILE', profile
                empty_profile_count += 1

    print 'Stats'
    print 'empty_profile_count',        empty_profile_count, '(no images or only site-avatar)'
    print 'profile_count',              profile_count
    print 'profile_true_positives',     profile_true_positives
    print 'profile_true_negatives',     profile_true_negatives
    print 'profile_false_positives',    profile_false_positives
    print 'profile_false_negatives',    profile_false_negatives
    print 'TPR (recall)', float(profile_true_positives)/float(profile_true_positives + profile_false_negatives) # sum(TP)/sum(condition positive)
    print 'FPR (1-prec)', float(profile_false_positives)/float(profile_false_positives + profile_true_negatives) # sum(FP)/sum(condition negative)
    print 'ACCURACY\t', float(profile_true_positives + profile_true_negatives)/float(profile_count) #sum(TP) + sum(TN) / sum(total population)
    print
    print 'Confusion Matrix:'
    print '\t\t predicted condition'
    print '\t\t SCAM, REAL', '\t(cond. positive/negative)'
    print 'true condition'
    print 'SCAM\t\t ', profile_true_positives, profile_false_negatives, '\t({})'.format(profile_true_positives + profile_false_negatives)
    print 'REAL\t\t ', profile_false_positives, profile_true_negatives, '\t({})'.format(profile_false_positives + profile_true_negatives)


def ___store_individual_predictions_append(samples, predicted, probabilities=[], probability_pos=-1, decision_function=[]):
    assert len(samples) == len(predicted), 'Parameters must have the same length.'

    #csvread = open('../results/labels.csv', 'rb')
    csvread = open('../results/testprobs.csv', 'rb')

    predicted_sample = {}
    probabilities_sample = {}
    decision_samples = {}
    for i in range(len(samples)):
        sampleID = samples[i]['id'][:-len('.json')]
        if predicted[i] == "real":
            predicted_sample[sampleID] = "0"
        elif predicted[i] == "scam":
            predicted_sample[sampleID] = "1"
        else: raise Exception("Unknown class " + predicted[i])

        if len(probabilities) > 0: 
            probabilities_sample[sampleID] = probabilities[i]

        if len(decision_function) > 0: 
            decision_samples[sampleID] = decision_function[i]

    with open('../results/test_out.csv', 'w') as output:

        freader = csv.reader(csvread, delimiter=',', quotechar='|')
        
        header = True
        for row in freader:

            if header:
                row.append('"captions_label"')
                if len(probabilities) > 0:
                    row.append('"captions_prob"')
                if len(decision_function) > 0:
                    row.append('"captions_score"')
                header = False
            else:
                row = ___normalize(row)
                profileID = row[0]
                #row.append('"' + str(predicted_sample[profileID]) + '"')
                row.append(str(predicted_sample[profileID]))
                if len(probabilities) > 0:
                    row.append(str(probabilities_sample[profileID][probability_pos][0]))
                if len(decision_function) > 0:
                    row.append(str(decision_samples[profileID]))

            output.write(','.join(row) + '\n')
            #print ', '.join(row) 


def ___store_individual_predictions(samples, predicted, probabilities=[], probability_pos=-1, decision_function=[]):
    assert len(samples) == len(predicted), 'Parameters must have the same length.'

    truth = {}
    predicted_sample = {}
    probabilities_sample = {}
    decision_samples = {}
    keys = []
    for i in range(len(samples)):
        sampleID = int(samples[i]['id'][:-len('.json')])
        keys.append(sampleID)

        if samples[i]['class'] == "real":
            truth[sampleID] = "0"
        elif samples[i]['class'] == "scam":
            truth[sampleID] = "1"
        else: raise Exception("Unknown class " + samples[i]['class'])
        
        if predicted[i] == "real":
            predicted_sample[sampleID] = "0"
        elif predicted[i] == "scam":
            predicted_sample[sampleID] = "1"
        else: raise Exception("Unknown class " + predicted[i])

        if len(probabilities) > 0: 
            probabilities_sample[sampleID] = probabilities[i]

        if len(decision_function) > 0: 
            decision_samples[sampleID] = decision_function[i]

    keys = sorted(keys)

    with open('../results/output.csv', 'w') as output:


        row = []
        row.append('file')
        row.append('truth')
        row.append('captions_label')
        if len(probabilities) > 0:
            row.append('captions_prob')
        if len(decision_function) > 0:
            row.append('captions_score')
        output.write(','.join(row) + '\n')

        for key in keys:
            row = []
            profileID = key
            row.append(str(profileID))
            row.append(str(truth[profileID]))
            row.append(str(predicted_sample[profileID]))
            if len(probabilities) > 0:
                row.append(str(probabilities_sample[profileID][probability_pos][0]))
            if len(decision_function) > 0:
                row.append(str(decision_samples[profileID]))

            output.write(','.join(row) + '\n')


def ___normalize(row):
    row_normalized = []
    for item in row:
        #if 'NA' in item:
        #    item = '0'
        row_normalized.append(item.replace('"', ''))
    return row_normalized

def addFeaturesToFullCsvFile():

    path_data_all = '../../repo/data/full.csv'
    #path_data_all = '../../repo/data/full_b.csv'
    frame_data_all = pandas.read_csv(path_data_all)


    splits = ['train', 'test', 'validation']

    for split in splits:

        #samples = fetch_samples('../profiles-split/' + split)
        samples = fetch_samples('../profiles-crossvalidation/' + split)

        for s in samples:
            id = s['id']
            if '.json' in id:
                id = id.replace('.json', '')
            indexes = frame_data_all.loc[frame_data_all['number'] == int(id)].index

            if len(indexes) == 0:
                print("WARNING: ignoring unknow index " + id)

            for index in indexes:
                frame_data_all.loc[index, 'captions'] = ','.join(s['features'].keys())

    frame_data_all.to_csv(path_data_all.replace('.csv', '+captions.csv'), index = False)

    print 'Done, check', path_data_all.replace('.csv', '+captions.csv')

def excludeVariantsFromFullCsvFile():
    
    path_data_all = '../../repo/data/full.csv'
    frame_data_all = pandas.read_csv(path_data_all)

    samples_folder_src = '../profiles-crossvalidation'
    samples_folder_exclude = '../profiles-crossvalidation' + '_excludeVariants'

    if not os.path.exists(samples_folder_exclude):
        os.mkdir(samples_folder_exclude)
    else: print samples_folder_exclude, 'already exists'


    splits = ['train', 'test', 'validation']

    for split in splits:

        src_folder = os.path.join(samples_folder_src, split)
        dst_folder = os.path.join(samples_folder_exclude, split)

        print 'Excluding files from', src_folder

        if not os.path.exists(dst_folder):
            os.mkdir(dst_folder)

        samples = fetch_samples(src_folder)

        for s in samples:
            id = s['id']
            if '.json' in id:
                id = id.replace('.json', '')

            indexes = frame_data_all.loc[frame_data_all['number'] == int(id)].index
            
            if len(indexes) == 0:
                print("WARNING: ignoring unknow index " + id)

            exclude = True
            for index in indexes:
                if not frame_data_all.loc[index, 'exclude']:
                    shutil.copyfile(os.path.join(src_folder, id + '.json'), os.path.join(dst_folder, id + '.json'))
                    exclude = False

            if exclude: print 'Excluding', id + '.json'

    print 'Done, results are in', samples_folder_exclude




if __name__ == "__main__" :

    
    #addFeaturesToFullCsvFile()
    excludeVariantsFromFullCsvFile()
