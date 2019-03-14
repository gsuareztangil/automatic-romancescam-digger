#!/usr/bin/env python
# -*- coding: utf-8 -*-
from random import shuffle
import json, os, unicodedata, itertools, operator, sys, csv
from nltk.tokenize import TweetTokenizer
from subprocess import call
from collections import defaultdict

#Use NLTK's TweetTokenizer
tw = TweetTokenizer()

#set path to LibShortText package
learning_path = '~/libshorttext-1.1/text-train.py'
prediction_path = '~/libshorttext-1.1/text-predict.py'

#Generate document instances in LibShortText format from a given directory containing JSON files.
#Format: gold_label \t feature1 feature2 feature3 [...] #profile_number \n 
def create_instances(dir, instanceDir):
	no_description_profiles = []
	files = os.listdir(dir)
	files = [file for file in files if 'json' in file]
	name = dir.split('/')[-1]
	instanceDoc = instanceDir + '/' + name + '.libshort_instances.txt'
	instanceFile = open(instanceDoc, 'w')
	for file in files:
		profile = file.split('.')[0]
		jsondoc = open(dir + '/' + file, 'r').read()
		config = json.loads(jsondoc)
		tag = config[u"scam"]
		description = config[u"description"]
		if description != None:
			description = description.replace('\n', ' ')
			words = tw.tokenize(description)
			words = [unicodedata.normalize('NFKD', w).encode('ascii', 'ignore') for w in words]
			if len(words) > 1:
				features = ' '.join(words)
				if tag == 1:
					instance = 'scam' + '\t' + features + ' #' + profile
				else:
					instance = 'real' + '\t' + features + ' #' + profile
				instanceFile.write(instance + '\n')
		else:
			no_description_profiles.append(profile)
	instanceFile.close()
	return instanceDoc, no_description_profiles	

#Train a description classification model based the generated document instances.
def Libshort_train(trainFile, modelDir):
	print trainFile
	libshort_learn = learning_path
	delim = ''
	parameters = ['P1', 'F3', 'L0', 'N1', 'G1']
	modelFile = modelDir + '/' + trainFile.split('/')[-1].split('.')[0] + '.' + delim.join(parameters) + '.libshort_model'
	call(['python', libshort_learn, '-P', parameters[0][-1], '-F', parameters[1][-1], '-L', parameters[2][-1], '-N', parameters[3][-1],'-G', parameters[4][-1], '-f', trainFile, modelFile], 0, None, None)
	return modelFile

#Generate predictions for the test and validation data
def Libshort_predict(testFile, modelFile, outDir):
	libshort_predict = prediction_path
	delim = ''
	outFile = testFile.split('/')[-1].split('.')[0]
	modelinfo = modelFile.split('/')[-1].split('.')[-2]
	outFile2 = outDir + '/' + outFile + '_' + modelinfo + '.libshort_out'
	call(['python', libshort_predict, '-f', testFile, modelFile, outFile2], 0, None, None)
	return outFile2

def binarize_label(label):
	if label == 'scam':
		return 1
	else:
		return 0

#Generate a csv file containing profile number, gold_label, predicted_label, scam_probability to be used by the ensemble learner.
def write_csv(outFile, testFile, no_descr_test):
	csv_dict = defaultdict(list)
	csv_name = outFile.split('/')[-1].split('_')[0] + '.csv'
	csvfile = open(os.path.dirname(os.path.dirname(outFile)) + '/' + csv_name, 'wb')
	csv_writer = csv.writer(csvfile, delimiter=',')
	csv_writer.writerow(['profile', 'gold_label', 'descriptionClassifier_label', 'descriptionClassifier_probability']) 
	results = open(outFile, 'r').readlines()
	results = [line.strip() for line in results[6:]]
	testInstances = open(testFile, 'r').readlines()
	testInstances = [instance.strip().split('#')[-1] for instance in testInstances]
	for profile, result in zip(testInstances, results):
		predictions = result.split('\t')
		csv_dict[int(profile)] = [binarize_label(predictions[1]), binarize_label(predictions[0]), predictions[-1]]
	for profile in no_descr_test:
		csv_dict[int(profile)] = [0,0,0]
	for item in sorted(csv_dict.keys()):
		csv_writer.writerow([item] + csv_dict[item])
	csvfile.close()
	return os.path.dirname(os.path.dirname(outFile))

#Create new directories, generate instances, initiate the experiments and generate csv files containing the results.
def initialize(trainDir, testDir, validationDir):
	outFiles=[]
	experimentDir = os.path.dirname(trainDir)
	experimentDir = os.path.join(experimentDir,"description_classification")
	if os.path.exists(experimentDir) == False: os.mkdir(experimentDir)
	instanceDir = os.path.join(experimentDir,"instances")
	if os.path.exists(instanceDir) == False: os.mkdir(instanceDir)
	modelDir = os.path.join(experimentDir,"models")
	if os.path.exists(modelDir) == False: os.mkdir(modelDir)
	outDir = os.path.join(experimentDir,"outfiles")
	if os.path.exists(outDir) == False: os.mkdir(outDir)
	trainFile, no_descr_train = create_instances(trainDir, instanceDir)
	testFile, no_descr_test = create_instances(testDir, instanceDir)
	validationFile, no_descr_validation = create_instances(validationDir, instanceDir)
	modelFile = Libshort_train(trainFile, modelDir)
	test_out = Libshort_predict(testFile, modelFile, outDir)
	validation_out = Libshort_predict(validationFile, modelFile, outDir)
	csv_test = write_csv(test_out, testFile, no_descr_test)
	write_csv(validation_out, validationFile, no_descr_validation)
	print "RESULTS are saved in csv-format in %s" %csv_test

if __name__ == "__main__":
	if sys.argv[1:]:
		trainDir = sys.argv[1]
		testDir = sys.argv[2]
		validationDir = sys.argv[3]
		initialize(trainDir, testDir, validationDir)
