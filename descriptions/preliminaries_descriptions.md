# Description Classifier

This page covers the current performance of the descriptions classifier. 

# Instance Creation
The first code block creates document instances for LibShortText for k-fold cross validation in Python.

```python
from random import shuffle
import json, os, unicodedata, itertools, operator, sys
from nltk.tokenize import TweetTokenizer

#Use NLTK's TweetTokenizer
tw = TweetTokenizer()

#Seed for random.shuffle, so you always get the same randomized dataset.
def seed():
	return 0.398283018721

#Return K number of training and test partitions
def Kfolds(texts, K=10):
	shuffle(texts, seed)
	for k in xrange(K):
		train = [x for i, x in enumerate(texts) if i % K != k] 
		test = [x for i, x in enumerate(texts) if i % K == k] 
		yield k, train, test

#Extract description from original json file, tokenize it and return LibShortText instance.
def extract_text(doc):	
	jsondoc = open(doc, 'r').read()
	config = json.loads(jsondoc)
	tag = config[u"scam"]
	description = config[u"description"]
	if description != None:
		words = tw.tokenize(description)
		words = [unicodedata.normalize('NFKD', w).encode('ascii', 'ignore') for w in words]
		if len(words) > 1:
			text = ' '.join(words)
			if tag == 1:
				out = 'scam' + '\t' + text
			else:
				out = 'real' + '\t' + text
			return (out, doc)

#Create instances from jsonfiles in a k-fold cross validation set-up.
if __name__ == "__main__":
	if sys.argv[1:]:
		dir = sys.argv[1]
		dir2 = sys.argv[2]
		if os.path.isfile(os.path.join(dir,".DS_Store")): os.remove(os.path.join(dir,".DS_Store"))
		instanceDir = os.path.join(dir2,"libshorttext_instances")
		if os.path.exists(instanceDir) == False: os.mkdir(instanceDir)
		files = os.listdir(dir)
		for k, train, test in Kfolds(files, K=10):
			train_instances = []
			test_instances = []
			for item in train:
				train_instances.append(extract_text(dir + item))
			for item in test:
				test_instances.append(extract_text(dir + item))
			train_instances = [it for it in train_instances if it != None]
			test_instances = [it for it in test_instances if it != None]
			train_file = open(instanceDir + '/train_0' + str(k) + '_BOWfeatures.txt', 'w')
			test_file = open(instanceDir + '/test_0' + str(k) + '_BOWfeatures.txt', 'w')

			for (out,doc) in train_instances:
				train_file.write(out + '\n')
			for (out, doc) in test_instances:
				test_file.write(out + '\n')
		print "LibShortText instances are saved in %s" %instanceDir
```

The same approach was used to extract character and word n-grams, using the following code.

```python
from nltk.util import ngrams

#Extract character or word n-grams with '[b]' representing the left pad symbol 
#(at the beginning of the string) and '[e]' representing the right pad symbol (at the end).
def extract_ngrams(input, n):
	ngram_feats = list(ngrams(input, n, True, True, '[b]', '[e]'))
	return ngram_feats
```

# Machine Learning
This code block trains a LibShortText model on each training partition of the data. Next, the model is evaluated on each test partition and the predictions are saved in a separate folder. 

```python
import numpy, sys
from subprocess import call

#Locate directory with LibShortText instances and initialize the experiments
def initialize(dir, parameters):
	outFiles=[]
	instanceDir = os.path.join(dir,"instances")
	trainFiles=sorted([os.path.join(instanceDir,file) for file in os.listdir(instanceDir) if 'train' in file])
	testFiles=sorted([os.path.join(instanceDir,file) for file in os.listdir(instanceDir) if ‘test' in file])
	if trainFiles == [] or testFiles == []:
		print “Train or test files are missing.”
	for trainFile, testFile in zip(trainFiles, testFiles):
		modelFile = Libshort_train(featDir, trainFile, parameters)
		outDir = Libshort_predict(featDir, testFile, parameters, modelFile)
	print “LibShortText predictions are saved in %s" %outDir	

#Train a LibShortText model
def Libshort_train(featDir, trainFile, parameters):
	#Add path
	libshort_learn = ‘~/libshorttext-1.1/text-train.py'
	modelDir = os.path.join(featDir, 'models')
	if os.path.exists(modelDir) == False: os.mkdir(modelDir)
	delim = ''
	modelFile = modelDir + '/' + trainFile.split('/')[-1] + '.' + delim.join(parameters) + '.model'
	call(['python', libshort_learn, '-P', parameters[0][-1], '-F', parameters[1][-1], '-L', parameters[2][-1], '-N', parameters[3][-1],'-G', parameters[4][-1], '-f', trainFile, modelFile], 0, None, None)
	return modelFile

#Apply LibShortText model on test data
def Libshort_predict(featDir, testFile, parameters, modelFile):
	#Add path
	libshort_predict = ‘~/libshorttext-1.1/text-predict.py'
	outDir = os.path.join(featDir, 'outfiles')
	if os.path.exists(outDir) == False: os.mkdir(outDir)
	delim = ''
	modelDir = modelFile.split('/')[:-1]
	modelDir = '/'.join(modelDir)
	outFile = outDir + '/' + modelFile.split('/')[-1] + '.' + delim.join(parameters) + '.out'
	call(['python', libshort_predict, '-f', testFile, modelFile, outFile], 0, None, None)
	return outDir

if __name__ == "__main__":
	if sys.argv[1:]:
		dir=sys.argv[1]
		#Adjust parameter combinations. See ~/libshorttext-1.1/README.txt for LibShortText parameter options.
		parameters = [['P0', 'F0', 'L0', 'N0', 'G1'], ['P0', 'F1', 'L0', 'N0', 'G1’]]
		for parameter in parameters:
			print parameter
			initialize(dir, parameter)
```

This code block trains and tests a Scikit-learn Multinomial Naive Bayes classification model on each partition using the LibShortText instance format. Other Scikit-learn classifiers were trained with the appropriate changes.

```python

import sys,os
from collections import Counter
from nltk.classify import SklearnClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

#Locate LibShortText instances and Initialize the experiments.
def initialize(dir):
	instanceDir = os.path.join(dir,"instances")
	trainFiles=sorted([os.path.join(instanceDir,file) for file in os.listdir(instanceDir) if 'train' in file])
	testFiles=sorted([os.path.join(instanceDir,file) for file in os.listdir(instanceDir) if 'test' in file])
	outDir = os.path.join(instanceDir, 'outfiles')
	if os.path.exists(outDir) == False: os.mkdir(outDir)
	for trainFile, testFile in zip(trainFiles, testFiles):
		out = testFile.split('/')[-1] + '.sklearn_NB.out'
		fout = os.path.join(outDir, out)
		outFile = open(fout, 'w')
		sklearn(trainFile, testFile, outFile)
	print "Scikit-learn classifier predictions are saved in %s" %outDir

def read_libshort_instances(file):
	o = open(file, 'r').readlines()
	lines = [i.strip() for i in o]
	instances = []
	for i, line in enumerate(lines):
		all = line.split('\t')
		label = all[0]
		if len (all) > 1:
			featureDict = Counter(all[1].split(' '))
			instances.append((featureDict, label))
	return instances


#Train and test a scikit-learn classification model.
def sklearn(trainFile, testFile, outFile):
	pipeline = Pipeline([('tfidf', TfidfTransformer()), ('Mnb', MultinomialNB())])
	classifier = SklearnClassifier(pipeline)
	train_instances = read_libshort_instances(trainFile)
	all_test = read_libshort_instances(testFile)
	goldLabels = []
	test_instances = []
	for (featDict, goldLabel) in all_test:
		test_instances.append(featDict)
		goldLabels.append(goldLabel)
	model = classifier.train(train_instances)
	predictedLabels = classifier.classify_many(test_instances)
	for goldLabel, predictedLabel in zip(goldLabels, predictedLabels):
		outFile.write(predictedLabel + '\t' + goldLabel + '\n')

if __name__ == "__main__":
	if sys.argv[1:]:
		dir=sys.argv[1]
		initialize(dir)
```
