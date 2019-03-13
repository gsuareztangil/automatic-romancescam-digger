# Description Classifier

This page covers the current performance of the descriptions classifier. 

# Instance Creation
The first code block creates document instances for LibShortText for k-fold cross validation in Python.

```Bash
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

#Create instances from jsonfiles.
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

The same approach was used to extract character n-grams, using the following code.

```Bash
from nltk.util import ngrams

#Extract character or word n-grams with '[b]' representing the left pad symbol (at the beginning of the string) and '[e]' representing the right pad symbol (at the end).
def extract_ngrams(input, n):
	ngram_feats = list(ngrams(input, n, True, True, '[b]', '[e]'))
	return ngram_feats
```



