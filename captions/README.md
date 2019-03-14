## Captions Classifier

This page explains the captions classifier currently in use in the [ensemble](../ensemble/README.md). 
For information on the extraction of captions, see the [feature extraction](features.md).

### Prerequisites

Python sklearn kit.

### Training

```
Usage: classifier.py [options]

Options:
  -h, --help            show this help message and exit
  -i INPUT, --input=INPUT
                        path : path to folder containing input files
  -d, --debug           debug
  -s, --given_split     Uses holdout validation with a given split
```

Example

`python classifier.py -i inputSamples --given_split`
