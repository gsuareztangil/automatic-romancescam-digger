## Descriptions Classifier

This page explains the descriptions classifier currently in use in the [ensemble](../ensemble/README.md). 
For information on the preliminary experimentation which lead to this design, see the [preliminary experiments](preliminaries.md).

### Prerequisites

This code uses Python version 2.7.14. Additionally, the LibShortText machine learning package (download available at https://www.csie.ntu.edu.tw/~cjlin/libshorttext/) is used. The classifier depends on the profile data stored as individual single-line JSON files, named according to their page name from the source, see the [data](../data/README.md) directory. 

### Training and output

Run the classifier script from the command-line using

`$ python run_descr_classifier.py trainDir testDir validationDir`

These directories should contain all JSON files assigned to training, testing and validation, see the [data](../data/README.md) directory. Two files will be created locally: `test_results.csv` and `validation_results.csv`. These are later used by the ensemble classifier.
