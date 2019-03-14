## Descriptions Classifier

This page explains the descriptions classifier currently in use in the [ensemble](../ensemble/README.md). 
For information on the preliminary experimentation which lead to this design, see the [preliminary experiments](preliminaries.md).

### Prerequisites

This code uses Python version 2.7.14. Additionally, the LibShortText machine learning package (download available at https://www.csie.ntu.edu.tw/~cjlin/libshorttext/) is used. The classifier depends on the profile data stored as individual single-line JSON files, named according to their page name from the source, see the [data](../data/README.md) directory. 

### Training and output

After adjusting the path to your LibShortText directory, run the classifier script from the command-line using

`$ python classify.py trainDir testDir validationDir`

The three given directories should contain all JSON files assigned to training, testing and validation, respectfully (see the [data](../data/README.md) directory). A new directory will be created locally called `description_classification`, which contains 3 subdirectories containing the generated document instances, the modelfiles and the outfiles. Additionally, 2 files will be created: `test_results.csv` and `validation_results.csv`. These are later used by the ensemble classifier.
