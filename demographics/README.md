## Demographics Classifier

This page explains the demographics classifier currently in use in the [ensemble](../ensemble/README.md). 
For information on the preliminary experimentation which lead to this design, see the [preliminary experiments](preliminaries.md).

### Prerequisites

This classifier code uses the R language, version 3.5.1.
The libraries used here and in the preliminary experiments can be installed with the following R commands.

```{r}
depends <- c("e1071","ROCR","PRROC","randomForest","C50","knitr")
install.packages(depends, dependencies=TRUE)
```

The classifier script depends on the availability of cleaned CSV formatted profile data. It will check for
the presence of a `train.csv`, `test.csv` and `validation.csv` in the `data` directory, using the profile
data from training to produce predictions for the test and validation sets.

To produce these files, use the `clean.py` and `csvise.py` scripts from the [data](../data/README.md) directory. 
To make use of location information, also run the `geocomp.py` script.

If you want the classifier script to perform k-fold crossvalidation, a column `fold` should be added to `train.csv`.
Profiles will be grouped according to their fold number. For a randomised 10-fold crossvalidation which prevents
variants of the same scam profile being allocated different folds, use the `redistribute.py` script from the data directory,
(which will also assign training and test sets in a 60/20/20 split).

### Training and Output

Run the classifier script from the command-line

`$ Rscript classify.r`

Or from within an R terminal.

```{r}
source(classify.r)
```

Performance figures on test, validation and within cross-validation (if applicable) will be shown as output.

Two files will be created locally: `test_results.csv` and `validation_results.csv`. These are later used by
the ensemble classifier.

