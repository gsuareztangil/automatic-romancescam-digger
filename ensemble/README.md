## Ensemble Classifier

The ensemble classifier uses the predictions made by the three individual component classifiers on the `test` set to learn weights,
and then uses these weights and the component classifiers' predictions for the `validation` set to make final predictions. 

### Prerequisites

This classifier code uses the R language, version 3.5.1.
The libraries used can be installed with the following commands.

```{r}
depends <- c("e1071","ROCR","PRROC")
install.packages(depends, dependencies=TRUE)
```

The classifier script will expect that `test_results.csv` and `validation_results.csv` exist in the subdirectories for each of the
classifier components. 

### Training

Run the classifier script from the command-line

`$ Rscript ensemble.r`

Or from within an R terminal.

```{r}
source(ensemble.r)
```


