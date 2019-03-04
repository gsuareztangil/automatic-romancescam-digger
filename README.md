# Automatic Romance Scam Classifier

This repository holds resources for a number of classifiers and preliminary experiments aimed at identifying online dating fraud profiles using only their accessible profile elements.

If you make use of these resources, please cite: (CITATION)


### Data

Due to ethical constraints, we do not host a static dataset. Instead, we provide scripts which allow for data to be scraped from a public scamlist and dating site. For more information, see [data/](data/README.md)

### Ensemble Classifier

The approach we use is an SVM ensemble classifier based on three components: a RF+NB classifier operating on multiple profile [demographic information](demographics/README.md), an SVM classifier operating on [self-descriptions](descriptions/README.md) and an SVM classifier operating on [automatically-extracted image captions](captions/README.md). The code for the [ensemble](ensemble/README.md) relies on each of these components.

### Appendices

A number of additional experiments and results pertaining to this dataset are available in [appendices](appendices/README.md).
