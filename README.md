# Assignment 2: Drug Review Dataset Analysis

In this assignment, we use the drug review dataset to predict the useful counts of a given review.

## File Digestion

This project contains the following four folders

**data**:

- raw: raw data (from download)
- dev: subset of raw data, for developer testing purposes
- preprocessed: store preprocessed features, ready to be fed into models for training

**save_models**: dump serialized models here

**src**: stores all the code fore necessary computations

- data: download and partition data
- preprocess: construct features
- models: stores all ML models
- eda: make some (fancy) plots

**images**: stores eda images.

## Citation

TODO: add more

One thing worth mentioning is that most research are done on predicting variables other than useful counts (ratings, conditions, etc). We are, in this sense, in deed groundbreaking! (jk).

data source: <https://archive-beta.ics.uci.edu/ml/datasets/drug+review+dataset+drugs+com>  

related tutorials: <https://medium.com/sfu-cspmp/draw-drug-review-analysis-work-96212ed98941>

related papers: <https://www.sciencedirect.com/science/article/pii/S1665642317300561#bib0105>

## For Developer

Here is the place for developers to jot down ideas. Feel free to populate.

### EDA

The following plots need to be generated:

- count of conditions
- violin plot of useful count for different ratings
- useful count by conditions
- t-SNE of unigram, bigram

### Features

The following features may be helpful

- TF-IDF (after removing stemmers and stop words)
- Sentiment scores (NLTK pretrained)
- condition OHE
- date?

Make sure to serialize model after training!

### Interpretability

The following needs to be examined:

- suspect a lower MSE for higher ratings: well-known phenomenon among patients, who like to look for similar patients, and patients who give abnormally low ratings probably encountered some unrelatable circumstances.
- SHAP values for feature analysis
