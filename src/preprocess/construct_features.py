"""
Store constructed features: made ready to feed to ML models 

Features include: 
- OHE of condition
- BOW of review 
- TF-IDF of review
- Word2Vec of review 
- Sentiment of review
- rating 
- date? 

Each preprocessed features may be stored separately, to be concatenated at the end. 
"""

# load packages
import os  
import re 
import numpy as np
import pandas as pd
import multiprocessing as mp 

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer as stemmer
from nltk.sentiment import SentimentIntensityAnalyzer

# function local pointer (speed hack)
STOPWORDS = set(stopwords.words('english'))
SIA = SentimentIntensityAnalyzer().polarity_scores
STEM = stemmer().stem

# specify path 
SAVE_FEAT_PATH = 'data/preprocessed'

# ======================
# ----- features -------
# ======================

def tokenize_reviews(df: pd.DataFrame, train=True):
    """ 
    tokenize review words, including 
    - lowercase
    - remove punctuations 
    - remove stop words
    - remove stemmer 

    :param df: the original dataframe 
    :param train: True if using train data, False otherwise 
    """
    def process_each(sentence):
        punc_removed = re.sub(r'[^\w\s]', '', sentence.lower())
        processed_sentence = ' '.join(
            # [STEM(x) for x in punc_removed.split() if x not in STOPWORDS]
            [x for x in punc_removed.split() if x not in STOPWORDS]  # TODO: need stemmer or not? 
        )
        return processed_sentence
    # process 
    reviews = df['review']
    processed_reviews = reviews.apply(process_each)
    # save 
    processed_reviews.to_csv(
        os.path.join(SAVE_FEAT_PATH, 'tokenized_reviews_{}.csv'.format('train' if train else 'test')))


