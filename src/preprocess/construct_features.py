"""
Store constructed features: made ready to feed to ML models 

Features include: 
- OHE of condition
- BOW of review 
- TF-IDF of review
- Sentiment of review
- rating 
- date? 

Each preprocessed features may be stored separately, to be concatenated at the end. 

Developer notes:
justification for removing numbers in reviews:
- give redundant information ("I will give a 9 out of 10", which is indicated in the rating)
- repetition of the medical instruction ("2.75g every 3 days", which is not interesting)
"""

# load packages
import os  
import re
import numpy as np
import pandas as pd
import multiprocessing as mp 
from scipy import sparse

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer as stemmer
from nltk.sentiment import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

# function local pointer (speed hack)
STOPWORDS = set(stopwords.words('english'))
STOPWORDS.update({
    'year', 'month', 'week', 'day', 'hour', 
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'
})  # TODO: justify this 
SIA = SentimentIntensityAnalyzer().polarity_scores
# STEM = stemmer().stem
STEM = stemmer('english').stem

# specify path 
SAVE_FEAT_PATH = 'data/preprocessed'

# ======================
# ---- select rows -----
# ======================

def remove_meaningless_rows(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """ 
    remove some meaningless rows. Currently found the followings: 
    - if condition is not inputted, then the condition will be replaced by 
      the number of useful counts. Let's discard these samples (1171 in total)
    """
    # subset 
    train_df = train_df.loc[~ train_df['condition'].astype(str).str.contains('</span>')]
    test_df = test_df.loc[~ test_df['condition'].astype(str).str.contains('</span>')]
    # save 
    train_df.to_csv(os.path.join(SAVE_FEAT_PATH, 'drugsComTrain_selected_raw.tsv'), sep='\t')
    test_df.to_csv(os.path.join(SAVE_FEAT_PATH, 'drugsComTest_selected_raw.tsv'), sep='\t')


# ======================
# ----- features -------
# ======================


# ------ features from review ----------

def tokenize_reviews(df: pd.DataFrame, train=True):
    """ 
    tokenize review words, including 
    - lowercase
    - remove punctuations, digits, and other non-utf-8 encodings 
    - remove stop words
    - stem words 
    - remove certain arguably meaningless words (year, month, week, day ...) 

    :param df: the original dataframe 
    :param train: True if using train data, False otherwise 
    """
    def process_each(sentence):
        # &#039; is a non-utf-8 encoding of "'". 
        # also remove digits ?
        punc_digits_removed = re.sub(r'&#039;|[^\w\s]|\d', '', sentence.lower())
        processed_sentence = ' '.join(
            [STEM(x) for x in punc_digits_removed.split() if x not in STOPWORDS]
            # [x for x in punc_digits_removed.split() if x not in STOPWORDS]  # TODO: do we need stemmer? 
        )
        return processed_sentence
    # process 
    reviews = df['review']
    processed_reviews = reviews.apply(process_each)
    # save 
    processed_reviews.to_csv(
        os.path.join(SAVE_FEAT_PATH, 'tokenized_reviews_{}.csv'.format('train' if train else 'test'))
    )

def read_tokenized_reviews():
    """
    return processed tokenized reviews, both train and test
    """
    train_reviews = pd.read_csv(
        os.path.join(SAVE_FEAT_PATH, 'tokenized_reviews_train.csv'),
        index_col=0,
        squeeze=True
    )
    test_reviews = pd.read_csv(
        os.path.join(SAVE_FEAT_PATH, 'tokenized_reviews_test.csv'), 
        index_col=0,
        squeeze=True
    )
    return train_reviews, test_reviews


def get_review_sentimens(df: pd.DataFrame, train=True):
    """
    compute review sentiment scores 
    :param df: 
    :param train: True if using training set, False otherwise
    """
    # compute 
    reviews = df[['review']]  # compute sentiments on the original dataset 
    sentiment_scores = reviews.apply(lambda x: list(SIA(x.item()).values()), result_type='expand', axis=1)
    sentiment_scores.columns = ['neg', 'neu', 'pos', 'compound']  # retain sementics
    # save 
    sentiment_scores.to_csv(
        os.path.join(SAVE_FEAT_PATH, 'sentiment_scores_{}.csv'.format('train' if train else 'test'))
    )


def get_review_tf_idf():
    """
    compute tf-idf of reviews 
    """
    # read preprocessed tokenized reviews
    train_reviews, test_reviews = read_tokenized_reviews()
    train_reviews = train_reviews.apply(lambda x: '' if pd.isna(x) else x)
    test_reviews = test_reviews.apply(lambda x: '' if pd.isna(x) else x)
    train_reviews = train_reviews.to_numpy()
    test_reviews = test_reviews.to_numpy()

    # train 
    tf_idf_model = TfidfVectorizer().fit(train_reviews)  # TODO: specify arguments for tf_idf
    train_reviews_tf_idf = tf_idf_model.transform(train_reviews)
    test_reviews_tf_idf = tf_idf_model.transform(test_reviews)

    # extract sementics 
    terms = tf_idf_model.get_feature_names_out()
    with open(os.path.join(SAVE_FEAT_PATH, 'reviews_tf_idf_terms.npy'), 'wb') as f:
        np.save(f, terms)

    # save 
    sparse.save_npz(os.path.join(SAVE_FEAT_PATH, 'reviews_tf_idf_train.npz'), train_reviews_tf_idf)
    sparse.save_npz(os.path.join(SAVE_FEAT_PATH, 'reviews_tf_idf_test.npz'), test_reviews_tf_idf)


def get_review_bow():
    """
    unigram and bigrams one hot encodings for reviews 
    """
    # read 
    train_reviews, test_reviews = read_tokenized_reviews()
    train_reviews = train_reviews.apply(lambda x: '' if pd.isna(x) else x)
    test_reviews = test_reviews.apply(lambda x: '' if pd.isna(x) else x)
    train_reviews = train_reviews.to_numpy()
    test_reviews = test_reviews.to_numpy()

    # train 
    count_vec_model = CountVectorizer(ngram_range=(1, 2)).fit(train_reviews)
    train_reviews_bow = count_vec_model.transform(train_reviews)
    test_reviews_bow = count_vec_model.transform(test_reviews)

    # extract sementics 
    terms = count_vec_model.get_feature_names_out()
    with open(os.path.join(SAVE_FEAT_PATH, 'reviews_bow_terms.npy'), 'wb') as f:
        np.save(f, terms)

    # save 
    sparse.save_npz(os.path.join(SAVE_FEAT_PATH, 'reviews_bow_train.npz'), train_reviews_bow)
    sparse.save_npz(os.path.join(SAVE_FEAT_PATH, 'reviews_bow_test.npz'), test_reviews_bow)


# ------------ features from drugName ---------------

def get_drug_name_ohe(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """ ohe of drug names """
    # extract
    train_drug_names = train_df[['drugName']]
    test_drug_names = test_df[['drugName']]
    # train
    ohe = OneHotEncoder(
        drop='first',       # guarantee linear model convergences
        dtype='int',        # save spaces
        handle_unknown='ignore'
    ).fit(train_drug_names)
    train_drug_names_ohe = ohe.transform(train_drug_names)
    test_drug_names_ohe = ohe.transform(test_drug_names)

    # extract sementics 
    terms = ohe.get_feature_names_out()
    with open(os.path.join(SAVE_FEAT_PATH, 'drug_names_ohe_terms.npy'), 'wb') as f:
        np.save(f, terms)

    # save
    sparse.save_npz(os.path.join(SAVE_FEAT_PATH, 'drug_names_ohe_train.npz'), train_drug_names_ohe)
    sparse.save_npz(os.path.join(SAVE_FEAT_PATH, 'drug_names_ohe_test.npz'), test_drug_names_ohe)


# ------------ features from condition ---------------

def get_condition_ohe(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """ ohe of condition """
    # extract 
    train_conditions = train_df[['condition']]
    test_conditions = test_df[['condition']]
    # train 
    ohe = OneHotEncoder(
        drop='first',       # guarantee linear model convergences 
        dtype='int',        # save spaces 
        handle_unknown='ignore'   
    ).fit(train_conditions)
    train_conditions_ohe = ohe.transform(train_conditions)
    test_conditions_ohe = ohe.transform(test_conditions)

    # extract sementics 
    terms = ohe.get_feature_names_out()
    with open(os.path.join(SAVE_FEAT_PATH, 'conditions_ohe_terms.npy'), 'wb') as f:
        np.save(f, terms)
    
    # save 
    sparse.save_npz(os.path.join(SAVE_FEAT_PATH, 'conditions_ohe_train.npz'), train_conditions_ohe)
    sparse.save_npz(os.path.join(SAVE_FEAT_PATH, 'conditions_ohe_test.npz'), test_conditions_ohe)

# =======================================
# --------- Concat All ------------------
# =======================================

def construct_X(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """ concat all features """
    # read 
    sentiments_train = pd.read_csv(os.path.join(SAVE_FEAT_PATH, 'sentiment_scores_train.csv'), index_col=0)
    sentiments_test = pd.read_csv(os.path.join(SAVE_FEAT_PATH, 'sentiment_scores_test.csv'), index_col=0)
    reviews_bow_train = sparse.load_npz(os.path.join(SAVE_FEAT_PATH, 'reviews_bow_train.npz'))
    reviews_bow_test = sparse.load_npz(os.path.join(SAVE_FEAT_PATH, 'reviews_bow_test.npz'))
    reviews_tf_idf_train = sparse.load_npz(os.path.join(SAVE_FEAT_PATH, 'reviews_tf_idf_train.npz'))
    reviews_tf_idf_test = sparse.load_npz(os.path.join(SAVE_FEAT_PATH, 'reviews_tf_idf_test.npz'))
    conditions_ohe_train = sparse.load_npz(os.path.join(SAVE_FEAT_PATH, 'conditions_ohe_train.npz'))
    conditions_ohe_test = sparse.load_npz(os.path.join(SAVE_FEAT_PATH, 'conditions_ohe_test.npz'))
    drug_names_ohe_train = sparse.load_npz(os.path.join(SAVE_FEAT_PATH, 'drug_names_ohe_train.npz'))
    drug_names_ohe_test = sparse.load_npz(os.path.join(SAVE_FEAT_PATH, 'drug_names_ohe_test.npz'))
    # concat 
    original_feats_train = train_df[['rating']]
    original_feats_test = test_df[['rating']]
    train_X = sparse.hstack((
        drug_names_ohe_train,
        conditions_ohe_train,
        reviews_bow_train,
        reviews_tf_idf_train,
        sentiments_train.to_numpy(),
        original_feats_train.to_numpy()  
    ))
    test_X = sparse.hstack((
        drug_names_ohe_test,
        conditions_ohe_test,
        reviews_bow_test,
        reviews_tf_idf_test,
        sentiments_test.to_numpy(),
        original_feats_test.to_numpy()
    ))
    # save 
    sparse.save_npz(os.path.join(SAVE_FEAT_PATH, 'train_X.npz'), train_X)
    sparse.save_npz(os.path.join(SAVE_FEAT_PATH, 'test_X.npz'), test_X)

    # sementics (column names, in the correct order)
    drug_names_terms = np.load(os.path.join(SAVE_FEAT_PATH, 'drug_names_ohe_terms.npy'), allow_pickle=True)
    condition_terms = np.load(os.path.join(SAVE_FEAT_PATH, 'conditions_ohe_terms.npy'), allow_pickle=True)
    reviews_bow_terms = np.load(os.path.join(SAVE_FEAT_PATH, 'reviews_bow_terms.npy'), allow_pickle=True)
    reviews_tf_idf_terms = np.load(os.path.join(SAVE_FEAT_PATH, 'reviews_tf_idf_terms.npy'), allow_pickle=True)
    columns = (
        list(drug_names_terms) 
        + list(condition_terms) 
        + list(reviews_bow_terms) 
        + list(reviews_tf_idf_terms)
        + ['neg', 'neu', 'pos', 'compound']
        + ['rating']
    )
    # save 
    pd.Series(columns).to_csv(os.path.join(SAVE_FEAT_PATH, 'column_names.csv'))


# TODO: remove 0? Quantile transformation?
def construct_y(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """ appropriate processing of y values """
    # extract 
    train_y = train_df[['usefulCount']]
    test_y = test_df[['usefulCount']]

    # quantile transformer 
    # two benefits: 1. more interpretability; 2. normal more amenable to the MSE metric
    qt = QuantileTransformer(
        n_quantiles=10000,              # TODO: number of quantiles to be determined
        output_distribution='normal'
    ).fit(train_y)
    train_y_transformed = qt.transform(train_y)
    test_y_transformed = qt.transform(test_y)

    # save 
    with open(os.path.join(SAVE_FEAT_PATH, 'train_y.npy'), 'wb') as f:
        np.save(f, train_y_transformed)
    with open(os.path.join(SAVE_FEAT_PATH, 'test_y.npy'), 'wb') as f:
        np.save(f, test_y_transformed)
