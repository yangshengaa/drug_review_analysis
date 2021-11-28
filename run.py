"""
run this module
data: download data 
eda: ...
train: ...
test: ...
"""

# load packages
import os 
import sys 
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# load files (EXPLICIT PASSING TO PREVENT LOADING TOO MANY STUFFS!)

# data 
from src.data.data_downloader import download_drug_reviews
from src.data.data_loader import load_data, partition_data

# eda 
from src.eda.visualize import (
    plot_basic_stats, 
    plot_condition_distribution,
    plot_useful_counts_distribution,
    plot_useful_counts_groupby_condition,
    plot_useful_counts_groupby_rating,
    plot_word_cloud,
    plot_unigram_emb,
    plot_bigram_emb,
    plot_rating_useful_counts_ts
)

# preprocessing 
from src.preprocess.construct_features import (
    tokenize_reviews,
    get_review_sentimens,
    get_review_bow,
    get_review_tf_idf,
    get_condition_ohe,
    get_drug_name_ohe,
    construct_X,
    construct_y,
    remove_meaningless_rows,

)

# models
from src.models.train_test_models import (
    load_train_test, 
    train_test_all_models, 
    prediction_mean_rank_correlation
)

# ==================
# ----- main -------
# ==================

def main(targets):
    """
    TODO: explain targets variable
    """
    # download data 
    if 'download' in targets:
        print('Downloading Data ...')
        download_drug_reviews()
        print('Download Complete')
    
    # partition to obtain dev
    if 'partition' in targets:
        print('Partitioning Data')
        partition_data()
        print('Partition Complete')
    
    # preprocessing (about 5 mins to run in total)
    if 'preprocess' in targets:
        print('Preprocessing ...')
        train_df = load_data(train=True, selected=False, dev=False)
        test_df = load_data(train=False, selected=False, dev=False)
        # remove rows 
        remove_meaningless_rows(train_df, test_df)

        # re-read data from the meaningful rows
        train_df = load_data(train=True, selected=True, dev=False)
        test_df = load_data(train=False, selected=True, dev=False)
        # on original datasets 
        tokenize_reviews(train_df, train=True)
        tokenize_reviews(test_df, train=False)
        get_review_sentimens(train_df, train=True)
        get_review_sentimens(test_df, train=False)
        get_condition_ohe(train_df, test_df)
        get_drug_name_ohe(train_df, test_df)

        # on preprocessed tokens 
        get_review_bow()
        get_review_tf_idf()

        # concat all 
        construct_X(train_df, test_df)
        construct_y(train_df, test_df)
        print('Preprocessing Complete')
    
    # EDA
    if 'eda' in targets:
        print("EDA ...")
        # plots on original dataset
        train_df = load_data(train=True, selected=True, dev=False)
        test_df = load_data(train=False, selected=True, dev=False)
        full_df = pd.concat([train_df, test_df])
        # plot_basic_stats(full_df)
        # plot_useful_counts_distribution(full_df)
        # plot_condition_distribution(full_df)
        # plot_useful_counts_groupby_condition(full_df)
        # plot_useful_counts_groupby_rating(full_df)
        # plot_rating_useful_counts_ts(full_df)

        # # plots on preprocessed data
        plot_word_cloud()
        # plot_unigram_emb()
        # plot_bigram_emb()
        print('EDA Complete')

    # train test 
    if 'train' in targets or 'test' in targets:
        print('Start Training')
        # read 
        train_X, train_y, test_X, test_y = load_train_test()
        # test 
        train_test_all_models(train_X, train_y, test_X, test_y)
        print('Training Complete')
        

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
