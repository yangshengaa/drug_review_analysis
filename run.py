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
    mse_by_rating,
    prediction_mean_rank_correlation,
    visualize_prediction_mean_rank_correlation,
    top_k_features_linear_model,
    top_k_features_bagging_model,
    top_k_features_xgb,
    top_k_features_lgbm,
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
        
    if 'analyze' in targets:
        print("Start Analyzing")
        # read 
        train_X, train_y, test_X, test_y = load_train_test()
        test_df = load_data(train=False, selected=True, dev=False)

        # pick best performing models 
        lasso_model_name = '_LassoRegressor_alpha_0.0001'
        ridge_model_name = '_RidgeRegressor_alpha_0.1'
        rf_model_name = '_RFRegressor_n_estimators_100_max_depth_30_n_jobs_-1'
        lgbm_model_name = '_LightGBMRegressor_n_estimators_5000_max_depth_30'
        xgb_model_name = '_XGBoostRegressor_n_estimators_1500_max_depth_30_n_jobs_-1'

        # extract top k features 
        # top_k_features_linear_model(lasso_model_name)
        # top_k_features_linear_model(ridge_model_name)
        # top_k_features_bagging_model(rf_model_name)
        # top_k_features_lgbm(lgbm_model_name)
        # top_k_features_xgb(xgb_model_name)

        # mse by parts 
        # mse_by_rating(lasso_model_name, test_df, test_X, test_y)
        # mse_by_rating(ridge_model_name, test_df, test_X, test_y)
        # mse_by_rating(rf_model_name, test_df, test_X, test_y)
        # mse_by_rating(lgbm_model_name, test_df, test_X, test_y)
        # mse_by_rating(xgb_model_name, test_df, test_X, test_y)

        # rank corr
        # model_names = os.listdir('saved_models/')
        # model_names = [model_name for model_name in model_names if '_alpha_' in model_name or 'max_depth' in model_name]
        # model_names.remove('_RFRegressor_n_estimators_5_max_depth_10_n_jobs_-1')
        # model_names.remove('_LightGBMRegressor_n_estimators_500_max_depth_10')

        # for model_name in model_names:
        #     prediction_mean_rank_correlation(model_name, test_df, test_X, test_y)

        # visualize rank corr 
        visualize_prediction_mean_rank_correlation()
        print('Analysis Complete')

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
