"""
Methods for training a model, including 
- train test loop 
- interpretibility check 
"""

# load packages 
import os 
import pickle
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr

import matplotlib.pyplot as plt 

from sklearn.metrics import mean_squared_error
# load models 
from src.models.models import (
    RidgeRegressor,
    LassoRegressor,
    RFRegressor,
    XGBoostRegressor,
    LightGBMRegressor
)

# specify paths 
SAVE_FEAT_PATH = 'data/preprocessed/'
SAVE_MODEL_PATH = 'saved_models/'
SAVE_IMAGE_PATH = 'images/'


# ====================
# ----- loading ------
# ====================

def load_train_test():
    """
    read and return train_X, train_y, test_X, test_y 
    """
    train_X = sparse.load_npz(os.path.join(SAVE_FEAT_PATH, 'train_X.npz'))
    train_y = np.load(os.path.join(SAVE_FEAT_PATH, 'train_y.npy')).reshape(1, -1).squeeze()
    test_X = sparse.load_npz(os.path.join(SAVE_FEAT_PATH, 'test_X.npz'))
    test_y = np.load(os.path.join(SAVE_FEAT_PATH, 'test_y.npy')).reshape(1, -1).squeeze()
    return train_X, train_y, test_X, test_y


# ===============================
# -------- experiments ----------
# ===============================

def train_test_all_models(train_X, train_y, test_X, test_y):
    """
    try out all models
    """
    # # ridge 
    # ridge_model = RidgeRegressor(
    #     train_X, train_y, test_X, test_y, 
    #     use_subset=True,
    #     alpha=0.1
    # )
    # train_test_loop(ridge_model)

    # # lasso 
    # lasso_model = LassoRegressor(
    #     train_X, train_y, test_X, test_y,
    #     use_subset=True,
    #     alpha=0.1
    # )
    # train_test_loop(lasso_model)

    # # random forest 
    # rf_model = RFRegressor(
    #     train_X, train_y, test_X, test_y, 
    #     use_subset=True,
    #     n_estimators=100, max_depth=30, n_jobs=-1
    # )
    # train_test_loop(rf_model)

    # # XGBoost 
    # xgb_model = XGBoostRegressor(
    #     train_X, train_y, test_X, test_y, 
    #     use_subset=True,
    #     n_estimators=1500, max_depth=30, n_jobs=-1
    # )
    # train_test_loop(xgb_model)

    # lightGBM 
    lgbm_model = LightGBMRegressor(
        train_X, train_y, test_X, test_y,
        n_estimators=5000, max_depth=30
    )
    train_test_loop(lgbm_model)


def train_test_loop(model):
    """ 
    for each model, perform the following train test methods 
    
    :param model: the model from the Regressor class 
    """
    model.train()         # train 
    print('{} Training Complete'.format(model.model_full_name))
    model.save_model()    # serialize and save 
    model.test()          # test 
    print('{} Testing Complete'.format(model.model_full_name))


# =====================================
# ----- eval (interpretibility) -------
# =====================================

def prediction_mean_rank_correlation(model_name, test_df, test_X, test_y):
    """
    Compute mean rank correlation coefficients 
    """
    # read model 
    model = pickle.load(open(os.path.join(SAVE_MODEL_PATH, model_name), 'rb'))

    # predictions 
    test_predictions = model.predict(test_X)

    # by condition 
    distinct_conditions = test_df['condition'].unique()
    rs = []
    for condition in distinct_conditions:
        idx = np.where(test_df['condition'] == condition)
        condition_rank_correlation, _ = spearmanr(test_y[idx], test_predictions[idx])
        if not np.isnan(condition_rank_correlation):
            rs.append(condition_rank_correlation)
    mean_r = np.mean(rs)
    # log 
    with open(os.path.join(SAVE_MODEL_PATH, 'rank_corr.txt'), 'a') as f:
        f.write('{} {}\n'.format(model_name, mean_r))

def visualize_prediction_mean_rank_correlation():
    """
    after obtaining the prediction mean rank correlations, 
    make a plot with different lines 
    """
    # read 
    rank_corr_df = pd.read_csv(os.path.join(SAVE_MODEL_PATH, 'rank_corr.txt'), sep=' ')
    mse_df = pd.read_csv(os.path.join(SAVE_MODEL_PATH, 'results.txt'), sep=' ', usecols=[0, 1, 2])
    rank_corr_df = rank_corr_df.merge(mse_df[['model', 'test_mse']], on='model', how='left').drop_duplicates('model', keep='last')
    # split model 
    ridge_models = rank_corr_df.loc[rank_corr_df['model'].str.contains('Ridge')].sort_values('test_mse')
    # print(ridge_models)
    lasso_models = rank_corr_df.loc[rank_corr_df['model'].str.contains('Lasso')].sort_values('test_mse')
    rf_models = rank_corr_df.loc[rank_corr_df['model'].str.contains('RFR')].sort_values('test_mse')
    lgbm_models = rank_corr_df.loc[rank_corr_df['model'].str.contains('LightGBM')].sort_values('test_mse')
    xgb_models = rank_corr_df.loc[rank_corr_df['model'].str.contains('XGB')].sort_values('test_mse')
    # plot
    plt.figure(figsize=(12, 7))
    plt.plot(ridge_models['test_mse'], ridge_models['rank_corr'], 'o-') 
    plt.plot(lasso_models['test_mse'], lasso_models['rank_corr'], 'o-')
    plt.plot(rf_models['test_mse'], rf_models['rank_corr'], 'o-')
    plt.plot(lgbm_models['test_mse'], lgbm_models['rank_corr'], 'o-')
    plt.plot(xgb_models['test_mse'], xgb_models['rank_corr'], 'o-')
    plt.legend(['Ridge', 'Lasso', 'RandomForest', 'LightGBM', 'XGBoost'])
    plt.xlabel('Test MSE')
    plt.ylabel('Mean Rank Correlation')
    plt.title('Model Refinements: Mean Rank Correlation and Test MSE')
    plt.savefig(os.path.join(SAVE_IMAGE_PATH, 'rank_corr_mse.png'), dpi=300)


def mse_by_rating(model_name, test_df, test_X, test_y):
    """
    Compute mse for different ratings 
    """
    # read model
    model = pickle.load(open(os.path.join(SAVE_MODEL_PATH, model_name), 'rb'))
    # predictions
    test_predictions = model.predict(test_X)
    # categorize
    negative_rating_idx = np.where(test_df['rating'] <= 4)
    neutral_rating_idx = np.where((test_df['rating'] > 4) & (test_df['rating'] <=7))
    pos_rating_idx = np.where(test_df['rating'] > 7)
    idxs = [negative_rating_idx, neutral_rating_idx, pos_rating_idx]
    mses = []
    for idx in idxs:
        mses.append(mean_squared_error(test_y[idx], test_predictions[idx]))
    # log 
    with open(os.path.join(SAVE_MODEL_PATH, 'mse_by_rating.txt'), 'a') as f:
        f.write('{} {} {} {}\n'.format(
            model_name, 
            *mses
        ))


# ==========================
# ------ top features ------
# ==========================

def top_k_features_linear_model(model_name: str, top_k: int = 10):
    """ 
    return the top k features of a linear model 
    developer note: the best performing linear models are: 
    - _RidgeRegressor_alpha_0.1
    - _LassoRegressor_alpha_0.0001

    :param model_name: the above mentioned model name
    """
    # read 
    model = pickle.load(open(os.path.join(SAVE_MODEL_PATH, model_name), 'rb'))
    feature_names = pd.read_csv(os.path.join(SAVE_FEAT_PATH, 'column_names.csv'), index_col=0, squeeze=True)
    # zip 
    feature_weight_list = list(zip(feature_names, model.coef_))
    feature_weight_list.sort(key=lambda x: -abs(x[1])) 
    # log 
    with open(os.path.join(SAVE_MODEL_PATH, 'top_k_features.txt'), 'a') as f:
        f.write(model_name)
        f.write(' ')
        f.write(str(feature_weight_list[:top_k]))
        f.write('\n')
    
def top_k_features_bagging_model(model_name: str, top_k: int = 10):
    """
    similar definition as above 
    - _RFRegressor_n_estimators_100_max_depth_30_n_jobs_-1
    """
    # read 
    model = pickle.load(open(os.path.join(SAVE_MODEL_PATH, model_name), 'rb'))
    feature_names = pd.read_csv(os.path.join(SAVE_FEAT_PATH, 'column_names.csv'), index_col=0, squeeze=True)
    # zip
    feature_weight_list = list(zip(feature_names, model.feature_importances_))
    feature_weight_list.sort(key=lambda x: -x[1])
    # log
    with open(os.path.join(SAVE_MODEL_PATH, 'top_k_features.txt'), 'a') as f:
        f.write(model_name)
        f.write(' ')
        f.write(str(feature_weight_list[:top_k]))
        f.write('\n')

def top_k_features_xgb(model_name: str, top_k: int = 10):
    """
    weight 
    - _XGBoostRegressor_n_estimators_1500_max_depth_30_n_jobs_-1
    """
    # read 
    model = pickle.load(open(os.path.join(SAVE_MODEL_PATH, model_name), 'rb'))
    feature_names = pd.read_csv(os.path.join(SAVE_FEAT_PATH, 'column_names.csv'), index_col=0, squeeze=True)
    # zip
    feature_importances = model.get_booster().get_score(
        importance_type='weight'
    )
    feature_importances = [(feature_names[int(feat_idx[1:])], feat_imp) for feat_idx, feat_imp in feature_importances.items()]
    feature_importances.sort(key=lambda x: -x[1])
    # log
    with open(os.path.join(SAVE_MODEL_PATH, 'top_k_features.txt'), 'a') as f:
        f.write(model_name)
        f.write(' ')
        f.write(str(feature_importances[:top_k]))
        f.write('\n')
    

def top_k_features_lgbm(model_name: str, top_k: int = 10):
    """
    weight (split in the lgbm context) of features 
    - _LightGBMRegressor_n_estimators_5000_max_depth_30
    """
    # read
    model = pickle.load(open(os.path.join(SAVE_MODEL_PATH, model_name), 'rb'))
    feature_names = pd.read_csv(os.path.join(SAVE_FEAT_PATH, 'column_names.csv'), index_col=0, squeeze=True)
    # zip
    feature_weight_list = list(zip(feature_names, model.feature_importances_))
    feature_weight_list.sort(key=lambda x: -x[1])
    # log
    with open(os.path.join(SAVE_MODEL_PATH, 'top_k_features.txt'), 'a') as f:
        f.write(model_name)
        f.write(' ')
        f.write(str(feature_weight_list[:top_k]))
        f.write('\n')
