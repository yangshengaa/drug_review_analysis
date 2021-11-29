"""
Methods for training a model, including 
- train test loop 
- interpretibility check 
"""

# load packages 
import os 
import pickle
import numpy as np
from scipy import sparse
from scipy.stats import spearmanr

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
    with open(os.path.joins(SAVE_MODEL_PATH, 'rank_corr.txt'), 'a') as f:
        f.write('{} {}\n'.format(model_name, mean_r))


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
