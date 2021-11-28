"""
Methods for training a model 
"""

# load packages 
import os 
import numpy as np
from scipy import sparse

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
    # lightGBM 
    lgbm_model = LightGBMRegressor(
        train_X, train_y, test_X, test_y,
        n_estimators=5000
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
