"""
All regression models, including: 
- RidgeRegressor
- LassoRegressor
- RFRegressor
- XGBoostRegressor
- LightGBMRegressor 

(delibrately use a different naming from the original one to eliminate confusion)

Each model is uniquely determined by a tuple 
(model_name, model_type, model_params)

For instance, if the tuple is 
(dev, RidgeRegressor, {C: 1})
then the model saved will be named 
'dev_RidgeRegressor_C_1'
"""

# TODO: add more regressors for comparison

# load packages 
import os 
import pickle 
import random
import numpy as np
import pandas as pd 
from datetime import datetime

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# specify path 
SAVE_MODEL_PATH = 'saved_models/'

# TODO: add a bunch of printing statements

# ==========================
# ----- master class -------
# ==========================

class Regressor():
    """ An Interface of all tested Regressors """
    def __init__(
            self, 
            train_X, 
            train_y, 
            test_X, 
            test_y, 
            model_name='',
            use_subset=False,
            **kwargs
        ):
        """ 
        :param train_X, train_y, test_X, test_y: as usual 
        :param model_name: facilitate saving different models (dev vs test etc)
        :param kwargs: all kwargs are for model parameters 
        """
        # store train and test 
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        self.use_subset = use_subset
        self.parameters = kwargs
        # specify model type 
        self.model = None
        self.model_name = model_name

        # io related
        self.model_full_name = '{}_{}_{}'.format(
            self.model_name,
            self.__class__.__name__,
            self.parse_parameters_to_str(self.parameters)
        )
        self.model_save_path = os.path.join(SAVE_MODEL_PATH, self.model_full_name)

    # ---------- io --------------
    def save_model(self):
        """ serialize and models """
        pickle.dump(self.model, open(self.model_save_path, 'wb'))

    def load_model(self):
        """ read serialized saved model """
        self.model = pickle.load(open(self.model_save_path, 'rb'))
    
    # --------- train test ---------
    def train(self):
        """ training process """
        if self.use_subset:
            subset_idx = random.sample(range(self.train_X.shape[0]), 50000)
            subset_train_X = self.train_X[subset_idx]
            subset_train_y = self.train_y[subset_idx]
            self.model.fit(subset_train_X, subset_train_y)
        else:
            self.model.fit(self.train_X, self.train_y)
    
    def test(self):
        """ testing process """
        # predict 
        train_predictions = self.model.predict(self.train_X)
        test_predictions = self.model.predict(self.test_X)
        train_mse = mean_squared_error(self.train_y, train_predictions)
        test_mse = mean_squared_error(self.test_y, test_predictions)
        # log 
        with open(os.path.join(SAVE_MODEL_PATH, 'results.txt'), 'a') as f:
            f.write('{} {} {} {}\n'.format(
                self.model_full_name,
                train_mse,
                test_mse, 
                datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            ))

    # -------- auxciliary -----------
    def parse_parameters_to_str(self, kwargs: dict):
        """ 
        for saving purposes: connect all names of parameters and parameters with _
        :param kwargs: the model parameters
        :return a '_' connected kwargs representation
        """
        param_list = [list(t) for t in kwargs.items()]
        all_param_list = sum(param_list, [])
        str_all_param_list = [str(x) for x in all_param_list]
        parsed_params = '_'.join(str_all_param_list)
        return parsed_params

# ==========================
# ----- sub classes --------
# ==========================

# delibrately use a different name from 
# the original model to make a difference

class RidgeRegressor(Regressor):
    def __init__(
        self, 
        train_X, train_y, test_X, test_y, 
        use_subset=False,
        model_name='', 
        **kwargs
    ):
        super().__init__(
            train_X, train_y, test_X, test_y, 
            use_subset=use_subset,
            model_name=model_name, 
            **kwargs
        )
        self.model = Ridge(**kwargs)

    
class LassoRegressor(Regressor):
    def __init__(
        self,
        train_X, train_y, test_X, test_y,
        use_subset=False,
        model_name='',
        **kwargs
    ):
        super().__init__(
            train_X, train_y, test_X, test_y,
            use_subset=use_subset,
            model_name=model_name,
            **kwargs
        )
        self.model = Lasso(**kwargs)


class RFRegressor(Regressor):
    def __init__(
        self,
        train_X, train_y, test_X, test_y,
        use_subset=False,
        model_name='',
        **kwargs
    ):
        super().__init__(
            train_X, train_y, test_X, test_y,
            use_subset=use_subset,
            model_name=model_name,
            **kwargs
        )
        self.model = RandomForestRegressor(**kwargs)

class XGBoostRegressor(Regressor):
    def __init__(
        self,
        train_X, train_y, test_X, test_y,
        use_subset=False,
        model_name='',
        **kwargs
    ):
        super().__init__(
            train_X, train_y, test_X, test_y,
            use_subset=use_subset,
            model_name=model_name,
            **kwargs
        )
        self.model = XGBRegressor(**kwargs)
    
class LightGBMRegressor(Regressor):
    def __init__(
        self,
        train_X, train_y, test_X, test_y,
        use_subset=False,
        model_name='',
        **kwargs
    ):
        super().__init__(
            train_X, train_y, test_X, test_y,
            use_subset=use_subset,
            model_name=model_name,
            **kwargs
        )
        self.model = LGBMRegressor(**kwargs)
