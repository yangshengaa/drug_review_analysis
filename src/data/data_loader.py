"""
load data into memory
"""

# load packages 
import os 
import numpy as np 
import pandas as pd

# specify path 
RAW_DATA_PATH = 'data/raw'
DEV_DATA_PATH = 'data/dev'
SAVE_FEAT_PATH = 'data/preprocessed'


def load_data(train=True, selected=True, dev=False):
    """ 
    load data into memory
    :param train: True if load train data, False otherwise 
    :param selected: True if use selected rows (after removing meaningless rows)
    :param dev: True if use the entire raw data, otherwise use subsets only (dev)
    :return data in a pandas dataframe format 
    """
    # specify paths 
    file_path = 'drugsCom{}_{}.tsv'.format(
        'Train' if train else 'Test',
        'selected_' * selected + ('dev' if dev else 'raw') 
    )
    if selected: 
        meta_path = SAVE_FEAT_PATH
    elif dev:
        meta_path = DEV_DATA_PATH
    else: 
        meta_path = RAW_DATA_PATH
    data_path = os.path.join(meta_path, file_path)
    # load into memory 
    df = pd.read_table(data_path, index_col=0) 
    return df


# ======================
# ------ dev only ------
# ======================

def partition_data():
    """
    partition the dataset into smaller subsets for testing purposes
    """
    # load data 
    train = load_data(train=True)
    test = load_data(train=False)
    # subset
    train_subset = train.iloc[:20000]
    test_subset = test.iloc[:2000]
    # save 
    train_subset.to_csv(os.path.join(DEV_DATA_PATH, 'drugsComTrain_dev.tsv'), sep='\t')
    test_subset.to_csv(os.path.join(DEV_DATA_PATH, 'drugsComTest_dev.tsv'), sep='\t')
