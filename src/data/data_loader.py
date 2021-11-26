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


def load_data(train=True, dev=False):
    """ 
    load data into memory
    :param train: True if load train data, False otherwise 
    :param dev: True if use the entire raw data, otherwise use subsets only (dev)
    :return data in a pandas dataframe format 
    """
    # specify paths 
    file_path = 'drugsCom{}_{}.tsv'.format(
        'Train' if train else 'Test',
        'dev' if dev else 'raw'
    )
    data_path = os.path.join(DEV_DATA_PATH if dev else RAW_DATA_PATH, file_path)
    # load into memory 
    df = pd.read_table(data_path, index_col=0) # TODO: date parser?
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
