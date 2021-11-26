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

# load files 
from src.data.data_downloader import download_drug_reviews
from src.data.data_loader import load_data, partition_data

# ==================
# ----- main -------
# ==================

def main(targets):
    """
    TODO: explain targets variable
    """
    # download data 
    if 'data' in targets:
        download_drug_reviews()
    

    # TODO: delete the block below. Only for developer testing
    if 'dev_test' in targets:
        partition_data()
        print(load_data(dev=True))

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
