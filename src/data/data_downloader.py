"""
Download data from source, store it to data/raw
"""

# load packages 
import os 
import urllib
import zipfile

# specify path 
RAW_DATA_PATH = 'data/raw'

def download_drug_reviews():
    """ download drug reviews data """
    # download
    drug_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip'
    drug_zip_datapath = os.path.join(RAW_DATA_PATH, 'drugsCom_raw.zip')
    urllib.request.urlretrieve(drug_url, drug_zip_datapath)
    # unzip
    with zipfile.ZipFile(drug_zip_datapath, 'r') as zip_ref:
        zip_ref.extractall(RAW_DATA_PATH)
    # remove zip 
    os.remove(drug_zip_datapath)
