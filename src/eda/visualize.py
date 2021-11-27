"""
EDA: make some plots 

plots include: 
- max, mean, .... (check)
- useful count distribution (check)
- ratings distribution 
- count of conditions (check)
- useful count by conditions 
- word2vec t_SNE of unigram and bigrams


For references:
the columns are: 
- drugName
- condition	
- review	
- rating	
- date	
- usefulCount
"""

# load packages 
import os
import numpy as np
import pandas as pd 
import multiprocessing as mp 
import matplotlib.pyplot as plt 

# specify paths 
SAVE_IMAGE_PATH = 'images'

# =======================
# ------ plots ----------
# =======================

def plot_basic_stats(df: pd.DataFrame):
    """
    report basic stats
    # TODO: explain each stats

    :param df: the full dataset
    # TODO: add more stats
    """
    # compute stats 
    num_distinct_drugs = df[['drugName']].nunique()
    num_distinct_conditions = df[['condition']].nunique()

    rating_stats = df['rating'].agg(['max', 'mean', 'min'])
    rating_stats.index = 'rating_' + rating_stats.index

    useful_count_stats = df['usefulCount'].agg(['max', 'mean', 'min'])
    useful_count_stats.index = 'useful_counts_' + useful_count_stats.index

    # concat
    stats_tbl = pd.concat([
        num_distinct_drugs, 
        num_distinct_conditions, 
        rating_stats, 
        useful_count_stats
    ], axis=0).rename('basic_stats')

    # save 
    stats_tbl.to_csv(os.path.join(SAVE_IMAGE_PATH, 'basic_stats.csv'))


def plot_useful_counts_distribution(df: pd.DataFrame):
    """ 
    plot the useful counts distribution 
    :param df: from the original dataset
    """
    fig = df['usefulCount'].plot(
        kind='kde',
        title='Useful Counts Distribution',
        xlabel='useful counts',
        ylabel='density',
        xlim=[0, 600]
    ).get_figure()
    fig.savefig(os.path.join(SAVE_IMAGE_PATH, 'useful_counts_dist.png'), dpi=300)


def plot_condition_distribution(df: pd.DataFrame):
    """ plot condition distribution """
    counts =  df['condition'].value_counts()
    fig = counts.iloc[:30].plot(    # extract top 30 condition category
        kind='bar',
        title='condition distribution',
        xlabel='condition',
        ylabel='count',
        fontsize=5
    ).get_figure()# .tight_layout()
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_IMAGE_PATH, 'condition_dist.png'), dpi=300)

def plot_useful_counts_groupby_condition(df: pd.DataFrame):
    """
    plot useful counts groupby condition 
    """
    counts = df['condition'].value_counts()
    top_conditions = counts.index[:10]
    for condition in top_conditions:
        df[df['condition'] == condition]['usefulCount'].plot(
            kind='kde',
            xlim=[-5, 200]
        )
    plt.show()
    # fig = df[['condition', 'usefulCount']].groupby('condition').plot(kind='kde')
    # fig.savefig(os.path.join(SAVE_IMAGE_PATH, 'useful_counts_groupby_condition.png'), dpi=300)
