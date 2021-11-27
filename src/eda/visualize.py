"""
EDA: make some plots 

plots include: 
- max, mean, .... (check)
- useful count distribution (check)
- ratings distribution (check)
- count of conditions (check)
- useful count by conditions (check)
- word2vec t_SNE of unigram and bigrams
- word cloud

(Select only some of them for final reports)


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

from PIL import Image
from wordcloud import WordCloud

# specify paths 
SAVE_IMAGE_PATH = 'images'
PREPROCESSED_FEAT_PATH = 'data/preprocessed'
plt.style.use('ggplot')  # better rendering 

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
        rot=45,
        ylabel='count',
        fontsize=5,
        figsize=(10, 5)
    ).get_figure()
    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_IMAGE_PATH, 'condition_dist.png'), dpi=300)


# TODO: violin plot or stacked kde? 
def plot_useful_counts_groupby_condition(df: pd.DataFrame):
    """
    plot useful counts groupby condition 
    """
    # compute 
    counts = df['condition'].value_counts()
    top_conditions = counts.index[:10]  # extract top 10

    # plot each kde 
    _, ax = plt.subplots()
    for condition in top_conditions:
        df[df['condition'] == condition]['usefulCount'].plot(
            ax=ax,
            kind='kde',
            xlim=[-5, 200],
            figsize=(12, 6)
        )
    plt.legend(top_conditions, fontsize=8)
    plt.xlabel('useful counts')
    plt.ylabel('density')
    plt.title('Useful Counts kde for Top 10 Conditions')

    # save fig 
    plt.savefig(os.path.join(SAVE_IMAGE_PATH, 'useful_counts_groupby_condition.png'), dpi=300)


def plot_useful_counts_groupby_rating(df: pd.DataFrame):
    """
    plot useful counts groupby ratings
    """
    # compute groupby 
    groupby_df = df[['rating', 'usefulCount']].groupby('rating')['usefulCount'].agg(['mean', 'size']).reset_index()

    # specify x y z 
    x = groupby_df['rating']
    y = groupby_df['mean'].rename('mean_useful_count')
    z = groupby_df['size'].rename('total_counts')

    # plot 
    plt.figure(figsize=(14, 7))
    plt.scatter(
        x, y, c=z, s=150, 
        cmap='magma'       # TODO: I think magma is good enough. But anyone wants a better colormap?
    )  
    cbar = plt.colorbar()
    cbar.set_label('count', rotation=270)
    cbar.ax.tick_params(labelsize=8)
    plt.xlabel('rating')
    plt.ylabel('mean useful counts')
    plt.title('Mean Useful Counts by Rating (market color = count)')
    plt.xticks(x)

    # annotate 
    for i, count in enumerate(z):
        plt.annotate(count, (x[i], y[i]), xytext=(x[i] - 0.3, y[i] + 0.3))  # offset for right dispaly of texts

    # save 
    plt.savefig(os.path.join(SAVE_IMAGE_PATH, 'useful_counts_groupby_rating.png'), dpi=300)


def plot_word_cloud():
    """ 
    word_cloud!
    """
    # read in preprocessed tokens 
    tokens_path_train = os.path.join(
        PREPROCESSED_FEAT_PATH, 'tokenized_reviews_train.csv')
    tokens_path_test = os.path.join(
        PREPROCESSED_FEAT_PATH, 'tokenized_reviews_test.csv')
    train_tokens = pd.read_csv(tokens_path_train)
    test_tokens = pd.read_csv(tokens_path_test)
    full_tokens = pd.concat([train_tokens, test_tokens])
    text = ' '.join(str(review) for review in full_tokens['review'].to_numpy())

    # wordcloud
    pill_bottle_mask = (1 - np.array(
        Image.open(os.path.join(SAVE_IMAGE_PATH, 'pill_bottle_mask.png')).convert('1')
    )) * 254 + 1 # convert to black and white and read in as a 2D array 
    wc = WordCloud(
        background_color='white', 
        max_words=1000,
        mode="RGBA", 
        mask=pill_bottle_mask.astype(np.int32),
        width=3000,
        height=2000,
        colormap='Set2'  # TODO: want a better colormap, maybe ...
    )
    wc.generate(text)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(os.path.join(SAVE_IMAGE_PATH, 'review_word_cloud.png'), dpi=1000)


# TODO: word2vec unigram and bigram t_SNE
