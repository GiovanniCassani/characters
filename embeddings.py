import os
import json
import numpy as np
import pandas as pd
import pickle as pkl
import src.models.nn as nn
import src.models.linear as lin

# GRID SEARCH: to do
#
# GRID SEARCH: DONE
# linear: lexical=True, M=5; nn: lexical=True, M=5          all ngrams and lexical

# PREDICTIONS: to do
#

# PREDICTIONS: DONE
# linear: lexical=True, M=0; nn: lexical=True, M=0          lexical only
# linear: lexical=False, M=2; nn: lexical=False, M=2        bigrams only
# linear: lexical=False, M=5; nn: lexical=False, M=5        all ngrams
# linear: lexical=True, M=5; nn: lexical=True, M=5          all ngrams and lexical

GRID = False
MODEL = 'nn'
LEXICAL = False
FEATURES = 'embedding'
M = 2
RND_ITERS = 50
TARGET = 'rating.mean'
GROUP = 'type'
ITEM = 'name'
ATTRIBUTES = ['gender', 'age', 'polarity']
HYPERPARAMETERS = json.load(open('output/gridSearch/hyperparameters.json', 'rb'))

print(LEXICAL, M, MODEL, FEATURES)

df_embeddings = pd.DataFrame(
    pkl.load(open('output/embeddings/name_embeddings_ngram{}_lexical_{}.pkl'.format(M, LEXICAL), 'rb'))
)

df_ratings = pd.read_csv('data/avgRatings_annotated.csv')
df_ratings[ITEM] = df_ratings[ITEM].map(lambda x: x.lower())
df = pd.merge(
    df_embeddings[[ITEM, FEATURES]],
    df_ratings[[ITEM, GROUP, 'attribute', TARGET]],
    on=ITEM
)

for attribute in ATTRIBUTES:
    print('##############{}##############'.format(attribute))
    df_subset = df.loc[df['attribute'] == attribute].reset_index()
    df_subset['{}_unit'.format(TARGET)] = df_subset[TARGET] / 50
    input_dim = len(df_subset[FEATURES][0])

    if GRID:
        out_dir = 'output/gridSearch'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        if MODEL == 'nn':
            search_df = nn.grid_search(
                df=df_subset,
                x_col=FEATURES,
                y_col='{}_unit'.format(TARGET),
                group_col=GROUP,
                units=[25, 50, 150, input_dim, input_dim*2],
                dropout=[0, 0.25, 0.5],
                activations=['tanh', 'sigmoid', 'relu'],
                n_layers=[1],
                learning_rates=[0.001, 0.0001]
            )
        elif MODEL == 'linear':
            search_df = lin.grid_search(
                df=df_subset,
                x_col=FEATURES,
                y_col='{}_unit'.format(TARGET),
                group_col=GROUP,
                alphas=list(np.linspace(0, 5, 101)),
                ratios=list(np.linspace(0, 1, 51))
            )
        search_df.to_csv(
            os.path.join(out_dir, '{}/{}_gridSearch_M{}_lexical_{}.csv'.format(MODEL, attribute, M, LEXICAL)),
            sep='\t', index=False, index_label=False
        )

    else:
        out_dir = 'output/predictions'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        if MODEL == 'nn':
            predictions_df = nn.predict(
                df=df_subset,
                x_col=FEATURES,
                y_col='{}_unit'.format(TARGET),
                items_col=ITEM,
                group_col=GROUP,
                units=HYPERPARAMETERS[FEATURES][attribute]['units'],
                dropout_rate=HYPERPARAMETERS[FEATURES][attribute]['dropout_rate'],
                act=HYPERPARAMETERS[FEATURES][attribute]['act'],
                n_layers=HYPERPARAMETERS[FEATURES][attribute]['n_layers'],
                lr=HYPERPARAMETERS[FEATURES][attribute]['lr']
            )
            for i in range(RND_ITERS):
                df_subset['rnd_{}'.format(FEATURES)] = np.random.permutation(df_subset[FEATURES])
                predictions_df_rnd = nn.predict(
                    df=df_subset,
                    x_col='rnd_{}'.format(FEATURES),
                    y_col='{}_unit'.format(TARGET),
                    items_col=ITEM,
                    group_col=GROUP,
                    units=HYPERPARAMETERS[FEATURES][attribute]['units'],
                    dropout_rate=HYPERPARAMETERS[FEATURES][attribute]['dropout_rate'],
                    act=HYPERPARAMETERS[FEATURES][attribute]['act'],
                    n_layers=HYPERPARAMETERS[FEATURES][attribute]['n_layers'],
                    lr=HYPERPARAMETERS[FEATURES][attribute]['lr']
                )
                predictions_df_rnd.to_csv(
                    os.path.join(
                        out_dir, '{}/{}_predictions_M{}_lexical_{}_rnd{}.csv'.format(MODEL, attribute, M, LEXICAL, i)
                    ),
                    sep='\t', index=False, index_label=False
                )

        elif MODEL == 'linear':
            predictions_df = lin.train(
                df=df_subset,
                x_col=FEATURES,
                y_col='{}_unit'.format(TARGET),
                items_col=ITEM,
                alpha=HYPERPARAMETERS[FEATURES][attribute]['alpha'],
                ratio=HYPERPARAMETERS[FEATURES][attribute]['l1_ratio']
            )
            for i in range(RND_ITERS):
                df_subset['rnd_{}'.format(FEATURES)] = np.random.permutation(df_subset[FEATURES])
                predictions_df_rnd = lin.train(
                    df=df_subset,
                    x_col='rnd_{}'.format(FEATURES),
                    y_col='{}_unit'.format(TARGET),
                    items_col=ITEM,
                    alpha=HYPERPARAMETERS[FEATURES][attribute]['alpha'],
                    ratio=HYPERPARAMETERS[FEATURES][attribute]['l1_ratio']
                )
                predictions_df_rnd.to_csv(
                    os.path.join(
                        out_dir, '{}/{}_predictions_M{}_lexical_{}_rnd{}.csv'.format(MODEL, attribute, M, LEXICAL, i)
                    ),
                    sep='\t', index=False, index_label=False
                )

        predictions_df.to_csv(
            os.path.join(out_dir, '{}/{}_predictions_M{}_lexical_{}.csv'.format(MODEL, attribute, M, LEXICAL)),
            sep='\t', index=False, index_label=False
        )
