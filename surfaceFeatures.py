import os
import json
import numpy as np
import pandas as pd
import src.models.nn as nn
import src.models.linear as lin
import src.preprocessing.phon as phon
from src.preprocessing.letters import unigrams

# TO DO
#
# DONE
# linear: phon; nn: phon;
# linear: letters; nn: letters;

GRID = True
MODEL = 'nn'                                    # alternatives: nn, linear,
FEATURES = 'phon_features'                           # alternatives: unigrams, phon_features
RND_ITERS = 50
TARGET = 'rating.mean'
GROUP = 'type'
ITEM = 'name'
ATTRIBUTES = ['gender', 'age', 'polarity']

HYPERPARAMETERS = json.load(open('output/gridSearch/hyperparameters.json', 'rb'))

df = pd.read_csv("data/avgRatings_annotated.csv")
df = phon.lett2phon(df, ITEM, 'ipa')
df = phon.extra_names('data/na_names_phonTrans.csv', df, ITEM, 'ipa')


for attribute in ATTRIBUTES:
    print('##############{}##############'.format(attribute))
    df_subset = df.loc[df['attribute'] == attribute].reset_index()
    if FEATURES == 'unigrams':
        df_subset = unigrams(df_subset, ITEM, FEATURES)
    elif FEATURES == 'phon_features':
        df_subset = phon.phon2features(df_subset, 'ipa', FEATURES)
    else:
        raise ValueError("Unsupported feature type: {}. Choose either 'unigrams' or 'phon_features'.".format(FEATURES))

    df_subset['{}_unit'.format(TARGET)] = df_subset[TARGET] / 50
    input_dim = len(df_subset[FEATURES][0])

    if GRID:
        out_dir = 'output/gridSearch'
        if MODEL == 'nn':
            search_df = nn.grid_search(
                df=df_subset,
                x_col=FEATURES,
                y_col='{}_unit'.format(TARGET),
                group_col=GROUP,
                units=[8, 16, input_dim, input_dim*2],
                dropout=[0, 0.25, 0.5],
                activations=['tanh', 'sigmoid'],
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
        search_df.to_csv('{}/{}/{}_gridSearch_{}.csv'.format(out_dir, MODEL, attribute, FEATURES),
                         sep='\t', index=False, index_label=False)

    else:
        out_dir = 'output/predictions'
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
            print('Baseline: {} random permutations.'.format(RND_ITERS))
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
                    os.path.join(out_dir,
                                 '{}/{}_predictions_{}_rnd{}.csv'.format(MODEL, attribute, FEATURES, i)),
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
            print('Baseline: {} random permutations.'.format(RND_ITERS))
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
                    os.path.join(out_dir,
                                 '{}/{}_predictions_{}_rnd{}.csv'.format(MODEL, attribute, FEATURES, i)),
                    sep='\t', index=False, index_label=False
                )

        predictions_df.to_csv(os.path.join(out_dir, '{}/{}_predictions_{}.csv'.format(MODEL, attribute, FEATURES)),
                              sep='\t', index=False, index_label=False)
