import os
import json
import numpy as np
import pandas as pd
import pickle as pkl
import fasttext as ft
import tensorflow as tf
import src.models.nn as nn
from src.dsm.ft import embed_string
import src.preprocessing.phon as phon
import src.preprocessing.letters as lett
from sklearn.linear_model import ElasticNet as enet
from sklearn.linear_model import Ridge as ridgeregr
from sklearn.linear_model import Lasso as lassoregr
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.linear_model import LinearRegression as linregr
from sklearn.model_selection import LeaveOneOut, StratifiedKFold

os.environ["OMP_NUM_THREADS"] = "32"

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

### COMBINATIONS TO DO:

### COMBINATIONS DONE:
# ngrams: M=5, LEXICAL=True
# bigrams: M=2, LEXICAL=False

MODEL = 'nn'                                # alternatives: nn
FEATURES = 'embedding'                      # alternatives: embedding, phon_features, unigrams
LEXICAL = False
M = 2                                       # alternatives: 2, 5
TARGET = 'rating.mean'
GROUP = 'type'
ITEM = 'name'
ATTRIBUTES = ['gender', 'age', 'polarity']
HYPERPARAMETERS = json.load(open('output/gridSearch/hyperparameters.json', 'rb'))
OUT_DIR = 'output/featureImportance/{}'.format(MODEL)
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

# read in names, their attributes and behavioral ratings
df = pd.read_csv('data/avgRatings_annotated.csv')

# read in feature spaces: regular feature vectors are used for training
if FEATURES == 'embedding':
    if LEXICAL and M == 0:
        ft_model = ft.load_model('output/FTmodels/coca_clean_d300_w5_m0_M0_rho0.7869.bin')
    else:
        ft_model = ft.load_model('output/FTmodels/coca_clean_d300_w5_m2_M5_rho0.7953.bin')

    df_embeddings = pd.DataFrame(
        pkl.load(open('output/embeddings/name_embeddings_ngram{}_lexical_{}.pkl'.format(M, LEXICAL), 'rb'))
    )
    df[ITEM] = df[ITEM].map(lambda x: x.lower())
    df = pd.merge(
        df_embeddings[[ITEM, FEATURES]],
        df[[ITEM, GROUP, 'attribute', TARGET]],
        on=ITEM
    )

elif FEATURES == 'unigrams':
    df = lett.unigrams(df, ITEM, FEATURES)

elif FEATURES == 'phon_features':
    df = phon.lett2phon(df, ITEM, 'ipa')
    df = phon.extra_names('data/na_names_phonTrans.csv', df, ITEM, 'ipa')
    df = phon.phon2features(df, 'ipa', FEATURES)
else:
    raise ValueError('Unrecognized feature {}!'.format(FEATURES))

# initialize data splitters
loo = LeaveOneOut()
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for attribute in ATTRIBUTES:

    print('##############{}##############'.format(attribute))
    df_subset = df.loc[df['attribute'] == attribute].reset_index()
    df_subset['{}_unit'.format(TARGET)] = df_subset[TARGET] / 50

    nametype = df_subset[GROUP]
    names = df_subset[ITEM]
    X = np.array([vector for vector in df_subset[FEATURES]], dtype=float)
    y = np.array(df_subset['{}_unit'.format(TARGET)], dtype=float)
    predictions = []

    # pull out the test name
    for dev_indices, test_index in loo.split(X):

        test_name = names.iloc[test_index].values[0]
        print(test_index, test_name, nametype[test_index].values[0])
        # encode test name
        if FEATURES == 'unigrams':
            test_name_dict = lett.loo_features(test_name)
        elif FEATURES == 'phon_features':
            ipa_transcription = df_subset.loc[df_subset[ITEM] == test_name, 'ipa'].values[0]
            test_name_dict = phon.loo_features(ipa_transcription)
        else:
            test_name_dict = embed_string(test_name, ft_model, min=2, max=M, lexical=LEXICAL, loo=True)
            # also consider the option where the full form is removed, if the input embeddings consider it
            print()
            if LEXICAL:
                test_name_dict[test_name] = embed_string(test_name, ft_model, min=2, max=M, lexical=False, loo=False)

        if MODEL == 'nn':
            # apply stratified sampling on the LOOCV dev split using the name type to stratify, to get training and
            # validation items (for early stopping) that are balanced in terms of name type
            type_dev = nametype[dev_indices]
            train_indices, val_indices = list(skf.split(dev_indices, type_dev))[0]

            x_train, x_val = tf.convert_to_tensor(X[train_indices]), tf.convert_to_tensor(X[val_indices])
            y_train, y_val = tf.convert_to_tensor(y[train_indices]), tf.convert_to_tensor(y[val_indices])
            input_dim = x_train.shape[1]

            model = nn.make(
                input_dim,
                units=HYPERPARAMETERS[FEATURES][attribute]['units'],
                dropout=HYPERPARAMETERS[FEATURES][attribute]['dropout_rate'],
                act=HYPERPARAMETERS[FEATURES][attribute]['act'],
                n_layers=HYPERPARAMETERS[FEATURES][attribute]['n_layers'],
                lr=HYPERPARAMETERS[FEATURES][attribute]['lr']
            )
            callback = EarlyStopping(monitor='val_loss', patience=7, verbose=0)
            model.fit(x_train, y_train, validation_data=(x_val, y_val),
                      epochs=150, batch_size=8, callbacks=[callback], verbose=0)

        elif MODEL == 'linear':
            alpha = HYPERPARAMETERS[FEATURES][attribute]['alpha']
            ratio = HYPERPARAMETERS[FEATURES][attribute]['l1_ratio']
            # use the full LOOCV dev split as training since no early stopping needed
            x_train, y_train = tf.convert_to_tensor(X[dev_indices]), tf.convert_to_tensor(y[dev_indices])
            if alpha > 0:
                if ratio == 1:
                    model = lassoregr(alpha=alpha, max_iter=5000)
                elif ratio == 0:
                    model = ridgeregr(alpha=alpha, max_iter=5000, solver='svd')
                else:
                    model = enet(alpha=alpha, l1_ratio=ratio, max_iter=5000)
            else:
                model = linregr()
            model.fit(x_train, y_train)
        else:
            raise ValueError('Unknown model {}!'.format(MODEL))

        for key, vec in test_name_dict.items():
            # predict the score for the test set (name)
            x_test = tf.reshape(tf.convert_to_tensor(vec), (1,input_dim)) if MODEL == 'nn' else np.array(vec)
            prediction = model.predict(x_test)
            record = {
                'name': test_name,
                'type': nametype[test_index].values[0],
                'attribute': attribute,
                'excluded': key,
                'feature_vector': vec,
                'lexical': LEXICAL,
                'features': FEATURES,
                'max_ngram': M,
                'model': MODEL,
                'true_rating': y[test_index][0],
                'predicted_rating': prediction[0][0]
            }
            predictions.append(record)

    pd.DataFrame(predictions).to_csv(
        os.path.join(OUT_DIR, '{}_featureImp_{}_M{}.csv'.format(attribute, FEATURES, M)),
        sep='\t', index=False, index_label=False
    )