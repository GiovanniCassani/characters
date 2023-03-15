import os
import numpy as np
import pandas as pd
import pickle as pkl
import fasttext as ft
from tqdm import tqdm
from src.dsm.ft import embed_string, get_avg_emb

"""
This script encodes target names as embeddings, relying on either a fully lexical fasttext model (which doesn't learn
embeddings for character ngrams) or a sub-lexical fasttext model (which does learn character embeddings). When using the
lexical model, made-up words, whose lexical form does not exist in the corpus, are embedded with the average lexical
embedding (the average embedding of every word in the vocabulary). When embedding names using the sub-lexical model, 
different n-gram sizes are considered and thus different embeddings are created, from purely relying on 2grams to 
2- and 3grams, 2-, 3-, and 4grams, and 2-, 3-, 4-, and 5grams.
fasttext models loaded here are generated using the train_ft.py script.
"""

### DONE
# - LEXICAL: True, MAX_NGRAM: 5, MIN_NGRAM: 2
# - LEXICAL: True, MAX_NGRAM: 0, MIN_NGRAM: 0

LEXICAL = True
MAX_NGRAM = 5
MIN_NGRAM = 2
OUT_FOLDER = 'output/embeddings'
if not os.path.exists(OUT_FOLDER):
    os.makedirs(OUT_FOLDER)

df_ratings = pd.read_csv('data/avgRatings_annotated.csv')
if LEXICAL and MAX_NGRAM == 0:
    model = ft.load_model('output/FTmodels/coca_clean_d300_w5_m{}_M{}_rho0.7869.bin'.format(MIN_NGRAM, MAX_NGRAM))
else:
    model = ft.load_model('output/FTmodels/coca_clean_d300_w5_m{}_M{}_rho0.7953.bin'.format(MIN_NGRAM, MAX_NGRAM))

ITEM = 'name'
df_ratings[ITEM] = df_ratings[ITEM].map(lambda x: x.lower())
df = df_ratings.drop_duplicates(subset=['name', 'type'])

if LEXICAL and MAX_NGRAM == 0:
    encoded = []
    avg_embedding = get_avg_emb(model, min=0, max=0)
    for _, row in tqdm(df.iterrows()):
        print(row['name'])
        emb = model[row['name']]
        if np.all(emb == 0):
            emb = avg_embedding
            print('name {} not found in vocab, avg embedding used.'.format(row['name']))

        encoded.append({
            'name': row['name'],
            'type': row['type'],
            'embedding': emb
        })
    with open(os.path.join(OUT_FOLDER, 'name_embeddings_ngram{}_lexical_{}.pkl'.format(MAX_NGRAM, LEXICAL)), 'wb') as f:
        pkl.dump(encoded, f)

else:
    for M in range(MIN_NGRAM, MAX_NGRAM + 1):
        encoded = []
        for _, row in df.iterrows():
            print(row['name'])
            encoded.append({
                'name': row['name'],
                'type': row['type'],
                'embedding': embed_string(row['name'], model, min=MIN_NGRAM, max=M, lexical=LEXICAL)
            })
        with open(os.path.join(OUT_FOLDER, 'name_embeddings_ngram{}_lexical_{}.pkl'.format(M, LEXICAL)), 'wb') as f:
            pkl.dump(encoded, f)
