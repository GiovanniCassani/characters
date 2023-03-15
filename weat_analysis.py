import os
import pandas as pd
import pickle as pkl
import fasttext as ft
import src.dsm.weat as weat


pole_words = {
    'polarity': {
        'right_pole': ['good', 'virtuous', 'kind', 'positive', 'right'],
        'left_pole': ['bad', 'evil', 'cruel', 'negative', 'wrong']
    },
    'age': {
        'right_pole': ['old', 'elderly', 'grandparent', 'grandfather', 'grandmother', 'adult'],
        'left_pole': ['young', 'youth', 'child', 'boy', 'girl', 'baby']
    },
    'gender': {
        'right_pole': ['female', 'feminine', 'woman', 'girl', 'women', 'she'],
        'left_pole': ['male', 'masculine', 'man', 'boy', 'men', 'he']
    }
}

### COMBINATIONs TO DO:

### COMBINATIONs DONE:
# - M: 5, LEXICAL: True
# - M: 5, LEXICAL: False
# - M: 2, LEXICAL: False
# - M: 0, LEXICAL: True

FEATURES = 'embedding'
LEXICAL = True
M = 0
TARGET = 'rating.mean'
GROUP = 'type'
ITEM = 'name'
ATTRIBUTES = ['gender', 'age', 'polarity']
OUT_DIR = 'output/predictions/weat'
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

if LEXICAL and M == 0:
    ft_model = ft.load_model('output/FTmodels/coca_clean_d300_w5_m0_M{}_rho0.7869.bin'.format(M))
    df_embeddings = pd.DataFrame(
        pkl.load(open('output/embeddings/name_embeddings_ngram{}_lexical_{}.pkl'.format(M, LEXICAL), 'rb'))
    )
else:
    ft_model = ft.load_model('output/FTmodels/coca_clean_d300_w5_m2_M5_rho0.7953.bin')
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

    associations = weat.similarities(
        model=ft_model,
        df=df_subset,
        wordset_a=pole_words[attribute]['left_pole'],
        wordset_b=pole_words[attribute]['right_pole'],
        item_col=ITEM,
        feature_col=FEATURES
    )

    pd.merge(
        associations,
        df_subset[[ITEM, GROUP, 'attribute', '{}_unit'.format(TARGET)]],
        on=ITEM
    ).to_csv(
        os.path.join(OUT_DIR, '{}_associations_ngram{}_lexical_{}.csv'.format(attribute, M, LEXICAL)),
        sep='\t', index=False, index_label=False
    )
