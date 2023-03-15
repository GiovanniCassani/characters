import pandas as pd
from scipy.spatial.distance import cosine


def similarities(model, df, wordset_a, wordset_b, item_col, feature_col):

    """
    :param model:       FastText object, a trained FastText model
    :param df:          Pandas df, minimally containing two columns, one with the target strings and one with their
                        feature representations
    :param wordset_a:   iterable of strings, constituting the examples for one pole of a semantic differential
    :param wordset_b:   iterable of strings, constituting the examples for the other pole of a semantic differential
    :param item_col:    str, indicating the column header of the target strings
    :param feature_col: str, indicating the column header of the feature representations
    :return:            a Pandas df storing the cosine similarity between the feature representations corresponding to
                        each string in the item_col and each word in the two input word-sets as represented in the input
                        FastText model
    """

    associations = []
    for name in df[item_col]:
        for w_a, w_b in zip(wordset_a, wordset_b):
            associations.append({
                    'name': name,
                    'feature_vector': df.loc[df[item_col] == name, feature_col].to_numpy()[0],
                    'associate': w_a,
                    'wordset': 'a',
                    'cosine_sim': 1 - cosine(df.loc[df[item_col] == name, feature_col].to_numpy()[0], model[w_a])
                })
            associations.append({
                    'name': name,
                    'feature_vector': df.loc[df[item_col] == name, feature_col].to_numpy()[0],
                    'associate': w_b,
                    'wordset': 'b',
                    'cosine_sim': 1 - cosine(df.loc[df[item_col] == name, feature_col].to_numpy()[0], model[w_b])
                })

    return pd.DataFrame(associations)
