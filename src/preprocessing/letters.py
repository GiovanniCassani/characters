import string

def count_vector(s):

    """
    :param s:   string
    :return:    a list containing counts of how often each English lowercase character occurs in the string
    """

    alphabet = list(string.ascii_lowercase)

    v = []
    for i in range(len(alphabet)):
        v.append(s.lower().count(alphabet[i]))
    return v

def loo_features(s):

    d = {}
    for i in range(len(s)):
        l = list(s.lower())
        key = l.pop(i)
        d[key] = count_vector(''.join(l))
    d['none'] = count_vector(s)

    return d


def unigrams(df, source_col, target_col):

    """
    :param df:              Pandas df
    :param source_col:      str, matching the header of the column containing the names to featurize
    :param target_col:      str, indicating the column header to use to store feature vectors
    :return:                the input df with an extra column containing featurized names
    """

    featurized = []
    for i in range(len(df[source_col])):
        featurized.append(count_vector(df.loc[i][source_col]))
    df[target_col] = featurized
    return df




