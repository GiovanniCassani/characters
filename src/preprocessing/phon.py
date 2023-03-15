import eng_to_ipa
import numpy as np
import pandas as pd


def make_phon_feat_dict():

    """
    :return:    dict, mapping abstract phonological features to symbols belonging to each category
    """

    return {
        "Obstruents": ["t","d","s","z","ɬ","ɮ","θ","ð","c","ɟ","ç","ʝ"],
        "Fricatives": ["ɸ","β","f","v","θ","ð","s","z","ʃ","ʒ","ʂ","ʐ","ç","ʝ","x","ɣ","χ","ʁ","ħ","ʕ","h","ɦ"],
        "Lateral Fricatives": ["ɬ","ɮ"],
        "Plosives": ["p","b","t","d","ʈ","ɖ","c","ɟ","k","g","q","ɢ","ʔ"],
        "Nasals": ["m","ɱ","n","ɳ","ɲ","ŋ","ɴ"],
        "Approximants": ["ʋ","ɹ","ɻ","j","ɰ"],
        "Lateral Approximants": ["l","ɭ","ʎ","ʟ"],
        "Alveolars": ["t","d","n","r","ɾ","s","z","ɬ","ɮ","ɹ","l"],
        "Glottals": ["ʔ","h","ɦ"],
        "Dentals": ["θ","ð"],
        "Bilabials": ["p","b","m","ʙ","ɸ","β"],
        "Sonorants": ["m","ɱ","n","ɳ","ɲ","ŋ","ɴ","l","ʎ","r","ɹ","ʀ"],
        "Velars": ["k","ɡ","ŋ","x","ɣ","ɰ","ʟ"],
        "Voiced": ["b","m","ʙ","β","ɱ","ⱱ","v","ʋ","ð","d","n","r","ɾ","z","ɮ","ɹ","l","ʒ","ɖ","ɳ","ɽ","ʐ","ɻ",
                   "ɭ","ɟ","ɲ","ʝ","j","ʎ","ɡ","ŋ","ɣ","ɰ","ʟ","ɢ","ɴ","ʀ","ʁ","ʕ","ɦ"],
        "Voiceless": ["p","ɸ","f","θ","t","s","ɬ","ʃ","ʈ","ʂ","c","ç","k","x","q","χ","ħ","ʔ","h"],
        "High": ["i","y","ɪ","ʏ","ɨ","ʉ","ʊ","ɯ","u"],
        "Mid": ["e","ø","ɘ","ɵ","ɤ","o","ə","ɛ","œ","ɜ","ɞ","ʌ","ɔ"],
        "Low": ["æ","ɐ","a","ɶ","ɑ","ɒ"],
        "Front": ["i","y","ɪ","ʏ","e","ø","ɛ","œ","æ","a","ɶ"],
        "Central": ["ɨ","ʉ","ɘ","ɵ","ə","ɜ","ɞ","ɐ"],
        "Back": ["ɯ","u","ʊ","ɤ","o","ʌ","ɔ","ɑ","ɒ"],
        "Rounded": ["y","ʏ","ø","œ","ɶ","ʉ","ɵ","ɞ","u","o","ɔ","ɒ"],
        "Unrounded": ["i","e","ɪ","ɛ","a","ɨ","ɘ","ɜ","ɯ","ɤ","ʌ","ɑ"]
    }


def lett2phon(df, source_col, target_col, verbose=False):

    """
    :param df:              Pandas df
    :param source_col:      str, matching the header of the column which contains the names to transform in IPA
    :param target_col:      str, indicating the header of column which will store the transcribed names
    :param verbose:         bool, if True prints info on how many names could not be trascribed automatically
    :return:                the input df with an extra column containing target names in IPA
    """

    tmp = []
    for name in df[source_col].values:
        phon = eng_to_ipa.convert(name)
        if phon[-1] == '*':
            tmp.append("NA")
        else:
            tmp.append(phon)

    if verbose:
        print("NA value count: ", len([tmp[i] for i in range(len(tmp)) if tmp[i] == "NA"]))

    df[target_col] = tmp
    return df


def extra_names(path, df, source_col, target_col):

    """
    :param path:        str, path to a file with transcriptions of names which cannot be transcribed automatically
    :param df:          Pandas df
    :param source_col:  str, matching the header of the column which contains the names to transform in IPA
    :param target_col:  str, indicating the header of column which will store the transcribed names
    :return:            the input df, filling in missing transcriptions
    """

    na_names_tr = pd.read_csv(path)  # "data/na_names_phonTrans.csv"

    for i in range(len(df[source_col])):
        for j in range(len(na_names_tr[source_col])):
            if df[source_col].loc[i] == na_names_tr[source_col][j]:
                df[target_col].loc[i] = na_names_tr[target_col][j]

    return df


def phon_count(ipa_transcript):

    """
    :param ipa_transcript:  str, in IPA symbols
    :return:                a vector containing the counts of how often each phonological features occurs in the input
    """

    feature_dict = make_phon_feat_dict()

    word_pv = np.zeros((1, len(feature_dict.keys())), dtype=int)
    word_pvdf = pd.DataFrame(word_pv, columns=list(feature_dict.keys()))
    for i in range(len(ipa_transcript)):
        for key in list(feature_dict.keys()):
            if ipa_transcript[i] in feature_dict[key]:
                word_pvdf[key][0] += 1
    return word_pvdf.iloc[0]


def loo_features(ipa_transcript):

    d = dict()
    feature_vector = phon_count(ipa_transcript)
    d['none'] = np.array(feature_vector)
    for feature in feature_vector[feature_vector.values > 0].index:
        feature_vector_copy = feature_vector.copy(deep=True)
        feature_vector_copy[feature_vector_copy.index == feature] += - 1
        d[feature] = np.array(feature_vector_copy)

    return d

def phon2features(df, source_col, target_col):

    """
    :param df:          Pandas df
    :param source_col:  str, matching the header of the column which contains the IPA strings
    :param target_col:  str, indicating the header of column which will store the featurized names
    :return:            the input df, with an extra column storing feature vectors derived from IPA symbols
    """

    featurized = []
    for i in range(len(df[source_col])):
        featurized.append(np.array(phon_count(df.loc[i][source_col]), dtype=int))
    df[target_col] = featurized
    return df
