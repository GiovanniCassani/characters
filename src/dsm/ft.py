import os
import warnings
import fasttext
import numpy as np
from src.dsm.men import corr


def grid(corpus, men, out_path, dimensionalities, window_sizes, min_values, max_values):

    """
    :param corpus:              str, path to a .txt file containing one sentence per line, with space separated tokens
    :param men:                 Pandas df consisting of three columns, w1, w2, and sim
    :param out_path:            str, indicating the folder where to store models
    :param dimensionalities:    list of ints, the possible values of vector dimensionality to explore
    :param window_sizes:        list of ints, the possible values of window sizes to explore (words to the left and
                                right of the target
    :param min_values:          list of ints, the smallest n-gram size the model will learn
    :param max_values:          list of ints, the largest n-gram size the model will learn
    :return:                    tuple(ft model, float): the trained FT model with the best correlation with human
                                relatedness scores with the correlation itself. The function also saves all trained
                                models to file.
    """

    best_model, best_rho = None, 0
    for dim in dimensionalities:
        for ws in window_sizes:
            for min_n in min_values:
                for max_n in max_values:
                    model = fasttext.train_unsupervised(input = corpus,
                                                        model = "skipgram",
                                                        dim = dim,
                                                        ws = ws,
                                                        minn = min_n,
                                                        maxn = max_n,
                                                        lr = 0.01,
                                                        epoch = 10,
                                                        minCount = 3,
                                                        thread = 64,
                                                        bucket=4000000)
                    rho = corr(model, men)
                    if rho > best_rho:
                        best_rho = rho
                        best_model = model

                    model.save_model(os.path.join(
                        out_path, '{}_d{}_w{}_m{}_M{}_rho{}.bin'.format(
                            os.path.splitext(corpus)[0].split('/')[1], dim, ws, min_n, max_n, np.round(rho, 4)))
                    )
                    del model

    return best_model, best_rho


def embed_string(s, model, min=0, max=0, lexical=True, loo=False):

    """
    :param s:       str, whose embedding we want to obtain
    :param model:   a fasttext object, storing a trained model
    :param min:     the smallest n-gram size we want to contribute to the final embedding. Default to 0 means all
                    ngrams are used.
    :param max:     the largest n-gram size we want to contribute to the final embedding. Default to 0 means no n-gram
                    is considered
    :param lexical: bool, indicates if the embedding of the full lexical form should be included (default to True means
                    the embedding of the full lexical form contributes to the output embedding)
    :param loo:     bool, indicates whether to create embeddings by excluding each n-gram in turn. Default to False
                    means all n-grams are considered in the output embedding. If set to True, the function returns
                    a dictionary mapping each n-gram that satisfies the conditions to the embedding obtained by not
                    considering that ngram. It makes most sense to set loo to True when setting max to a number higher
                    than 0 and lexical to False.
    :return:        the embedding of the input word in the input model derived only from n-grams of the desired size. If
                    loo = True, returns a dictionary mapping strings (the character ngrams) to embeddings (obtained
                    by excluding the embedding of the key ngram).

    This function allows to decide whether to embed the string only considering its lexical form, only (some of) its
    bigrams, or a combination of (some of) its bigrams and the lexical form.
    - min=0, max=0 and lexical=False return a 0-vector of the appropriate dimensionality
    - min=0, max=0 and lexical=True return only the vector of the lexical form
    - min=n, max=m and lexical=False return a vector resulting from the combination of vectors of ngrams of length
                between n and m (both included)
    - min=n, max=m and lexical=True return a vector resulting from the combination of the vector of the lexical form and
                vectors of ngrams of length between n and m (both included)
    """

    vec = np.zeros(model.get_dimension())   # initialize zero vector of the appropriate dimension
    n = 0                                   # initialize counter for averaging vector
    ngrams, hashes = model.get_subwords(s)  # get ngrams and corresponding hashes from the target string
    if lexical:
        if ngrams[0] == s:
            vec += model.get_input_vector(hashes[0])   # get the lexical embedding if the lexical flag is set to True
            n += 1
        else:
            pass                                        # if the lexical embedding doesn't exist, skip

    # loop through the ngrams and filter those of the appropriate length.
    if ngrams[0] == s:
        filtered = {(ngram, hash) for ngram, hash in zip(ngrams[1:], hashes[1:]) if min <= len(ngram) <= max}
    else:
        filtered = {(ngram, hash) for ngram, hash in zip(ngrams, hashes) if min <= len(ngram) <= max}
    n += len(filtered)      # increment the counter by the number of valid ngrams

    # fetch the vectors corresponding to each n-gram and sum them to the vector initialized before (with or without
    # lexical form, depending on the value of the argument lexical
    full_vec = np.copy(vec)
    if len(filtered):
        for ngram, hash in filtered:
            full_vec += model.get_input_vector(hash)
    else:
        warnings.warn('No ngram survived the min and max cutoffs ({}; {}).'.format(min, max))

    if loo:
        d = dict()
        # store the vector resulting from all ngrams as reference
        d['none'] = full_vec / n
        if len(filtered):
            # loop through all valid ngrams to make a vector when knocking out each of them
            for knockout_ngram, _ in filtered:
                # make a copy of the vector to update the same base vector for each ngram
                loo_vec = np.copy(vec)
                # update the vector by looping through all valid ngrams and skip the one being excluded
                for curr_ngram, curr_hash in filtered:
                    if curr_ngram != knockout_ngram:
                        loo_vec += model.get_input_vector(curr_hash)
                # normalize the vector by appropriate number of ngrams
                d[knockout_ngram] = loo_vec / (n - 1)
        else:
            raise ValueError(
                'No ngram of the appropriate size ({}; {}), cannot compute leave-one-out embeddings'.format(min, max)
            )
        return d

    else:
        # if no leave-one-out featurization is required, return the vector resulting from all ngrams
        return full_vec / n


def get_avg_emb(model, min=0, max=0, lexical=True):

    """
    :param model:   a fasttext object, storing a trained model
    :param min:     the smallest n-gram size we want to contribute to the final embedding. Default to 0 means all ngrams
                    are used.
    :param max:     the largest n-gram size we want to contribute to the final embedding. Default to 0 means only the
                    full lexical form is used
    :param lexical: bool, indicates if the embedding of the full lexical form should be included
    :return:        the average embedding given the entire model vocabulary and target ngrams
    """

    avg_vec = np.zeros(model.get_dimension())

    for word in model.words:
        avg_vec += embed_string(word, model, min=min, max=max, lexical=lexical)
    return avg_vec / len(model.words)


