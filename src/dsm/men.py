from scipy.spatial.distance import cosine


def corr(model, men):

    """
    :param model:
    :param men:
    :return:
    """

    men['cos_sim'] = men.apply(lambda row: 1 - cosine(model[row['w1']], model[row['w2']]), axis=1)
    return men['cos_sim'].corr(men['sim'], method='spearman')

