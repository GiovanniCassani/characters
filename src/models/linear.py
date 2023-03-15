import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.linear_model import ElasticNet as enet
from sklearn.linear_model import LinearRegression as linregr
from sklearn.linear_model import Ridge as ridgeregr
from sklearn.linear_model import Lasso as lassoregr


def grid_search(df, x_col, y_col, group_col, alphas, ratios):

    """
    :param df:              Pandas df
    :param x_col:           str, matching the df column storing feature vectors
    :param y_col:           str, matching the df column storing values to be predicted
    :param group_col:       str, matching the df column storing the variable over which to match the folds
    :param alphas:          list of floats, possible values of the penalty (needs to be > 0)
    :param ratios:          list of floats, possible values for the mixing parameter (0 =< ratio =< 1)
    :return:

    Trains and evaluates Ridge regressors using k-fold CV, reporting mean squared error on each fold.
    """

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    X = np.array([vector for vector in df[x_col]], dtype=float)
    y = np.array(df[y_col], dtype=float)

    hyperparams = itertools.product(alphas, ratios)
    results = []

    for alpha, ratio in hyperparams:

        print('###alpha: {}; l1 ratio: {}'.format(alpha, ratio))

        for fold_id, (train_indices, test_indices) in enumerate(skf.split(df, df[[group_col]])):

            x_train, x_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            model = None
            if alpha > 0:
                if ratio == 0:
                    model = ridgeregr(alpha=alpha, max_iter=5000, solver='svd')
                elif ratio == 1:
                    model = lassoregr(alpha=alpha, max_iter=5000)
                else:
                    model = enet(alpha=alpha, l1_ratio=ratio, max_iter=5000)
            else:
                model = linregr()
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            mse = mean_squared_error(y_test, y_pred)
            results.append({
                'fold': fold_id,
                'alpha': alpha,
                'l1_ratio': ratio,
                'mse': mse
            })

    return pd.DataFrame(results)


def train(df, x_col, y_col, items_col, alpha, ratio):

    """
    :param df:              Pandas df
    :param x_col:           str, matching the header of the column containing the input feature vectors
    :param y_col:           str, matching the header of the column containing the values to be predicted
    :param items_col:       str, matching the header of the column containing the items labels
    :param alpha:           float, penalty (needs to be > 0)
    :param ratio:           float, the mixing parameter
    :return:                Pandas df containing predicted values for each item using LOO - for every item in the set,
                            an ElasticNet regressor is trained on all the other items and used to predict the target
                            variable for the held out data point
    """

    loo = LeaveOneOut()

    X = np.array([vector for vector in df[x_col]], dtype=float)
    y = np.array(df[y_col], dtype=float)
    names = df[items_col]

    # list for the prediction of each name's rating
    predictions = []

    # do the cross validation split (test is going to be only one name)
    for train_indices, test_index in loo.split(X):

        x_train, x_test = X[train_indices], X[test_index]
        y_train, y_test = y[train_indices], y[test_index]

        # list with names in the test set and their type, again for easier use later
        test_names = list(names.iloc[test_index])

        # fit the model on this fold's training set
        model = None
        if alpha > 0:
            if ratio == 0:
                model = ridgeregr(alpha=alpha, max_iter=5000, solver='svd')
            elif ratio == 1:
                model = lassoregr(alpha=alpha, max_iter=5000)
            else:
                model = enet(alpha=alpha, l1_ratio=ratio, max_iter=5000)
        else:
            model = linregr()
        model.fit(x_train, y_train)
        # predict the score for the test set (name)
        prediction = model.predict(np.array(x_test))

        # append the name, its type, its feature representation, its actual rating,
        # and its predicted rating and prediction probability into the predictions list
        for i in range(len(x_test)):
            predictions.append({
                'name': test_names[i],
                'feature_vector': x_test[i].tolist(),
                'true_rating': y_test[i],
                'predicted_rating': prediction[i]
            })

    return pd.DataFrame(predictions)
