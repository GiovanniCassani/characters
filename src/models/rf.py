import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor as rf_regr


def grid_search(df, x_col, y_col, group_col, estimators, max_depth, max_features):

    """
    :param df:              Pandas df
    :param x_col:           str, matching the df column storing feature vectors
    :param y_col:           str, matching the df column storing values to be predicted
    :param group_col:       str, matching the df column storing the variable over which to match the folds
    :param estimators:      list of int, possible number of trees in the forest to evaluate
    :param max_depth:       list of int, possible number for the maximum depth of each tree
    :param max_features:    list of misc, possible fractions of features to use to build each tree
    :return:

    Trains and evaluates random forest regressors using k-fold CV, reporting mean squared error on each fold.
    """

    #### NOTE: For the 'extra' NN grid search, change the list of nodes and
    #### dropout values to search and print!! I am not making this an extra
    #### function because I already have enough and it's simple to understand.

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    X = np.array([vector for vector in df[x_col]], dtype=float)
    y = np.array(df[y_col], dtype=float)

    hyperparams = itertools.product(estimators, max_depth, max_features)
    results = []

    for estimator, depth, features in hyperparams:

        print('###estimator: {}; depth: {}; n features: {}'.format(estimator, depth, features))

        for fold_id, (train_indices, test_indices) in enumerate(skf.split(df, df[[group_col]])):

            x_train, x_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            model = None
            model = rf_regr(n_estimators=estimator, max_depth=depth, n_jobs=16, oob_score=True, max_features=features)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            mse = mean_squared_error(y_test, y_pred)
            results.append({
                'fold': fold_id,
                'estimator': estimator,
                'max_depth': depth,
                'max_features': features,
                'mse': mse
            })

    return pd.DataFrame(results)


def train(df, x_col, y_col, items_col, estimator, max_depth, max_features):

    """
    :param df:              Pandas df
    :param x_col:           str, matching the header of the column containing the input feature vectors
    :param y_col:           str, matching the header of the column containing the values to be predicted
    :param items_col:       str, matching the header of the column containing the items labels
    :param estimator:      list of int, possible number of trees in the forest to evaluate
    :param max_depth:       list of int, possible number for the maximum depth of each tree
    :param max_features:    list of misc, possible fractions of features to use to build each tree
    :return:                Pandas df containing predicted values for each item using LOO - for every item in the set,
                            a Randm Forest regressor is trained on all the other items and used to predict the target
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
        model = rf_regr(n_estimators=estimator, max_depth=max_depth, n_jobs=16, oob_score=True, max_features=max_features)
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
