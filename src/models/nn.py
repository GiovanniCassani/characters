import os
import random
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras.layers import Dropout
from tensorflow.keras.initializers import RandomNormal

os.environ["OMP_NUM_THREADS"] = "32"

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)


def make(input_dim, units, dropout, act, n_layers, lr):

    """
    :param input_dim:   int, the dimensionality of the input layer
    :param units:       int, how many nodes to have in the dense hidden layer
    :param dropout:     float, the proportion of drop out (number between 0 and 1)
    :param act:         str, the activation function for the hidden layer
    :param n_layers:    int, how many hidden layers - assumes the same number of units, dropout and activation for all
                        hidden layers (doesn't make much sense to add more than 2)
    :param lr:          float, learning rate
    :return:            Keras object,

    Creates a neural network with custom hyper-parameters using only dense layers.
    """

    random.seed(42)
    tf.random.set_seed(42)

    fnn_model = Sequential()
    # always add one hidden layer (dimensionality equal to the input dimensionality)
    fnn_model.add(Dense(
        units,
        input_dim=input_dim,
        kernel_initializer=RandomNormal(mean=0, stddev=0.05),
        activation=act
    ))
    fnn_model.add(Dropout(dropout))

    # add other layers if necessary, with the same activation and number of units as the first hidden layer
    for _ in range(n_layers - 1):
        fnn_model.add(Dense(
            units,
            input_dim=units,
            kernel_initializer=RandomNormal(mean=0, stddev=0.05),
            activation=act
        ))
        fnn_model.add(Dropout(dropout))
    fnn_model.add(Dense(1, activation='linear'))
    fnn_model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')

    return fnn_model


def grid_search(df, x_col, y_col, group_col, units, dropout, activations, n_layers, learning_rates):

    """
    :param df:              Pandas df
    :param x_col:           str, matching the df column storing feature vectors
    :param y_col:           str, matching the df column storing values to be predicted
    :param group_col:       str, matching the df column storing the variable over which to match the folds
    :param units:           list of ints, containing the different values for number of hidden units to search over
    :param dropout:         list of floats, containing the different values for dropout to search over
    :param activations:     list of str, containing the different activation function to search over
    :param n_layers:        list of ints, containing the number of hidden layers to try
    :param learning_rates:  list of floats, indicating the different learning rates to try
    :return:                Pandas df, providing the loss on each fold.

    Trains and evaluates several neural networks using k-fold CV, reporting performance (MSE) on each fold together
    with the hyper-parameters of the model which yielded that performance
    """

    #### NOTE: For the 'extra' NN grid search, change the list of nodes and
    #### dropout values to search and print!! I am not making this an extra
    #### function because I already have enough and it's simple to understand.

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    X = np.array([vector for vector in df[x_col]], dtype=float)
    y = np.array(df[y_col], dtype=float)

    hyperparams = itertools.product(units, dropout, activations, n_layers, learning_rates)
    results = []

    for unit_val, dropout_val, act, layers, lr in hyperparams:

        print(
            '###units: {}; dropout: {}; activation: {}; n_layers: {}; lr: {}'.format(
                unit_val, dropout_val, act, layers, lr
            )
        )

        for fold_id, (train_indices, test_indices) in enumerate(skf.split(df, df[[group_col]])):

            x_train, x_test = tf.convert_to_tensor(X[train_indices]), tf.convert_to_tensor(X[test_indices])
            y_train, y_test = tf.convert_to_tensor(y[train_indices]), tf.convert_to_tensor(y[test_indices])
            input_dim = x_train.shape[1]

            model = None
            model = make(input_dim, unit_val, dropout_val, act, layers, lr)
            callback = EarlyStopping(monitor='val_loss', patience=7, verbose=0)
            model.fit(x_train, y_train, validation_data=(x_test, y_test),
                      epochs=100, batch_size=8, callbacks=[callback], verbose=0)
            y_pred = model.predict(x_test)
            mse = mean_squared_error(y_test, y_pred)
            results.append({
                'fold': fold_id,
                'nodes': unit_val,
                'dropout': dropout_val,
                'act': act,
                'n_layers': layers,
                'lr': lr,
                'mse': mse
            })

    return pd.DataFrame(results)


def predict(df, x_col, y_col, items_col, group_col, units, dropout_rate, act, n_layers, lr):

    """
    :param df:              Pandas df
    :param x_col:           str, matching the header of the column containing the input feature vectors
    :param y_col:           str, matching the header of the column containing the values to be predicted
    :param items_col:       str, matching the header of the column containing the items labels
    :param group_col:       str, matching the header of df column storing the variable over which to match the folds
                            when deriving validation data for early stopping
    :param units:           int, indicating how many units per hidden layer
    :param dropout_rate:    float [0;1[ indicating the dropout rate to apply after each layer
    :param act:             str, indicating the activation function per layer
    :param n_layers:        int, indicating how many Dense layers to add to the network
    :param lr:              float, the learning rate
    :return:                Pandas df containing predicted values for each item using LOO - for every item in the set,
                            a network is trained on all the other items and used to predict the target variable for the
                            held out data point

    Performs LOOCV on the input DataFrame, yielding an output DataFrame with the predicted value for each input item
    obtained from a network trained on the other items (using 10% as validation data)
    """

    loo = LeaveOneOut()
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=111)

    X = np.array([vector for vector in df[x_col]], dtype=float)
    y = np.array(df[y_col], dtype=float)
    type = df[group_col]
    test_names = df[items_col]

    # list for the prediction of each name's rating
    predictions = []

    # do the cross validation split (test is going to be only one name)
    for dev_indices, test_index in loo.split(X):

        type_dev = type[dev_indices]
        # further split the training into proper training and validation to use early stopping and prevent overfitting
        train_indices, val_indices = list(skf.split(dev_indices, type_dev))[0]

        x_train, x_val, x_test = tf.convert_to_tensor(X[train_indices]), tf.convert_to_tensor(X[val_indices]), \
                                 tf.convert_to_tensor(X[test_index])
        y_train, y_val, y_test = tf.convert_to_tensor(y[train_indices]), tf.convert_to_tensor(y[val_indices]), \
                                 tf.convert_to_tensor(y[test_index])

        input_dim = x_train.shape[1]
        test_name = test_names[test_index]

        # fit the model on this fold's training set
        model = None
        model = make(input_dim, units, dropout_rate, act, n_layers, lr)
        callback = EarlyStopping(monitor='val_loss', patience=7, verbose=0)
        model.fit(x_train, y_train, validation_data=(x_val, y_val),
                  epochs=150, batch_size=8, callbacks=[callback], verbose=0)
        # predict the score for the test set (name)
        prediction = model.predict(x_test)

        # append the name, its type, its feature representation, its actual rating,
        # and its predicted rating and prediction probability into the predictions list
        for i in range(len(x_test)):
            predictions.append({
                'name': test_name.values[0],
                'feature_vector': x_test[i].numpy().tolist(),
                'true_rating': y_test[i].numpy(),
                'predicted_rating': prediction[i][0]
            })

    return pd.DataFrame(predictions)
