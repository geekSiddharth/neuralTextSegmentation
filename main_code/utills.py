from collections import Counter

import numpy as np


def prepare_loaded_dataset_for_training(X, Y, ONE_SIDE_CONTEXT_SIZE):
    """
    :param X: X as by load_datset of process_documents
    :param Y: Y as by load_datset of process_documents
    :param ONE_SIDE_CONTEXT_SIZE: CONTEXT SIDE
    :return: X,Y for training in keras
    """
    X_left_context, X_main_context, X_right_context = [], [], []
    Y_ = []
    for i in range(0, len(X)):
        for j in range(ONE_SIDE_CONTEXT_SIZE + 1, len(X[i]) - ONE_SIDE_CONTEXT_SIZE - 1):
            X_left_context.append(np.array(X[i][j - ONE_SIDE_CONTEXT_SIZE:j + 1]))
            X_main_context.append(np.array([X[i][j]]))
            X_right_context.append(np.array(X[i][j:j + ONE_SIDE_CONTEXT_SIZE + 1]))

            if Y[i][j]:
                Y_.append(1)
            else:
                Y_.append(0)

    X = [np.array(X_left_context),
         np.array(X_main_context),
         np.array(X_right_context)]
    Y = Y_

    del X_left_context, X_main_context, X_right_context, Y_

    return X, Y


def get_class_weights(y, smooth_factor=0):
    """
    Returns the weights for each class based on the frequencies of the samples
    :param smooth_factor: factor that smooths extremely uneven weights
    :param y: list of true labels (the labels must be hashable)
    :return: dictionary with the weight for each class
    """
    counter = Counter(y)

    if smooth_factor > 0:
        p = max(counter.values()) * smooth_factor
        for k in counter.keys():
            counter[k] += p

    majority = max(counter.values())

    return {cls: float(majority / count) for cls, count in counter.items()}
