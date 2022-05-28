from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np

from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    train_losses = np.zeros(cv)
    validation_losses = np.zeros(cv)
    folds = np.array_split(X, cv)
    labels = np.array_split(y, cv)

    for k in range(cv):
        sample = np.concatenate(np.delete(folds, k, 0))
        label = np.concatenate(np.delete(labels, k, 0))
        model = estimator.fit(sample, label)
        train_losses[k] = scoring(label, model.predict(sample))
        validation_losses[k] = scoring(labels[k], model.predict(folds[k]))

    return train_losses.mean(), validation_losses.mean()
