from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pio.renderers.default = "browser"


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    X = np.linspace(-1.2, 2, n_samples)
    eps = np.random.normal(0, noise, n_samples)
    y_noiseless = (X + 3) * (X + 2) * (X + 1) * (X - 1) * (X - 2)
    y = y_noiseless + eps
    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(X), pd.Series(y), 2 / 3)
    train_X, test_X = train_X.iloc[:, 0].to_numpy(), test_X.iloc[:, 0].to_numpy()
    train_y, test_y = train_y.to_numpy(), test_y.to_numpy()
    go.Figure([
        go.Scatter(x=X, y=y_noiseless, mode="lines", name="True (noiseless)"),
        go.Scatter(x=train_X, y=train_y, mode="markers", name="Train", marker=dict(color='orange')),
        go.Scatter(x=test_X, y=test_y, mode="markers", name="Test", marker=dict(color='green'))
    ], layout=go.Layout(title_text="True model and the train/test sets of f(x)",
                        xaxis={"title": "x"},
                        yaxis={"title": "f(x)"})).show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_losses, validation_losses = np.zeros(11), np.zeros(11)
    for k in range(11):
        poly_estimator = PolynomialFitting(k)
        train_losses[k], validation_losses[k] = cross_validate(poly_estimator, train_X, train_y, mean_square_error)
    go.Figure([
        go.Scatter(x=[k for k in range(11)], y=train_losses, mode="markers + lines", name="Train Errors"),
        go.Scatter(x=[k for k in range(11)], y=validation_losses, mode="markers + lines", name="Validation Errors"),
    ], layout=go.Layout(title_text="Average training and validation errors as a function of k",
                        xaxis={"title": "k"},
                        yaxis={"title": "Average Error"})).show()
    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_k = np.argmin(validation_losses)
    poly_estimator = PolynomialFitting(best_k)
    poly_estimator.fit(train_X, train_y)
    print(f"k^*: {best_k}, test error: {np.round(poly_estimator.loss(test_X, test_y), 2)}")


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    raise NotImplementedError()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    raise NotImplementedError()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
