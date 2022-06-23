import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from sklearn.metrics import roc_curve, auc

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.metrics import misclassification_error
from IMLearn.model_selection import cross_validate
from IMLearn.utils import split_train_test

import plotly.graph_objects as go

from utils import custom


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """

    values = []
    callback_weights = []

    def inner(solver, weights, val, grad, t, eta, delta):
        values.append(val)
        callback_weights.append(weights)

    callback = inner
    return callback, values, callback_weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    min_vals = []
    for l_module in [L1, L2]:
        fig = go.Figure(layout=go.Layout(xaxis_title='GD Iterations',
                                         xaxis_range=[0, 1000],
                                         yaxis_title='Norm',
                                         yaxis_range=[0, 3],
                                         title=f"{l_module.__name__} Convergence rate"))
        for eta in etas:
            learning_rate = FixedLR(eta)
            module = l_module(init)
            callback, values, weights = get_gd_state_recorder_callback()
            gd = GradientDescent(learning_rate, callback=callback)
            gd.fit(module, None, None)
            plot_descent_path(l_module, np.array(weights), f"{l_module.__name__} Descent path with eta: {eta}").show(
                renderer="browser")
            fig.add_trace(go.Scatter(x=np.arange(len(values)), y=values, mode="lines", name=f"eta: {eta}"))
            min_vals.append(min(values))
            print(f"Lowest loss achieved when minimizing {l_module.__name__} with eta={eta}: {min(values)}")

        fig.show(renderer="browser")
        print(f"Lowest loss achieved when minimizing {l_module.__name__}: {min(min_vals)}\n")


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    # Plot algorithm's convergence for the different values of gamma
    # Plot descent path for gamma=0.95
    fig = go.Figure(layout=go.Layout(xaxis_title='GD Iterations',
                                     # xaxis_range=[0, 1000],
                                     yaxis_title='Norm',
                                     # yaxis_range=[0, 3],
                                     title=f"L1 Exponential Convergence rate"))
    min_vals = []
    for gamma in gammas:
        learning_rate = ExponentialLR(eta, gamma)
        module = L1(init)
        callback, values, weights = get_gd_state_recorder_callback()
        gd = GradientDescent(learning_rate, callback=callback)
        gd.fit(module, None, None)
        fig.add_trace(go.Scatter(x=np.arange(len(values)), y=values, mode="lines", name=f"gamma: {gamma}"))
        min_vals.append(min(values))
        print(f"Lowest loss achieved when minimizing L1 with gamma={gamma}: {min(values)}")
        if gamma == 0.95:
            plot_descent_path(L1, np.array(weights), f"L1 Descent path with gamma: {gamma}").show(
                renderer="browser")
    fig.show(renderer="browser")
    print(f"Lowest loss achieved when minimizing L1: {min(min_vals)}\n")


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting convergence rate of logistic regression over SA heart disease data
    logreg = LogisticRegression(solver=GradientDescent(FixedLR(1e-4), max_iter=20000))
    X_train, y_train, X_test, y_test = X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()
    logreg.fit(X_train, y_train)

    fpr, tpr, thresholds = roc_curve(y_train, logreg.predict_proba(X_train))
    c = [custom[0], custom[-1]]
    go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False, marker_size=5,
                         marker_color=c[1][1],
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$"))).show(renderer="browser")

    alpha_star = np.argmax(tpr - fpr)
    best_alpha = thresholds[alpha_star]
    print(f"The alpha that achieves the optimal ROC value is: {best_alpha} \n")

    logreg.alpha_ = best_alpha
    optimal_loss = logreg.loss(X_test, y_test)
    print(f"Model's loss for Optimal alpha is: {optimal_loss}\n")

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    lambdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    for pen in ["l1", "l2"]:
        losses = []
        for lam in lambdas:
            logreg = LogisticRegression(solver=GradientDescent(FixedLR(1e-4), max_iter=20000), penalty=pen, lam=lam)
            train_loss, validation_loss = cross_validate(logreg, X_train, y_train, misclassification_error)
            losses.append(validation_loss)

        lambda_star = np.argmin(losses)
        best_lambda = lambdas[lambda_star]
        print(f"Best lambda for {pen} is: {best_lambda}\n")

        best_logreg = LogisticRegression(solver=GradientDescent(FixedLR(1e-4), max_iter=20000), penalty=pen,
                                         lam=best_lambda)
        best_logreg.fit(X_train, y_train)
        optimal_loss = best_logreg.loss(X_test, y_test)
        print(f"Model's loss for {pen} with best lambda is: {optimal_loss}\n")


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
