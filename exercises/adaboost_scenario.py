import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from IMLearn.metrics import accuracy
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pio.renderers.default = "browser"


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    ada_ensemble = AdaBoost(DecisionStump, n_learners)
    ada_ensemble.fit(train_X, train_y)

    fitted_learners = np.arange(1, n_learners)
    training_loss, test_loss = [], []
    for t in fitted_learners:
        training_loss.append(ada_ensemble.partial_loss(train_X, train_y, t))
        test_loss.append(ada_ensemble.partial_loss(test_X, test_y, t))
    go.Figure([
        go.Scatter(x=fitted_learners, y=training_loss, mode="lines + markers", name="Training data"),
        go.Scatter(x=fitted_learners, y=test_loss, mode="lines + markers", name="Test data")
    ], layout=go.Layout(
        title=f"Training and Test errors as a function of the number of fitted learners with noise: {noise}",
        xaxis_title="Number of fitted learners", yaxis_title="Prediction Error")).show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig = make_subplots(rows=2, cols=2, subplot_titles=[f"Ensemble of size: {num}" for num in T],
                        horizontal_spacing=0.01, vertical_spacing=.03)

    for i, num in enumerate(T):
        fig.add_traces(
            [decision_surface(lambda x: ada_ensemble.partial_predict(x, num), lims[0], lims[1], showscale=False),
             go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                        marker=dict(color=test_y, colorscale=[custom[0], custom[-1]],
                                    line=dict(color="black", width=1)))], rows=(i // 2) + 1, cols=(i % 2) + 1)

    fig.update_layout(
        title=f"Decision boundary obtained by using the ensemble up to iteration 5, 50, 100 and 250 with noise: "
              f"{noise}",
        margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()

    # Question 3: Decision surface of best performing ensemble
    test_error = []
    for num in fitted_learners:
        test_error.append(ada_ensemble.partial_loss(test_X, test_y, num))
    best_ensemble = np.argmin(np.array(test_error)) + 1

    fig = go.Figure(
        [decision_surface(lambda x: ada_ensemble.partial_predict(x, best_ensemble), lims[0], lims[1], showscale=False),
         go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                    marker=dict(color=test_y, colorscale=[custom[0], custom[-1]], line=dict(color="black", width=1)))])

    acc = accuracy(test_y, ada_ensemble.partial_predict(test_X, best_ensemble))
    fig.update_layout(
        title=f"Decision boundary obtained by using the ensemble of size {best_ensemble} and noise: {noise}. "
              f"accuracy: {acc}",
        margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()

    # Question 4: Decision surface with weighted samples
    fig = go.Figure(
        [decision_surface(lambda x: ada_ensemble.partial_predict(x, n_learners), lims[0], lims[1], showscale=False),
         go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                    marker=dict(color=train_y.astype(int), colorscale=[custom[0], custom[-1]],
                                line=dict(color="black", width=1),
                                size=(ada_ensemble.D_ / np.max(ada_ensemble.D_) * 5)))])
    fig.update_layout(
        title=f"Decision boundary obtained by using the weighted ensemble of size {n_learners} and noise: {noise}",
        margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    for noise_level in [0, 0.4]:
        fit_and_evaluate_adaboost(noise_level)
