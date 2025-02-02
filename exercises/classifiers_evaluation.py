import numpy as np

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi

pio.renderers.default = "browser"

LOSS_PLOT_X = "Iteration number"
LOSS_PLOT_Y = "Loss"
LOSS_PLOT_TITLE = "Loss as function of fitting iteration in {0} data"


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f'../datasets/{f}')

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        Perceptron(callback=lambda fit, a, b: losses.append(fit.loss(X, y))).fit(X, y)
        # Plot figure of loss as function of fitting iteration
        px.line(x=np.arange(len(losses)), y=losses, labels={'x': LOSS_PLOT_X, 'y': LOSS_PLOT_Y},
                title=LOSS_PLOT_TITLE.format(n)).show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f'../datasets/{f}')

        # Fit models and predict over training set
        lda_model = LDA().fit(X, y)
        gnb_model = GaussianNaiveBayes().fit(X, y)
        models = [gnb_model, lda_model]
        models_titles = ["Gaussian Naive Bayes", "LDA"]

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        # Add traces for data-points setting symbols and colors

        from IMLearn.metrics import accuracy
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=[rf"{models_titles[i]} Classifier, accuracy: {accuracy(y, m.predict(X))}" for
                                            i, m in enumerate(models)],
                            horizontal_spacing=0.01, vertical_spacing=.03)
        symbols = np.array(["circle", "diamond", "star"])

        for i, m in enumerate(models):
            fig.add_traces([
                go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                           marker=dict(color=m.predict(X), symbol=symbols[y],
                                       colorscale=[custom[0], custom[4], custom[-1]],
                                       line=dict(color="black", width=1)))], rows=(i // 2) + 1, cols=(i % 2) + 1)

            # Add `X` dots specifying fitted Gaussians' means
            fig.add_traces(go.Scatter(x=m.mu_[:, 0], y=m.mu_[:, 1], mode="markers", showlegend=False,
                                      marker=dict(color="black", symbol="x", size=10)), rows=(i // 2) + 1,
                           cols=(i % 2) + 1)

            # Add ellipses depicting the covariances of the fitted Gaussians
            for col in range(len(m.mu_)):
                cov = m.cov_ if type(m) == LDA else np.diag(m.vars_[col])
                fig.add_traces(get_ellipse(m.mu_[col], cov), rows=(i // 2) + 1, cols=(i % 2) + 1)

        fig.update_layout(title=rf"Classification of {f} Dataset", margin=dict(t=100)).update_xaxes(
            visible=False).update_yaxes(visible=False)
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
