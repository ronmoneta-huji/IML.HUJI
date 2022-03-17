from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

UNI_MU = 10
UNI_VAR = 1
SAMPLES_AMOUNT = 1000
ABS_DIST_PLOT_MODE = 'markers+lines'
ABS_DIST_PLOT_NAME = r'|$\widehat\mu$ - \mu$|'
ABS_DIST_PLOT_TITLE = r"$\text{absolute distance between the estimated and true value of the expectation, " \
                      r"as a function of the sample size}$"
ABS_DIST_PLOT_XTITLE = "$m\\text{ - number of samples}$"
ABS_DIST_PLOT_YTITLE = "absolute distance"
PDF_PLOT_MODE = 'markers'
PDF_PLOT_NAME = r'PDF(X)'
PDF_PLOT_TITLE = r"$\text{Empirical PDFs of fitted model}$"
PDF_PLOT_XTITLE = "ordered sample values"
PDF_PLOT_YTITLE = "PDF of sample values"
PLOT_HEIGHT = 300
Q2_INCREMENT = 10
pio.renderers.default = "browser"
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    uvg = UnivariateGaussian()
    mu, var, m = UNI_MU, UNI_VAR, SAMPLES_AMOUNT
    X = np.random.normal(mu, var, m)
    uvg.fit(X)

    print(uvg.mu_, uvg.var_)

    # Question 2 - Empirically showing sample mean is consistent
    absolute_distance = []
    for i in range(Q2_INCREMENT, SAMPLES_AMOUNT, Q2_INCREMENT):
        uvg.fit(X[:i])
        absolute_distance.append(np.abs(uvg.mu_ - mu))
    go.Figure(
        [go.Scatter(x=np.linspace(Q2_INCREMENT, SAMPLES_AMOUNT, num=SAMPLES_AMOUNT // Q2_INCREMENT, endpoint=True),
                    y=absolute_distance, mode=ABS_DIST_PLOT_MODE, name=ABS_DIST_PLOT_NAME)],
        layout=go.Layout(title=ABS_DIST_PLOT_TITLE, xaxis_title=ABS_DIST_PLOT_XTITLE, yaxis_title=ABS_DIST_PLOT_YTITLE,
                         height=PLOT_HEIGHT)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    go.Figure(
        [go.Scatter(x=X, y=uvg.pdf(X), mode=PDF_PLOT_MODE, name=PDF_PLOT_NAME)],
        layout=go.Layout(title=PDF_PLOT_TITLE, xaxis_title=PDF_PLOT_XTITLE, yaxis_title=PDF_PLOT_YTITLE,
                         height=PLOT_HEIGHT)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    # test_multivariate_gaussian()
