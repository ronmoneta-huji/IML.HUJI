from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

SAMPLES_AMOUNT = 1000
PLOT_HEIGHT = 900
PLOT_WIDTH = 900
####univariate gaussian####
UNI_MU = 10
UNI_VAR = 1
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
Q2_INCREMENT = 10
####multivariate gaussian####
MULTI_MU = np.array([0, 0, 4, 0])
MULTI_COV = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
LIKELIHOOD_PLOT_TITLE = "Log likelihood heatmap for models with expectation [f1,0,f3,0] and given true covariance matrix"
LIKELIHOOD_PLOT_XTITLE = "f3 values"
LIKELIHOOD_PLOT_YTITLE = "f1 values"
Q5_LOWER_BOUND = -10
Q5_UPPER_BOUND = 10
Q5_SPACES = 200

pio.renderers.default = "browser"
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    uvg = UnivariateGaussian()
    mu, var, m = UNI_MU, UNI_VAR, SAMPLES_AMOUNT
    X = np.random.normal(mu, var, m)
    uvg.fit(X)

    print(f"(expectation, variance): {uvg.mu_},{uvg.var_}")


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
    mvg = MultivariateGaussian()
    mu, cov, m = MULTI_MU, MULTI_COV, SAMPLES_AMOUNT
    X = np.random.multivariate_normal(mu, cov, m)
    mvg.fit(X)

    print(f"\nestimated expectation: \n{mvg.mu_}\n\n estimated covariance matrix: \n{mvg.cov_}")


    # Question 5 - Likelihood evaluation
    f1 = f3 = np.linspace(Q5_LOWER_BOUND, Q5_UPPER_BOUND, Q5_SPACES)
    q5_mu = np.array(np.meshgrid(f1, [0], f3, [0])).T.reshape(Q5_SPACES, Q5_SPACES, 4)
    single_mu_likelihood = lambda cur_mu: mvg.log_likelihood(cur_mu, cov, X)
    likelihood = np.array([list(map(single_mu_likelihood, q5_mu[row])) for row in range(q5_mu.shape[0])]).T

    go.Figure(go.Heatmap(x=f3, y=f1, z=likelihood),
              layout=go.Layout(title=LIKELIHOOD_PLOT_TITLE, xaxis_title=LIKELIHOOD_PLOT_XTITLE,
                               yaxis_title=LIKELIHOOD_PLOT_YTITLE, height=PLOT_HEIGHT, width=PLOT_WIDTH)).show()


    # Question 6 - Maximum likelihood
    f1_index, f3_index = np.where(likelihood == np.amax(likelihood))
    # would also probably work with np.argmax(likelihood, keepdims=True), but not supported in our version
    print(f"\nmodel that achieved the maximum log-likelihood value(f1,f3): \n{np.around([f1[f1_index], f3[f3_index]], 3)}")


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
