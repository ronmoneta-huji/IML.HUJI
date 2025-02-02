from typing import Tuple, Any

import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

FILE_PATH = r"../datasets/City_Temperature.csv"
ISRAEL_SCATTER_PLOT_TITLE = "Israeli average daily temperature change as a function of the day of the year"
ISRAEL_BAR_PLOT_TITLE = "Israeli standard deviation of the daily temperatures as a function of the month"
ISRAEL_FIT_BAR_PLOT_TITLE = "Test Error as a function of Polynomial Degree"
COUNTRIES_BAR_PLOT_TITLE = "Israeli fitted model’s error over each of the other countries"

pio.templates.default = "simple_white"
pio.renderers.default = "browser"


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    # remove extreme temp values
    data = data[data["Temp"] >= -20]  # based on searched the lowest temp recorded in those city ever
    data["DayOfYear"] = data.loc[:, 'Date'].dt.dayofyear
    return data


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    data = pd.read_csv(filename, parse_dates=['Date']).dropna().drop_duplicates()
    data = preprocess_data(data)
    return data


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data_mat = load_data(FILE_PATH)

    # Question 2 - Exploring data for specific country
    data_mat["Year"] = data_mat.loc[:, "Year"].astype(str)
    israel_subset = data_mat.loc[(data_mat['Country'] == 'Israel')]
    px.scatter(x=israel_subset["DayOfYear"], y=israel_subset["Temp"], color=israel_subset["Year"],
               labels={'x': "DayOfYear", 'y': "Temp"}, title=ISRAEL_SCATTER_PLOT_TITLE).show()

    israel_grouped_subset = israel_subset.groupby("Month").std()
    px.bar(x=israel_grouped_subset.index, y=israel_grouped_subset["Temp"],
           labels={'x': "Month", 'y': "Standard Deviation of Temp"}, title=ISRAEL_BAR_PLOT_TITLE).show()

    # # Question 3 - Exploring differences between countries
    countries_grouped_subset = data_mat.groupby(['Country', 'Month']).agg({'Temp': ['mean', 'std']})
    px.line(x=countries_grouped_subset.index.get_level_values(1), y=countries_grouped_subset["Temp"]["mean"],
            color=countries_grouped_subset.index.get_level_values(0), error_y=countries_grouped_subset["Temp"]["std"],
            labels={'x': "Month", 'y': "Average Temperature"}, title=ISRAEL_BAR_PLOT_TITLE).show()

    # Question 4 - Fitting model for different values of `k`
    design_mat, response = israel_subset.drop(columns=["Temp"]), israel_subset["Temp"]
    israel_train_X, israel_train_y, israel_test_X, israel_test_y = split_train_test(design_mat, response)
    israel_DoY_train_X, israel_DoY_test_X = israel_train_X["DayOfYear"], israel_test_X["DayOfYear"]

    loss = np.empty(10)
    for k in range(1, 11):
        poly_model = PolynomialFitting(k)
        poly_model.fit(israel_DoY_train_X.to_numpy(), israel_train_y.to_numpy())
        loss[k - 1] = np.around(poly_model.loss(israel_DoY_test_X.to_numpy(), israel_test_y.to_numpy()), 2)

    for key, val in enumerate(loss):
        print(f"K:{key + 1} Test error:{val}\n")

    px.bar(x=np.arange(1, 11), y=loss,
           labels={'x': "Polynomial Degree", 'y': "Test Error over the Test set"},
           title=ISRAEL_FIT_BAR_PLOT_TITLE).show()

    # Question 5 - Evaluating fitted model on different countries
    DoY_design_mat = design_mat["DayOfYear"]
    poly_model = PolynomialFitting(5)
    poly_model.fit(DoY_design_mat, response)
    country_list = data_mat["Country"].unique()
    country_list = country_list[country_list != "Israel"]
    loss = np.empty(len(country_list))

    for i, c in enumerate(country_list):
        country_full = data_mat.loc[(data_mat['Country'] == c)]
        country_test_x = country_full["DayOfYear"]
        country_test_y = country_full["Temp"]
        loss[i] = poly_model.loss(country_test_x.to_numpy(), country_test_y.to_numpy())

    px.bar(x=country_list, y=loss,
           labels={'x': "Country", 'y': "Model Error over different countries"},
           title=COUNTRIES_BAR_PLOT_TITLE).show()
