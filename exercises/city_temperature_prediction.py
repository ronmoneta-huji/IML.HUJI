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

    # Question 3 - Exploring differences between countries

    # Question 4 - Fitting model for different values of `k`

    # Question 5 - Evaluating fitted model on different countries
