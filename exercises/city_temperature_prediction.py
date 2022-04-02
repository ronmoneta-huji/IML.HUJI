from typing import Tuple, Any

import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

FILE_PATH = r"../datasets/City_Temperature.csv"

pio.templates.default = "simple_white"


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    # remove extreme temp values
    data = data[data["Temp"] >= -20]  # based on searched the lowest temp recorded in those city ever
    data["DayOfYear"] = data['Date'].dt.dayofyear
    return data


def load_data(filename: str) -> Tuple[pd.DataFrame, pd.Series]:
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
    return data.drop(columns=["Temp"]), data["Temp"]


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    design_mat, response = load_data(FILE_PATH)

    # Question 2 - Exploring data for specific country
    raise NotImplementedError()

    # Question 3 - Exploring differences between countries
    raise NotImplementedError()

    # Question 4 - Fitting model for different values of `k`
    raise NotImplementedError()

    # Question 5 - Evaluating fitted model on different countries
    raise NotImplementedError()
