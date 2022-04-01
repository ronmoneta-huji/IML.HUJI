from datetime import datetime
from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression
from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"

FILE_PATH = r"../datasets/house_prices.csv"
DATETIME_FORMAT = '%Y%m%dT%H%M%S'


def remove_impossible_amount(data):
    cols = data.columns.tolist()
    cols.remove("long")
    cols.remove("date")
    ge_cols = ['bedrooms', 'waterfront', 'view', 'sqft_basement', 'yr_renovated']
    for col in cols:
        if col in ge_cols:
            data = data[data[col] >= 0]
        else:
            data = data[data[col] > 0]

    return data


def remove_impossible_relation(data):
    data = data[(data['yr_renovated'] > 0) & (data['yr_built'] < data['yr_renovated']) |
                ((data['yr_renovated'] == 0) & (data['yr_built'] > data['yr_renovated']))]
    data = data[data['sqft_lot'] >= data['sqft_living']]
    return data


def handle_categorical_vars(data):
    # handle zip - to categorical
    one_hot_zip = pd.get_dummies(data["zipcode"])
    data = pd.concat([data, one_hot_zip], axis=1)
    return data


def derived_features(data):
    # handle date -
    data['date'] = pd.to_datetime(data['date'], format=DATETIME_FORMAT)
    data['year'] = pd.DatetimeIndex(data['date']).year
    data['month'] = pd.DatetimeIndex(data['date']).month
    data['day'] = pd.DatetimeIndex(data['date']).day

    # True building age
    data['building_true_age'] = datetime.now().year - data["yr_built"]
    # renovation flag
    data['renovation_flag'] = np.where(data['yr_renovated'] > 0, 1, 0)

    return data


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    data = remove_impossible_amount(data)
    data = remove_impossible_relation(data)
    data = handle_categorical_vars(data)
    data = derived_features(data)

    return data


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    data = pd.read_csv(filename).dropna().drop_duplicates()
    data = preprocess_data(data)

    return data.drop(columns=["price"]), data["price"]


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    design_mat, response = load_data(FILE_PATH)

    # Question 2 - Feature evaluation with respect to response
    raise NotImplementedError()

    # Question 3 - Split samples into training- and testing sets.
    raise NotImplementedError()

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    raise NotImplementedError()
