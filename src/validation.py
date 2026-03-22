import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import cross_validate

def mdape(y_pred, y_test):
    """
    Calculate the Median Absolute Percentage Error (MdAPE) between predicted values and true values.

    MdAPE is a statistical measure that evaluates the accuracy of a predictive model by calculating
    the median of the absolute percentage errors. It is widely used for error analysis in forecasting
    and regression problems.

    :param y_pred: Array or series containing the predicted values.
    :param y_test: Array or series containing the true values corresponding to predictions.
    :return: The MdAPE value as a float, representing the median of the absolute percentage errors across
        the given dataset.
    """
    error = abs(y_test - y_pred)
    return np.median((error * 100) / y_test)

def real_world_mdape(y_pred_log, y_test_log):
    """
    Calculate the Median Absolute Percentage Error (MdAPE) for log-transformed predicted
    and actual values.

    This function computes the MdAPE by first transforming the log-transformed predicted
    and actual values back to their original scale using the exponential function. It then
    calculates the absolute percentage error for each data point and computes the median
    of these errors.

    :param y_pred_log: Array-like, log-transformed predicted values.
    :param y_test_log: Array-like, log-transformed actual (true) values.
    :return: Median Absolute Percentage Error (MdAPE) expressed as a percentage.
    :rtype: float
    """
    y_pred_actual = np.exp(y_pred_log)
    y_test_actual = np.exp(y_test_log)

    error = np.abs(y_test_actual - y_pred_actual)
    return np.median(error / y_test_actual) * 100

def r2(y_pred, y_test):
    sse = sum((y_test - y_pred) ** 2)
    sst = sum((y_test - np.mean(y_test)) ** 2)
    return 1 - (sse / sst)

def get_eval_plots(y_pred, y_test) -> plt.Figure:
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))

    axs[0].scatter(x=y_pred, y=y_test)
    axs[0].set_title("QQ Plot (y_pred vs y_test)")
    axs[0].set_ylabel("y_pred")
    axs[0].set_xlabel("y_test")

    axs[1].scatter(x=y_pred, y=y_test - y_pred)
    axs[1].set_title("Residual Plot (y_pred vs residual)")
    axs[1].set_ylabel("y_pred")
    axs[1].set_xlabel("residual")

    return fig

def evaluate(y_pred, y_test):
    print(f'R2: {r2(y_pred, y_test):.4f}')
    print(f'MdAPE (log-scale): {mdape(y_pred, y_test):.2f}%')
    print(f'MdAPE (dollar-scale): {real_world_mdape(y_pred, y_test):.2f}%')
