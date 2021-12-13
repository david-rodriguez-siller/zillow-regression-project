import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def sse(y, yhat):
    return (residuals(y, yhat) ** 2).sum()

def mse(y, yhat):
    n = y.shape[0]
    return sse(y, yhat) / n

def rmse(y, yhat):
    return (mse(y, yhat)) ** .5

def ess(y, yhat):
    return ((yhat - y.mean()) ** 2).sum()

def tss(y):
    return ((y - y.mean()) ** 2).sum()

def r2_score(y, yhat):
    return ess(y, yhat) / tss(y)

def regression_errors(y, yhat):
    return pd.Series({
        'sse': sse(y, yhat),
        'ess': ess(y, yhat),
        'tss': tss(y),
        'mse': mse(y, yhat),
        'rmse': rmse(y, yhat),
    })

def baseline_mean_errors(y):
    yhat = y.mean()
    return {
        'sse': sse(y, yhat),
        'mse': mse(y, yhat),
        'rmse': rmse(y, yhat),
    }

def better_than_baseline(y, yhat):
    rmse_baseline = rmse(y, y.mean())
    rmse_model = rmse(y, yhat)
    return rmse_model < rmse_baseline