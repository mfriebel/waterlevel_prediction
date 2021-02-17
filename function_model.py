import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA


def RSME(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2)).round(2)

def MAE(y_true, y_pred):
    return round(np.mean(np.abs(y_true - y_pred)),2)

def cv_split(data, splits):
    '''Splits training data in moving training and validations sets'''
    split_len = int(len(data)/(splits+1))
    
    train_splits = []
    val_splits = []
    for split in range(1,splits+1):
        train = data[:split*split_len]
        val = data[split*split_len:split*split_len+split_len]
        train_splits.append(train)
        val_splits.append(val)
        
    return train_splits, val_splits

def create_season(date):
    return date.month%12 // 3 + 1

# Adapted machinelearningmastery.com (https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/)
def series_to_supervised(df, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Pandas DataFrame
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
    Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = df.shape[1]
    var_name = df.columns
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [(f'{var_name[j]}(t-%d)' % (i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [(f'{var_name[j]}(t)') for j in range(n_vars)]
        else:
            names += [(f'{var_name[j]}' % (i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def plot_model_performance(y_train, y_pred, y_test, title, min_date, max_date, figsize=(20,5)):
    plt.figure(figsize=figsize)
    sns.lineplot(x = y_train.index, y=y_train, label='training')
    sns.lineplot(x = y_pred.index, y=y_pred, label='pred_mean')
    sns.lineplot(x = y_test.index, y=y_test, label='test')
    plt.title(f'{title} -  RSME: {RSME(y_test, y_pred)}, MAE: {MAE(y_test, y_pred)}')
    plt.xlim([datetime.strptime(min_date, "%Y-%m-%d"), datetime.strptime(max_date, "%Y-%m-%d")])
    plt.show()

def walk_forward_prediction_ARIMA(y_train, y_test, window):
    forecast_series = pd.DataFrame(index=y_test.index, columns=[x for x in range(window)])
    training = y_train.copy()
    for date in y_test.index:
        res_m = ARIMA(training, order=(1,1,1)).fit()
        forecast_series.loc[date] = res_m.forecast(window)[0]
        training.loc[date] = y_test.loc[date]
    columns = [f'yt+{x}' for x in forecast_series.columns]
    forecast_series.columns = columns
    return forecast_series.astype(float)

def arima_time_series_cv(data, splits, metric, order=(1,1,1)):
    split_len = int(len(data)/(splits+1))
    
    error = []
    for split in range(1,splits+1):
        train = data[:split*split_len]
        test = data[split*split_len:split*split_len+split_len]
        model = ARIMA(train, order=order)
        res = model.fit()
        y_pred = res.forecast(len(test.index))
        error.append(metric(test, y_pred))
        
    return error