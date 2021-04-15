import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
#from fbprophet import Prophet
from sklearn.metrics import r2_score


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

def plot_model_performance(y_train, y_pred, y_test, title ,min_date, max_date, figsize=(20,5)):
    f = plt.figure(figsize=figsize)
    sns.lineplot(x = y_train.index, y=y_train, label='training')
    sns.lineplot(x = y_pred.index, y=y_pred, label='prediction')
    sns.lineplot(x = y_test.index, y=y_test, label='test')
    plt.title(f'{title} -  RSME: {RSME(y_test, y_pred)}, MAE: {MAE(y_test, y_pred)}')
    plt.xlim([datetime.strptime(min_date, "%Y-%m-%d"), datetime.strptime(max_date, "%Y-%m-%d")])
    return f

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

def walkforward_data_ls(training, test, window, freq):
     
    if freq == 'W' or freq == '7D':
        date_diff = timedelta(weeks=window-1)
    elif freq == 'D':
        date_diff = timedelta(weeks=window-1)
    else:
        raise TypeError 
    
    training_list = []
    test_list = []
    split_list = test.index[::window]

    for i in range(len(split_list)):
        date = split_list[i]
        if i == 0:
            training_list.append(training)
            test_list.append(test.loc[date : date+date_diff])
        else:
            date_train = split_list[i-1]
            training_list.append(pd.concat([training, test.loc[split_list[0] : date_train+date_diff]]))
            test_list.append(test.loc[date : date+date_diff])
        
    return training_list, test_list

def walkforward_baseline(y_train, y_test, freq, window):
    ## walkforward baseline mean
    wlk_fwd_y_train, wlk_fwd_y_test = walkforward_data_ls(y_train, y_test, window, freq)

    baseline_pred_list = []
    for i in range(len(wlk_fwd_y_train)):
        
        fit_model = wlk_fwd_y_train[i].mean() # calculate overall mean of training set 
        # append mean x-times (window) to list
        baseline_pred_list.append([fit_model for i in range(len(wlk_fwd_y_test[i]))]) 

    # reduce list of lists to a list    
    baseline_pred_list = np.concatenate(baseline_pred_list).ravel().tolist()
    baseline_forecast = pd.Series(baseline_pred_list, index=y_test.index) 

    return baseline_forecast

def walkforward_naive(y_train, y_test, freq, window):
    ## walkforward naive approach 
    wlk_fwd_y_train, wlk_fwd_y_test = walkforward_data_ls(y_train, y_test, window, freq)

    pred_list = []
    for i in range(len(wlk_fwd_y_train)):
        
        fit_model = wlk_fwd_y_train[i][-1] # use last values of training set for pred
        # append mean x-times (window) to list
        pred_list.append([fit_model for i in range(len(wlk_fwd_y_test[i]))]) 

    # reduce list of lists to a list    
    pred_list = np.concatenate(pred_list).ravel().tolist()
    baseline_forecast = pd.Series(pred_list, index=y_test.index) 

    return baseline_forecast

def walkforward_ARIMA(y_train, y_test, arima_order, freq, window):
    ## walkforward ARIMA
    wlk_fwd_y_train, wlk_fwd_y_test = walkforward_data_ls(y_train, y_test, window, freq)

    pred_list = []
    for i in range(len(wlk_fwd_y_train)):
        
        res =  ARIMA(wlk_fwd_y_train[i], order=arima_order).fit() 
        # append mean x-times (window) to list
        pred_list.append(res.forecast(window)[0]) 

    # reduce list of lists to a list    
    pred_list = np.concatenate(pred_list).ravel().tolist()
    diff = len(pred_list) - len(y_test)
    if diff > 0:
        forecast = pd.Series(pred_list[:-diff], index=y_test.index)
    else:
        forecast = pd.Series(pred_list, index=y_test.index)  

    return forecast

def walkforward_Prophet_uni(y_train, y_test, freq, window):
    ## walkforward ARIMA
    wlk_fwd_y_train, wlk_fwd_y_test = walkforward_data_ls(y_train, y_test, window, freq)

    pred_list = []
    for i in range(len(wlk_fwd_y_train)):
        
        training = wlk_fwd_y_train[i].reset_index()
        training.columns = ['ds', 'y']

        m_prophet = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
        m_prophet.fit(training)
        future_frame = m_prophet.make_future_dataframe(periods=window, freq='D', include_history=False)
        forecast = m_prophet.predict(future_frame)
        pred_list.append(forecast.yhat.values) 

    # reduce list of lists to a list    
    pred_list = np.concatenate(pred_list).ravel().tolist()
    diff = len(pred_list) - len(y_test)
    if diff > 0:
        forecast = pd.Series(pred_list[:-diff], index=y_test.index)
    else:
        forecast = pd.Series(pred_list, index=y_test.index)   

    return forecast

def walkforward_Catboost(model, X_train, y_train, X_test, y_test, freq, window):
    walk_fwd_X_train, walk_fwd_X_test = walkforward_data_ls(X_train, X_test, window, freq)
    walk_fwd_y_train, walk_fwd_y_test = walkforward_data_ls(y_train, y_test, window, freq)

    boost_pred_list = []
    for i in range(len(walk_fwd_X_train)):
        
        model.fit(walk_fwd_X_train[i], walk_fwd_y_train[i])
        boost_pred_list.append(model.predict(walk_fwd_X_test[i]))
        
    boost_pred_list = np.concatenate(boost_pred_list).ravel().tolist()
    boost_forecast = pd.Series(boost_pred_list, index=y_test.index) 

    return boost_forecast

def plot_CV_Catboost(model, X_train, y_train, splits):
    X_training, X_validation = cv_split(X_train, splits)
    y_training, y_validation = cv_split(y_train, splits)

    f, ax = plt.subplots(nrows=splits, ncols=1, figsize=(20,15), sharex=True)
    for i in range(splits):
        model.fit(X_training[i], y_training[i])
        prediction = pd.Series(model.predict(X_validation[i]), index=X_validation[i].index)
        
        sns.lineplot(x = y_training[i].index, y=y_training[i], label='training', ax=ax[i])
        sns.lineplot(x = prediction.index, y=prediction, label='CatBoost_prediction', ax=ax[i])
        sns.lineplot(x = y_validation[i].index, y=y_validation[i], label='validation', ax=ax[i])
        ax[i].set_title(f'CatBoost - R2: {round(r2_score(y_validation[i], prediction),3)}, RSME: {RSME(y_validation[i], prediction)}, MAE: {MAE(y_validation[i], prediction)}')
    plt.show()