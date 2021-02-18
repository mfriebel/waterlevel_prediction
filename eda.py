import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, datetime
from statsmodels.tsa.seasonal import seasonal_decompose

def plot_time_series(dataframe, min_date, max_date, plotsize, label_dict=None, sharex=True):
    '''Plot time series data'''
    
    if type(dataframe) == pd.Series:
        f, ax = plt.subplots(nrows=1, ncols=1, figsize=plotsize)
        sns.lineplot(x=dataframe.index, y=dataframe)
        ax.set_xlim([datetime.strptime(min_date, "%Y-%m-%d"), datetime.strptime(max_date, "%Y-%m-%d")])
        if label_dict == None:
            None
        else:
            ax.set_ylabel(label_dict[dataframe.name])

    else:
        f, ax = plt.subplots(nrows=len(dataframe.columns), ncols=1, figsize=plotsize, sharex=sharex)
        for element in enumerate(dataframe.columns):
            sns.lineplot(x=dataframe.index, y=dataframe[element[1]], ax=ax[element[0]])
            ax[element[0]].set_xlim([datetime.strptime(min_date, "%Y-%m-%d"), datetime.strptime(max_date, "%Y-%m-%d")])
            
            if label_dict == None:
                None
            else:
                ax[element[0]].set_ylabel(label_dict[element[1]])
    return f

def plot_nan_replacement(series, min_date, max_date, plotsize, sharex=True):
    '''Plot replacement of NaN with mean, forward-fill and interploation'''
    f, ax = plt.subplots(nrows=3, ncols=1, figsize=plotsize, sharex=sharex)
    subset = series.loc[min_date : max_date]

    sns.lineplot(x=subset.index, y=subset.fillna(subset.mean()), label='mean', ax=ax[0])
    sns.lineplot(x=subset.index, y=subset, label='original', ax=ax[0])

    sns.lineplot(x=subset.index, y=subset.ffill(), label='ffill', ax=ax[1])
    sns.lineplot(x=subset.index, y=subset, label='original', ax=ax[1])

    sns.lineplot(x=subset.index, y=subset.interpolate(), label='interpolate', ax=ax[2])
    sns.lineplot(x=subset.index, y=subset, label='original', ax=ax[2])

    plt.legend()
    plt.show()

def decompose_features(series, period, model_type='additive', plotsize=(20,10), sharex=True):
    '''Plot decomposed (based on moving average) time-series and return results object'''
    results = seasonal_decompose(series, model=model_type, period=period, extrapolate_trend='freq')
    f, ax = plt.subplots(nrows=4, ncols=1, figsize=plotsize, sharex=sharex)
    sns.lineplot(x=series.index, y=results.observed, ax=ax[0])
    sns.lineplot(x=series.index, y=results.trend, ax=ax[1])
    sns.lineplot(x=series.index, y=results.seasonal, ax=ax[2])
    sns.lineplot(x=series.index, y=results.resid, ax=ax[3])
    plt.show()
    
    return results

def plot_all_decomposed_trends(df, period, min_date, max_date, label_dict, model_type='additive', plotsize=(20,15), sharex=True):
    features = df.columns

    f, ax = plt.subplots(nrows=len(features), ncols=1, figsize=plotsize, sharex=sharex)

    for i in range(len(features)):
        decomposed = seasonal_decompose(df[features[i]], model=model_type, period=period, extrapolate_trend='freq')
        trend = decomposed.trend
        trend = trend.loc[min_date : max_date]
        sns.lineplot(x=trend.index, y=trend, ax=ax[i])
        ax[i].set_ylabel(label_dict[features[i]])
        #ax[i].axvline(x=date(2012, 9, 1), color='grey')

    #plt.legend()
    return f

def plot_all_decomposed_seasonality(df, period, min_date, max_date, label_dict, model_type='additive', plotsize=(20,15), sharex=True):
    features = df.columns

    f, ax = plt.subplots(nrows=len(features), ncols=1, figsize=plotsize, sharex=sharex)

    for i in range(len(features)):
        decomposed = seasonal_decompose(df[features[i]], model=model_type, period=period, extrapolate_trend='freq')
        seasonal = decomposed.seasonal
        seasonal = seasonal.loc[min_date : max_date]
        sns.lineplot(x=seasonal.index, y=seasonal, ax=ax[i])
        ax[i].set_ylabel(label_dict[features[i]])
        #ax[i].set_ylim([df[features[i]].min(), df[features[i]].max()])

    return f

def ma_decompose(series, window_trend, window_seasonality):
    trend = series.rolling(window_trend).mean()
    seasonality = (series - trend).rolling(window_seasonality).mean()
    residual = series - trend - seasonality
    
    return trend, seasonality, residual 

def plot_decomposition(series, trend, seasonality, residual):
    f, ax = plt.subplots(nrows=4, ncols=1, sharex=True, figsize = (20,10))
    ax[0].plot(series)
    ax[0].set_ylim([series.min(), series.max()])
    ax[0].set_ylabel = 'Observed'
    ax[1].plot(trend)
    ax[1].set_ylim([series.min(), series.max()])
    ax[1].set_ylabel = 'Trend'
    ax[2].plot(seasonality)
    ax[2].set_ylabel = 'Seasonality'
    ax[3].plot(residual)
    ax[3].set_ylabel = 'Residual'
    #plt.title(series.name)
    plt.show()