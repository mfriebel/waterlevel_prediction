#%%
# Import 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, datetime
from statsmodels.tsa.arima_process import index2lpol
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from fbprophet import Prophet
from catboost import CatBoostRegressor
sns.set()

import eda
from function_model import RSME, MAE, cv_split, plot_model_performance, series_to_supervised

#%%
# Read data
df = pd.read_csv("./data/Aquifer_Petrignano.csv", index_col=0, parse_dates=True, dayfirst=True)
df.index.freq = 'D'

# Rename columns
columns = ['Rf_BU', 'DtG_P24', 'DtG_P25', 'T_BU', 'T_Pe', 'V_Pe', 'H_FCP']
df.columns = columns

#%%
# Check data
print(df.count) # All total amount of values
print(df.isna().sum()) # Missing values
print((df.index - df.index.shift(-1)).value_counts()) # Check for consistent spacing

eda.plot_time_series(df, '2006-04-01', '2020-06-30', (15, 20))
#%%
# Replace zero values
df['T_Pe'] = np.where(((df['T_Pe'] == 0) & (df['T_Pe'].shift(1) == 0)) | (df['T_Pe'].diff(-1) > 10) | ((df['T_Pe'] == 0) & (df['T_Pe'].shift(-1) == 0)), np.nan, df['T_Pe'])
df['V_Pe'] = np.where(df['V_Pe'] == 0, np.nan, df['V_Pe'])
df['H_FCP'] = np.where(df['H_FCP'] == 0, np.nan, df['H_FCP'])

#%%
# Impute missing values
df['T_Pe'] = df['T_Pe'].fillna(df['T_BU'])
df['V_Pe'] = df['V_Pe'].interpolate()
df['H_FCP'] = df['H_FCP'].interpolate()
df['DtG_P25'] = df['DtG_P25'].interpolate()
df['DtG_P24'] = df['DtG_P24'].interpolate()
df= df.loc['2009':] # Cut Dataframe observed values for all features

#%%
#Downsampling
col_mean = ['DtG_P25', 'H_FCP', 'V_Pe', 'T_BU']
col_sum = 'Rf_BU'
df_weekly = df[col_mean].resample('7D').mean()
df_weekly[col_sum ] = df[[col_sum]].resample('7D').sum()

#%%
# Train-Test-Split
width = 52
for column in df_weekly:
    lag = df_weekly.shift(width - 1)
    df_weekly[f'{column}_trend'] = lag[column].rolling(window=width).mean()

df_weekly['V_Pe_trend_4M'] = df_weekly['V_Pe_trend'].shift(4*4)
df_weekly['H_FCP_trend_6M'] = df_weekly['H_FCP_trend'].shift(6*4)
df_weekly['Rf_BU_trend_6M'] = df_weekly['Rf_BU_trend'].shift(6*4) 

freq = '7D'
train = df_weekly[:-52]
test = df_weekly[-52:]
y_train = train['DtG_P25']
y_test = test['DtG_P25']

##% 
# Baseline Model - Naive Approach
y_pred_naive = pd.Series(y_train[-1], index=test.index)
plot_model_performance(y_train, y_pred_naive, y_test, 'Baseline-Naive', '2016-01-01', '2020-07-01')

#%%

result = adfuller(y_train)
print('ADF Statistic: %f' % result[0] + '\np-value: %f' % result[1])

#%%
# ARIMA - univariant
mod = ARIMA(y_train, order=(1,1,1), freq=freq)
res = mod.fit() 
res.summary()

residuals = pd.DataFrame(res.resid)
f, ax = plt.subplots(1, 2, figsize=(30,5))
residuals.plot(title='Residuals', ax=ax[0])
ax[0].set_ylim([-1,1])
residuals.plot(title='Density', kind='kde', ax=ax[1])
ax[1].set_xlim([-1,1])
plt.show()

plot_model_performance(y_train, res.forecast(len(test.index)), y_test, 'ARIMA(1,1,1)', '2016-01-01', '2020-07-01')

#%%
# Facebook Prophet - Simple
y_train_prophet = y_train.reset_index()
y_train_prophet.columns = ['ds', 'y']

m_prophet_1 = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
m_prophet_1.fit(y_train_prophet)

future_frame = m_prophet_1.make_future_dataframe(periods=len(y_test), freq=freq, include_history=False)
prophet_simple = m_prophet_1.predict(future_frame)

plot_model_performance(y_train, prophet_simple.set_index('ds').yhat, y_test, 'Prophet_Simple', '2016-01-01', '2020-07-01')
#%%
# Facebook Prophet - Multivariat
train_prophet = train.reset_index()
train_prophet.rename(columns = {'Date' : 'ds', 'DtG_P25' : 'y'}, inplace=True)

regressors = ['T_BU', 'Rf_BU', 'V_Pe', 'H_FCP']
m_prophet_2 = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
for regressor in regressors:
    m_prophet_2.add_regressor(regressor, mode='additive')
m_prophet_2.fit(train_prophet)

future_frame_cpx = m_prophet_2.make_future_dataframe(periods=len(y_test), freq=freq, include_history=False)
future_frame_cpx = future_frame.join(test[regressors].reset_index(drop=True))
prophet_complex = m_prophet_2.predict(future_frame_cpx)

plot_model_performance(y_train, prophet_complex.set_index('ds').yhat, y_test, 'Prophet_Simple', '2016-01-01', '2020-07-01')

#%%
# GradientBoostingTrees

# Features
x_columns = ['T_BU', 'Rf_BU', 'V_Pe', 'H_FCP']

y_train = train['DtG_P25']
X_train = train[x_columns]
X_train = X_train.join(pd.get_dummies(season(X_train.index), drop_first=True).set_index(X_train.index))
X_train['year'] = X_train.index.year
X_train['month'] = (X_train.index.year - 2008) * X_train.index.quarter

y_test = test['DtG_P25']
X_test = test[x_columns]
X_test = X_test.join(pd.get_dummies(season(X_test.index), drop_first=True).set_index(X_test.index))
X_test['year'] = X_test.index.year
X_test['month'] = (X_test.index.year - 2008) * X_test.index.quarter

#%%
## Simple Boosting Tree

boost_simple = CatBoostRegressor(iterations=100, learning_rate=0.05, depth=3, verbose=False)
boost.fit(X_train, y_train)

yhat = pd.Series(boost.predict(X_train), index=X_train.index)

f, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,5))
sns.lineplot(x = y_train.index, y=y_train, label='training')
sns.lineplot(x = yhat.index, y=yhat, label='prediction')
plt.show()

pd.Series(boost.feature_importances_, index=X_train.columns).plot(kind='bar')

## Cross-Validation

X_training, X_validation = cv_split(X_train, 5)
y_training, y_validation = cv_split(y_train, 5)

