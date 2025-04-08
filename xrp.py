#Lodaing the libraries

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

#Lodaing the dataset
url = "C:/Users/shirshak/Downloads/XRP_Historical_Data.csv"
xrp = pd.read_csv(url, parse_dates=["Date"], index_col='Date').sort_index()

xrp = xrp[['Price']]

#print("XRP Data Head:")
#print(xrp.head())

#print(xrp.tail())

#plotting the XRP PRICE

plt.figure(figsize=(12, 6))
plt.plot(xrp['Price'], color='blue')
plt.title('XRP Price (2019–2024)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.show()

# Checking Stationarity
def adf_test(series):
    result = adfuller(series.dropna())
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'  {key}: {value}')

print("Stationarity Test for XRP Price:")
adf_test(xrp['Price'])

#  Train/Test Split
train_size = int(len(xrp) * 0.8)
train = xrp.iloc[:train_size]
test = xrp.iloc[train_size:]

#  Auto-ARIMA Model
model = auto_arima(
    train['Price'],
    seasonal=False,
    trace=True,
    stepwise=True,
    suppress_warnings=True,
    error_action='ignore'
)

print(model.summary())


# Forecast & Evaluate
forecast, conf_int = model.predict(
    n_periods=len(test),
    return_conf_int=True
)

forecast_dates = test.index
forecast_series = pd.Series(forecast, index=forecast_dates)


# Calculate metrics
mae = mean_absolute_error(test['Price'], forecast)
rmse = np.sqrt(mean_squared_error(test['Price'], forecast))
mape = np.mean(np.abs((test['Price'] - forecast) / test['Price'])) * 100
r2 = r2_score(test['Price'], forecast)

print(f'\nMAE: {mae:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'MAPE: {mape:.2f}%')
print(f"R²: {r2:.4f}")

# Visualize Forecast
plt.figure(figsize=(14, 6))
plt.plot(train['Price'], label='Train', color='blue')
plt.plot(test['Price'], label='Actual', color='green')
plt.plot(forecast_series, label='Forecast', color='red', linestyle='--')
plt.fill_between(forecast_dates, conf_int[:, 0], conf_int[:, 1], color='gray', alpha=0.2)
plt.title('XRP Price Forecast with ARIMA')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()