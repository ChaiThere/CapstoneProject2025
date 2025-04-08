# Loading the libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Loading the dataset for Cardano
url = "C:/Users/shirshak/Downloads/Cardano Historical Data.csv"  # Update path if necessary
cardano = pd.read_csv(url, parse_dates=["Date"], index_col='Date').sort_index()

# Assuming the CSV contains a 'Price' column
cardano = cardano[['Price']]

# Plotting the original Cardano Price
plt.figure(figsize=(12, 6))
plt.plot(cardano['Price'], color='blue')
plt.title('Cardano (ADA) Price (2019–2024)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.show()

# -------------------------
# Combined Log and Differencing Transformation
# -------------------------
# Log transform the price
cardano['LogPrice'] = np.log(cardano['Price'])

# Take the first difference of the log-transformed price
cardano['LogDiffPrice'] = cardano['LogPrice'].diff()

# Drop the NaN created by differencing
cardano_transformed = cardano.dropna()

# Plot the transformed series
plt.figure(figsize=(12, 6))
plt.plot(cardano_transformed['LogDiffPrice'], color='purple')
plt.title('Cardano Log-Differenced Price')
plt.xlabel('Date')
plt.ylabel('Log Difference')
plt.show()

# -------------------------
# Stationarity Test on the Log-Differenced Data
# -------------------------
def adf_test(series):
    result = adfuller(series.dropna())
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'  {key}: {value}')

print("Stationarity Test for Cardano Log-Differenced Price:")
adf_test(cardano_transformed['LogDiffPrice'])

# -------------------------
# Train/Test Split
# -------------------------
train_size = int(len(cardano_transformed) * 0.8)
train = cardano_transformed.iloc[:train_size]
test = cardano_transformed.iloc[train_size:]

# -------------------------
# Auto-ARIMA Model on Transformed Data
# -------------------------
model = auto_arima(
    train['LogDiffPrice'],
    seasonal=False,
    trace=True,
    stepwise=True,
    suppress_warnings=True,
    error_action='ignore'
)

print(model.summary())

# -------------------------
# Forecast on the Log-Differenced Scale
# -------------------------
n_periods = len(test)
forecast_diff, conf_int_diff = model.predict(
    n_periods=n_periods,
    return_conf_int=True
)

forecast_diff_series = pd.Series(forecast_diff, index=test.index)

# -------------------------
# Back-Transform Forecast to Original Price Scale
# -------------------------
# To revert the differenced logs, add the forecasted differences cumulatively to the last observed log price in the training set.
last_log_price = train['LogPrice'].iloc[-1]
forecast_log = last_log_price + forecast_diff_series.cumsum()

# Exponentiate to get the price forecast in original scale
forecast_price = np.exp(forecast_log)

# Similarly, back-transform the confidence intervals
conf_int_lower = np.exp(last_log_price + np.cumsum(conf_int_diff[:, 0]))
conf_int_upper = np.exp(last_log_price + np.cumsum(conf_int_diff[:, 1]))
conf_int = np.column_stack([conf_int_lower, conf_int_upper])

# -------------------------
# Evaluate the Forecast
# -------------------------
# Note: To compare forecasts to the actual prices, we use the original price data in the test set.
# Extract the actual price corresponding to the forecast period
actual_price = cardano.loc[test.index]['Price']

mae = mean_absolute_error(actual_price, forecast_price)
rmse = np.sqrt(mean_squared_error(actual_price, forecast_price))
mape = np.mean(np.abs((actual_price - forecast_price) / actual_price)) * 100
r2 = r2_score(actual_price, forecast_price)

print(f'\nMAE: {mae:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'MAPE: {mape:.2f}%')
print(f"R²: {r2:.4f}")

# -------------------------
# Visualize Forecast vs Actual Prices
# -------------------------
plt.figure(figsize=(14, 6))
plt.plot(cardano.loc[:train.index[-1]]['Price'], label='Train', color='blue')
plt.plot(actual_price, label='Actual', color='green')
plt.plot(forecast_price, label='Forecast', color='red', linestyle='--')
plt.fill_between(test.index, conf_int[:, 0], conf_int[:, 1], color='gray', alpha=0.2)
plt.title('Cardano (ADA) Price Forecast (Back-transformed from Log-Differenced Data)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()
