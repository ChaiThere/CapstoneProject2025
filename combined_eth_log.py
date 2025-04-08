import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- 1. Data Loading and Cleaning ---
data_path = "C:/Users/shirshak/Downloads/Ethereum Historical Data.csv"
df = pd.read_csv(data_path, parse_dates=['Date'], index_col='Date').sort_index()

# Remove commas and convert 'Price' to numeric
df['Price'] = df['Price'].str.replace(',', '', regex=True)
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df.dropna(subset=['Price'], inplace=True)

# --- 2. Create Transformations ---
# Log Transformation: stabilizes variance
df['LogPrice'] = np.log(df['Price'])

# Log Differencing: removes trend from log data
df['LogPriceDiff'] = df['LogPrice'].diff()
df_diff = df['LogPriceDiff'].dropna()  # drop the NaN from first diff

# --- 3. Train/Test Split ---
# We'll use an 80/20 split for both approaches.
split_index = int(len(df) * 0.8)
# For the log-only model, use all available log prices.
train_log = df.iloc[:split_index].copy()
test_log = df.iloc[split_index:].copy()

# For the log-differenced model, note that differencing loses the first observation.
# Align the series accordingly.
train_diff = df_diff.iloc[:(split_index-1)].copy()  # because first diff is lost in training
test_diff = df_diff.iloc[(split_index-1):].copy()     # test_diff now has the same length as test_log

# --- 4. Build and Forecast with Log-Transformed Model ---
# Auto_ARIMA will decide on the necessary differencing.
model_log = auto_arima(
    train_log['LogPrice'],
    seasonal=False,
    trace=True,
    stepwise=True,
    suppress_warnings=True,
    error_action='ignore'
)
print("Log-Transformed Model Summary:")
print(model_log.summary())

n_periods = len(test_log)
forecast_log, conf_int_log = model_log.predict(n_periods=n_periods, return_conf_int=True)

# Convert forecast from log scale to original price scale
forecast_log_price = np.exp(forecast_log)
conf_int_log_price = np.exp(conf_int_log)

forecast_index = test_log.index
forecast_series_log = pd.Series(forecast_log_price, index=forecast_index)

# --- 5. Build and Forecast with Log-Differenced Model ---
# Build the model on the differenced series.
model_diff = auto_arima(
    train_diff,
    seasonal=False,
    trace=True,
    stepwise=True,
    suppress_warnings=True,
    error_action='ignore'
)
print("\nLog-Differenced Model Summary:")
print(model_diff.summary())

n_periods_diff = len(test_diff)
forecast_diff, conf_int_diff = model_diff.predict(n_periods=n_periods_diff, return_conf_int=True)

# To get forecasts on the log-price scale, integrate (cumulative sum) the forecasted differences.
# Use the last value of the training log-price as the starting point.
last_train_log_price = train_log['LogPrice'].iloc[-1]
forecast_log_from_diff = last_train_log_price + np.cumsum(forecast_diff)

# Integrate the confidence intervals
conf_int_lower = last_train_log_price + np.cumsum(conf_int_diff[:, 0])
conf_int_upper = last_train_log_price + np.cumsum(conf_int_diff[:, 1])

# Convert integrated log forecasts back to original price scale
forecast_diff_price = np.exp(forecast_log_from_diff)
conf_int_diff_price = np.column_stack((np.exp(conf_int_lower), np.exp(conf_int_upper)))

# Use the full test_log index for alignment
forecast_index_diff = test_log.index
forecast_series_diff = pd.Series(forecast_diff_price, index=forecast_index_diff)

# --- 6. Evaluation of Both Approaches ---
# For the Log-Transformed model:
actual_log = test_log['Price']
mae_log = mean_absolute_error(actual_log, forecast_series_log)
rmse_log = np.sqrt(mean_squared_error(actual_log, forecast_series_log))
mape_log = np.mean(np.abs((actual_log - forecast_series_log) / actual_log)) * 100
r2_log = r2_score(actual_log, forecast_series_log)

print("\nEvaluation Metrics for Log-Transformed Model:")
print(f"MAE: {mae_log:.4f}")
print(f"RMSE: {rmse_log:.4f}")
print(f"MAPE: {mape_log:.2f}%")
print(f"R²: {r2_log:.4f}")

# For the Log-Differenced model, compare forecasts to the full test set.
actual_diff = test_log['Price']
mae_diff = mean_absolute_error(actual_diff, forecast_series_diff)
rmse_diff = np.sqrt(mean_squared_error(actual_diff, forecast_series_diff))
mape_diff = np.mean(np.abs((actual_diff - forecast_series_diff) / actual_diff)) * 100
r2_diff = r2_score(actual_diff, forecast_series_diff)

print("\nEvaluation Metrics for Log-Differenced Model:")
print(f"MAE: {mae_diff:.4f}")
print(f"RMSE: {rmse_diff:.4f}")
print(f"MAPE: {mape_diff:.2f}%")
print(f"R²: {r2_diff:.4f}")

# --- 7. Plotting the Forecasts ---
plt.figure(figsize=(14, 6))
plt.plot(train_log['Price'], label='Train', color='blue')
plt.plot(test_log['Price'], label='Actual', color='green')
plt.plot(forecast_series_log, label='Forecast (Log Model)', color='red', linestyle='--')
plt.fill_between(
    forecast_index,
    conf_int_log_price[:, 0],
    conf_int_log_price[:, 1],
    color='gray',
    alpha=0.2,
    label='Confidence Interval (Log Model)'
)
plt.title('Ethereum Price Forecast (Log-Transformed Model)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

plt.figure(figsize=(14, 6))
plt.plot(train_log['Price'], label='Train', color='blue')
plt.plot(test_log['Price'], label='Actual', color='green')
plt.plot(forecast_series_diff, label='Forecast (Log-Differenced Model)', color='purple', linestyle='--')
plt.fill_between(
    forecast_index_diff,
    conf_int_diff_price[:, 0],
    conf_int_diff_price[:, 1],
    color='orange',
    alpha=0.2,
    label='Confidence Interval (Log-Diff Model)'
)
plt.title('Ethereum Price Forecast (Log-Differenced Model)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()
