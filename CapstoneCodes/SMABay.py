import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
import os
from pytrends.request import TrendReq
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import random

# Feature Engineering: Moving Averages

def add_moving_averages(df, column, windows=[7, 14, 30]):
    """
    Adds Simple Moving Averages (SMA) and Exponential Moving Averages (EMA) to the DataFrame.
    """
    for window in windows:
        df[f'{column}_SMA_{window}'] = df[column].rolling(window=window, min_periods=1).mean()
        df[f'{column}_EMA_{window}'] = df[column].ewm(span=window, adjust=False).mean()
    return df

# Data Collection

def load_crypto_data_from_csv(file_path, crypto_name):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    if 'Price' in df.columns:
        df.rename(columns={'Price': f'{crypto_name}_Close'}, inplace=True)
    else:
        df.rename(columns={'Close': f'{crypto_name}_Close'}, inplace=True)
    df[f'{crypto_name}_Close'] = df[f'{crypto_name}_Close'].replace({',': ''}, regex=True)
    df[f'{crypto_name}_Close'] = pd.to_numeric(df[f'{crypto_name}_Close'], errors='coerce')
    return df[['Date', f'{crypto_name}_Close']]

# Execution
def main():
    start_date = pd.to_datetime("2023-02-24").date()
    end_date = pd.to_datetime("2025-02-24").date()
    base_path = r"C:\Users\Sahil\OneDrive\Desktop\Capstone Project"
    file_map = {
        'BTC': 'Bitcoin Historical Data.csv',
        'ETH': 'Ethereum Historical Data.csv',
        'XRP': 'XRP Historical Data.csv',
        'ADA': 'Cardano Historical Data.csv'
    }
    crypto_list = ['BTC', 'ETH', 'XRP', 'ADA']
    crypto_dfs = []
    for crypto in crypto_list:
        file_path = os.path.join(base_path, file_map[crypto])
        try:
            df = load_crypto_data_from_csv(file_path, crypto)
            df = add_moving_averages(df, f'{crypto}_Close')
            crypto_dfs.append(df)
            print(f"Loaded data for {crypto}")
        except Exception as e:
            print(f"Error loading {crypto}: {e}")
    merged_crypto_df = pd.concat(crypto_dfs, axis=1).dropna()
    feature_cols = ['BTC_Close', 'BTC_Close_SMA_7', 'BTC_Close_SMA_14', 'BTC_Close_SMA_30',
                    'BTC_Close_EMA_7', 'BTC_Close_EMA_14', 'BTC_Close_EMA_30']
    df_features = merged_crypto_df[feature_cols]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_features)
    seq_length = 30
    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i-seq_length:i])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    X_train, X_val = X_train[:int(len(X_train) * 0.9)], X_train[int(len(X_train) * 0.9):]
    y_train, y_val = y_train[:int(len(y_train) * 0.9)], y_train[int(len(y_train) * 0.9):]
    
    def optimize_lstm(X_train, y_train, X_val, y_val, input_shape):
        def lstm_evaluate(lstm_units, dropout_rate, learning_rate):
            model = Sequential([
                LSTM(int(lstm_units), return_sequences=True, input_shape=input_shape),
                Dropout(dropout_rate),
                LSTM(int(lstm_units), return_sequences=True),
                Dropout(dropout_rate),
                LSTM(int(lstm_units)),
                Dropout(dropout_rate),
                Dense(1)
            ])
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error')
            history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), verbose=0)
            return -min(history.history['val_loss'])
        pbounds = {'lstm_units': (32, 128), 'dropout_rate': (0.1, 0.5), 'learning_rate': (0.0001, 0.01)}
        optimizer = BayesianOptimization(f=lstm_evaluate, pbounds=pbounds, verbose=2, random_state=42)
        optimizer.maximize(init_points=5, n_iter=15)
        return optimizer.max
    
    best_params = optimize_lstm(X_train, y_train, X_val, y_val, (X_train.shape[1], X_train.shape[2]))
    best_units = int(best_params['params']['lstm_units'])
    best_dropout = best_params['params']['dropout_rate']
    best_lr = best_params['params']['learning_rate']
    model = Sequential([
        LSTM(best_units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(best_dropout),
        LSTM(best_units, return_sequences=True),
        Dropout(best_dropout),
        LSTM(best_units),
        Dropout(best_dropout),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_lr), loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))
    y_pred = model.predict(X_test)
    y_test_inv = scaler.inverse_transform(np.c_[y_test, np.zeros((len(y_test), len(feature_cols) - 1))])[:, 0]
    y_pred_inv = scaler.inverse_transform(np.c_[y_pred.flatten(), np.zeros((len(y_pred), len(feature_cols) - 1))])[:, 0]
    mape = mean_absolute_percentage_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    print(f"MAPE: {mape * 100:.2f}%")
    print(f"RMSE: {rmse:.2f}")
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_inv, label="True Price")
    plt.plot(y_pred_inv, label="Predicted Price")
    plt.title("BTC Price Prediction using LSTM with Bayesian Optimization")
    plt.xlabel("Time Steps")
    plt.ylabel("BTC Price (USD)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
