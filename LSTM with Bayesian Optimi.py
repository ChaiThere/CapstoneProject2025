# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:45:50 2025

@author: Sahil
"""

import pandas as pd
import numpy as np
import yfinance as yf  # For Yahoo Finance data
from pytrends.request import TrendReq
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
import random
import os
import keras_tuner as kt  # Bayesian optimization with Keras Tuner

# ---------------------------
# Data Collection Functions
# ---------------------------

def load_crypto_data_from_yahoo(ticker, crypto_name, start_date, end_date):
    """
    Downloads cryptocurrency data from Yahoo Finance and renames the 'Close' column.
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    data = data[['Date', 'Close']]
    data.rename(columns={'Close': f'{crypto_name}_Close'}, inplace=True)
    return data

def get_google_trends_data(start_date, end_date, keyword='bitcoin'):
    """
    Gets Google Trends data for a given keyword.
    """
    pytrend = TrendReq(hl='en-US', tz=360)
    timeframe = f'{start_date.strftime("%Y-%m-%d")} {end_date.strftime("%Y-%m-%d")}'
    pytrend.build_payload([keyword], cat=0, timeframe=timeframe, geo='', gprop='')
    trend_data = pytrend.interest_over_time()
    if 'isPartial' in trend_data.columns:
        trend_data = trend_data.drop('isPartial', axis=1)
    trend_data.reset_index(inplace=True)
    trend_data.rename(columns={keyword: 'Trend', 'date': 'Date'}, inplace=True)
    return trend_data[['Date', 'Trend']]

def get_sentiment_data(start_date, end_date):
    """
    Simulates daily sentiment scores for Bitcoin using VADER.
    """
    dates = pd.date_range(start_date, end_date, freq='D')
    analyzer = SentimentIntensityAnalyzer()
    headlines = [
        "Bitcoin soars as institutional interest grows",
        "Bitcoin falls amid market uncertainty",
        "Investors optimistic about Bitcoin's future",
        "Bitcoin struggles due to regulatory concerns",
        "Bitcoin shows mixed signals in volatile market"
    ]
    sentiment_scores = []
    for _ in dates:
        selected = random.sample(headlines, k=3)
        scores = [analyzer.polarity_scores(headline)['compound'] for headline in selected]
        sentiment_scores.append(np.mean(scores))
    return pd.DataFrame({'Date': dates, 'Sentiment': sentiment_scores})

# ---------------------------
# Data Merging & Preprocessing.
# ---------------------------

def merge_crypto_data(data_frames):
    """
    Merges multiple cryptocurrency data tables on the 'Date' column.
    """
    from functools import reduce
    merged_df = reduce(lambda left, right: pd.merge(left, right, on='Date', how='outer'), data_frames)
    if isinstance(merged_df.columns, pd.MultiIndex):
        merged_df.columns = merged_df.columns.get_level_values(0)
    merged_df.sort_values('Date', inplace=True)
    merged_df.fillna(method='ffill', inplace=True)
    merged_df.fillna(method='bfill', inplace=True)
    return merged_df

def merge_external_factors(crypto_df, trends_df, sentiment_df):
    """
    Merges external factors (Google Trends and Sentiment) with the cryptocurrency data.
    """
    df = pd.merge(crypto_df, trends_df, on="Date", how="outer")
    df = pd.merge(df, sentiment_df, on="Date", how="outer")
    df.sort_values("Date", inplace=True)
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)
    return df

def create_sequences(data, seq_length):
    """
    Creates sequences from the scaled data for training the model.
    Each sequence uses 'seq_length' days to predict the next day's target (first column).
    Returns arrays X (sequences) and y (targets).
    """
    X, y = [], []
    if len(data) < seq_length:
        return np.array(X), np.array(y)
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def invert_scale(scaler, data, col_index=0):
    """
    Reverses the scaling transformation on a single feature column.
    """
    full = np.zeros((len(data), scaler.n_features_in_))
    full[:, col_index] = data
    inv = scaler.inverse_transform(full)[:, col_index]
    return inv

# ---------------------------
# Hypermodel for Bayesian Optimization
# ---------------------------

def build_model(hp):
    """
    Constructs a hypermodel with hyperparameters for the number of units,
    dropout rates, and learning rate.
    """
    model = Sequential()
    # Bidirectional LSTM layer
    model.add(Bidirectional(LSTM(
        units=hp.Int('units1', min_value=32, max_value=128, step=16),
        return_sequences=True
    ), input_shape=input_shape))
    model.add(Dropout(rate=hp.Float('dropout1', min_value=0.1, max_value=0.5, step=0.1)))
    
    # Second LSTM layer
    model.add(LSTM(
        units=hp.Int('units2', min_value=32, max_value=128, step=16),
        return_sequences=True))
    model.add(Dropout(rate=hp.Float('dropout2', min_value=0.1, max_value=0.5, step=0.1)))
    
    # Third LSTM layer with the fixed step value.
    model.add(LSTM(
        units=hp.Int('units3', min_value=32, max_value=128, step=16)))
    model.add(Dropout(rate=hp.Float('dropout3', min_value=0.1, max_value=0.5, step=0.1)))
    
    # Output layer
    model.add(Dense(1))
    
    # Learning rate parameter
    lr = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='mean_squared_error')
    return model


# ---------------------------
# Main Pipeline with Bayesian Optimization.
# ---------------------------

def main():
    # Increase the data range: using data from January 1, 2019 to February 24, 2025.
    start_date = pd.to_datetime("2019-01-01").date()
    end_date = pd.to_datetime("2025-02-24").date()
    
    # Mapping crypto symbols to their Yahoo Finance ticker codes.
    ticker_map = {
        'BTC': 'BTC-USD',
        'ETH': 'ETH-USD',
        'XRP': 'XRP-USD',
        'ADA': 'ADA-USD'
    }
    crypto_list = ['BTC', 'ETH', 'XRP', 'ADA']
    
    # Load data for each crypto from Yahoo Finance.
    crypto_dfs = []
    for crypto in crypto_list:
        ticker = ticker_map[crypto]
        try:
            df = load_crypto_data_from_yahoo(ticker, crypto, start_date, end_date)
            crypto_dfs.append(df)
            print(f"Loaded data for {crypto} from Yahoo Finance")
        except Exception as e:
            print(f"Error loading {crypto}: {e}")
    
    # Merge the crypto data.
    merged_crypto_df = merge_crypto_data(crypto_dfs)
    print("Merged Crypto Data Head:")
    print(merged_crypto_df.head())

    # Get external data.
    trends_df = get_google_trends_data(start_date, end_date, keyword='bitcoin')
    sentiment_df = get_sentiment_data(start_date, end_date)
    
    # Merge external factors with crypto data.
    merged_df = merge_external_factors(merged_crypto_df, trends_df, sentiment_df)
    print("Merged Data with External Factors Head:")
    print(merged_df.head())

    # Define sequence length.
    seq_length = 30  # Using the past 30 days to predict the next day's price.
    
    # Loop over each cryptocurrency.
    for crypto in crypto_list:
        print(f"\nProcessing {crypto}...")
        target_col = f"{crypto}_Close"
        if target_col not in merged_df.columns:
            print(f"Column {target_col} not found. Skipping {crypto}.")
            continue
        
        # Prepare features: closing price, Trend, Sentiment, and a 10-day SMA.
        df_features = merged_df[[target_col, 'Trend', 'Sentiment']].copy()
        df_features['SMA_10'] = df_features[target_col].rolling(window=10, min_periods=1).mean()
        
        # Scale features with a feature range of (-1, 1) to preserve dynamic range.
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = scaler.fit_transform(df_features)
        
        # Create sequences.
        X, y = create_sequences(scaled_data, seq_length)
        if len(X) == 0:
            print(f"Not enough data to create sequences for {crypto}.")
            continue
        
        # Split into training (80%) and testing (20%) sets.
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Set global input_shape for the hypermodel.
        global input_shape
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        # Define callbacks to be used during tuning.
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
        ]
        
        # Initialize Bayesian Optimization tuner.
        tuner = kt.BayesianOptimization(
            build_model,
            objective='val_loss',
            max_trials=5,
            executions_per_trial=1,
            directory=f"tuner_logs/{crypto}",
            project_name=f"{crypto}_crypto_lstm"
        )
        
        # Search for the best hyperparameters.
        tuner.search(X_train, y_train,
                     epochs=50,
                     batch_size=32,
                     validation_split=0.1,
                     callbacks=callbacks,
                     verbose=1)
        
        # Retrieve the best model.
        best_model = tuner.get_best_models(num_models=1)[0]
        best_model.summary()
        
        # Evaluate on the test set.
        y_pred = best_model.predict(X_test)
        y_test_inv = invert_scale(scaler, y_test, col_index=0)
        y_pred_inv = invert_scale(scaler, y_pred.flatten(), col_index=0)
        
        mape = mean_absolute_percentage_error(y_test_inv, y_pred_inv)
        rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
        r2 = r2_score(y_test_inv, y_pred_inv)
        print(f"{crypto} - Test MAPE: {mape * 100:.2f}%")
        print(f"{crypto} - Test RMSE: {rmse:.2f}")
        print(f"{crypto} - Test R-squared: {r2:.2f}")
        
        # Plot predictions along with evaluation metrics.
        plt.figure(figsize=(12, 6))
        plt.plot(y_test_inv, label=f"True {crypto} Price")
        plt.plot(y_pred_inv, label=f"Predicted {crypto} Price")
        plt.title(f"{crypto} Price Prediction using Bayesian Optimized LSTM")
        plt.xlabel("Time Steps")
        plt.ylabel(f"{crypto} Price (USD)")
        plt.legend()
        plt.text(0.05, 0.95, f"MAPE: {mape * 100:.2f}%\nRMSE: {rmse:.2f}\nR²: {r2:.2f}",
                 transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        plt.show()

if __name__ == "__main__":
    main()
