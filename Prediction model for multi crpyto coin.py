# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:45:50 2025

@author: Sahil
"""

import pandas as pd
import numpy as np
from pytrends.request import TrendReq
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
import random
import os

# ---------------------------
# Data Collection Functions
# ---------------------------

def load_crypto_data_from_csv(file_path, crypto_name):
    """
    Reads a CSV file containing cryptocurrency data, converts the 'Date' column to datetime,
    cleans up the price column, and renames it to '<crypto_name>_Close'.
    """
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    if 'Price' in df.columns:
        df.rename(columns={'Price': f'{crypto_name}_Close'}, inplace=True)
    else:
        df.rename(columns={'Close': f'{crypto_name}_Close'}, inplace=True)
    
    df[f'{crypto_name}_Close'] = df[f'{crypto_name}_Close'].replace({',': ''}, regex=True)
    df[f'{crypto_name}_Close'] = pd.to_numeric(df[f'{crypto_name}_Close'], errors='coerce')
    return df[['Date', f'{crypto_name}_Close']]

def get_google_trends_data(start_date, end_date, keyword='bitcoin'):
    """
    Gets Google Trends data for a given keyword and returns a DataFrame with 'Date' and 'Trend'.
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
    Simulates a daily sentiment score for Bitcoin using VADER.
    Returns a DataFrame with 'Date' and 'Sentiment'.
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
# Data Merging & Preprocessing
# ---------------------------

def merge_crypto_data(data_frames):
    """
    Merges multiple cryptocurrency DataFrames on 'Date', sorts the data,
    and fills missing values using forward and backward fill.
    """
    from functools import reduce
    merged_df = reduce(lambda left, right: pd.merge(left, right, on='Date', how='outer'), data_frames)
    merged_df.sort_values('Date', inplace=True)
    merged_df.fillna(method='ffill', inplace=True)
    merged_df.fillna(method='bfill', inplace=True)
    return merged_df

def merge_external_factors(crypto_df, trends_df, sentiment_df):
    """
    Merges external factors (Google Trends and Sentiment) with the cryptocurrency data on 'Date'.
    """
    df = pd.merge(crypto_df, trends_df, on="Date", how="outer")
    df = pd.merge(df, sentiment_df, on="Date", how="outer")
    df.sort_values("Date", inplace=True)
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)
    return df

def create_sequences(data, seq_length):
    """
    Creates sequences from the scaled data.
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

# ---------------------------
# LSTM Model & Training.
# ---------------------------

def build_lstm_model(input_shape):
    """
    Builds an LSTM model with increased complexity:
      - The first layer is a Bidirectional LSTM with 64 units.
      - Followed by two LSTM layers with 64 units each.
      - Uses dropout of 0.3 in each layer.
      - Uses a lower learning rate to avoid training saturation.
    Returns the compiled model.
    """
    model = Sequential()
    
    # Increased complexity with Bidirectional LSTM.
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(64))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    
    # Lower learning rate for better convergence.
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
                  loss='mean_squared_error')
    return model

def invert_scale(scaler, data, col_index=0):
    """
    Reverses the scaling transformation for a single feature column.
    """
    full = np.zeros((len(data), scaler.n_features_in_))
    full[:, col_index] = data
    return scaler.inverse_transform(full)[:, col_index]

# ---------------------------
# Main Pipeline.
# ---------------------------

def main():
    # Define the date range.
    start_date = pd.to_datetime("2019-01-01").date()
    end_date = pd.to_datetime("2024-12-31").date()

    # Folder path for CSV files.
    base_path = r"C:\Users\Sahil\OneDrive\Desktop\Capstone Project"
    file_map = {
        'BTC': 'Bitcoin Historical Data.csv',
        'ETH': 'Ethereum Historical Data.csv',
        'XRP': 'XRP Historical Data.csv',
        'ADA': 'Cardano Historical Data.csv'
    }
    crypto_list = ['BTC', 'ETH', 'XRP', 'ADA']
    
    # Load data for each cryptocurrency.
    crypto_dfs = []
    for crypto in crypto_list:
        file_path = os.path.join(base_path, file_map[crypto])
        try:
            df = load_crypto_data_from_csv(file_path, crypto)
            crypto_dfs.append(df)
            print(f"Loaded data for {crypto}")
        except Exception as e:
            print(f"Error loading {crypto}: {e}")
    
    # Merge cryptocurrency data.
    merged_crypto_df = merge_crypto_data(crypto_dfs)
    print("Merged Crypto Data Head:")
    print(merged_crypto_df.head())

    # Get external factors.
    trends_df = get_google_trends_data(start_date, end_date, keyword='bitcoin')
    sentiment_df = get_sentiment_data(start_date, end_date)
    
    # Merge external factors with crypto data.
    merged_df = merge_external_factors(merged_crypto_df, trends_df, sentiment_df)
    print("Merged Data with External Factors Head:")
    print(merged_df.head())

    # Define sequence length.
    seq_length = 30  # Using the past 30 days to predict the next day's price.
    
    # Process each cryptocurrency.
    for crypto in crypto_list:
        print(f"\nProcessing {crypto}...")
        target_col = f"{crypto}_Close"
        if target_col not in merged_df.columns:
            print(f"Column {target_col} not found. Skipping {crypto}.")
            continue
        
        # Prepare features: closing price, Trend, and Sentiment.
        df_features = merged_df[[target_col, 'Trend', 'Sentiment']].copy()
        
        # Use MinMaxScaler with a feature range of (-1, 1) to preserve dynamic range.
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
        
        # Build and summarize the model.
        model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
        model.summary()
        
        # Train the model.
        history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)
        
        # Make predictions on the test data.
        y_pred = model.predict(X_test)
        
        # Convert predictions back to the original scale.
        y_test_inv = invert_scale(scaler, y_test, col_index=0)
        y_pred_inv = invert_scale(scaler, y_pred.flatten(), col_index=0)
        
        # Calculate evaluation metrics.
        mape = mean_absolute_percentage_error(y_test_inv, y_pred_inv)
        rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
        r2 = r2_score(y_test_inv, y_pred_inv)
        print(f"{crypto} - Test MAPE: {mape * 100:.2f}%")
        print(f"{crypto} - Test RMSE: {rmse:.2f}")
        print(f"{crypto} - Test R-squared: {r2:.2f}")
        
        # Plot true vs. predicted prices with evaluation metrics annotated.
        plt.figure(figsize=(12, 6))
        plt.plot(y_test_inv, label=f"True {crypto} Price")
        plt.plot(y_pred_inv, label=f"Predicted {crypto} Price")
        plt.title(f"{crypto} Price Prediction using LSTM")
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

