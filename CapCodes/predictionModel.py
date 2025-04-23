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
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import random


# Data Collection Functions.
def load_bitcoin_data_from_csv(file_path):
    """
    Load Bitcoin price data from a local CSV file.
    Parameters:
        file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: A DataFrame containing the 'Date' and 'BTC_Close' columns.
    """
    df = pd.read_csv(file_path)
    
    # Convert the 'Date' column to datetime format.
    df['Date'] = pd.to_datetime(df['Date'])

    # Ensure the 'Close' column exists and is correctly formatted.
    if 'Price' in df.columns:
        df.rename(columns={'Price': 'BTC_Close'}, inplace=True)

    # Remove commas if present in the price column and convert to numeric.
    df['BTC_Close'] = df['BTC_Close'].replace({',': ''}, regex=True)
    df['BTC_Close'] = pd.to_numeric(df['BTC_Close'], errors='coerce')

    return df[['Date', 'BTC_Close']]

def get_google_trends_data(start_date, end_date, keyword='bitcoin'):
    """Fetch Google Trends data for a given keyword using pytrends."""
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
    Simulate daily Bitcoin sentiment scores using VADER.
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
        avg_score = np.mean(scores)
        sentiment_scores.append(avg_score)
    df_sentiment = pd.DataFrame({'Date': dates, 'Sentiment': sentiment_scores})
    return df_sentiment


# Data Merging & Preprocessing.
def merge_data(btc, trends, sentiment):
    """Merge Bitcoin, Google Trends, and Sentiment data on the Date column."""
    
    # Convert 'Date' columns to date objects for consistent merging
    btc['Date'] = pd.to_datetime(btc['Date']).dt.date
    trends['Date'] = pd.to_datetime(trends['Date']).dt.date
    sentiment['Date'] = pd.to_datetime(sentiment['Date']).dt.date
    
    df = pd.merge(btc, trends, on="Date", how="outer")
    df = pd.merge(df, sentiment, on="Date", how="outer")
    df.sort_values(by="Date", inplace=True)
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)
    return df

def create_sequences(data, seq_length):
    """Create sequences of features for LSTM training."""
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i, 0])  # BTC_Close as target
    return np.array(X), np.array(y)


# LSTM Model & Training
def build_lstm_model(input_shape):
    """Build a multi-layer LSTM model using Keras."""
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def invert_scale(scaler, data, col_index=0):
    """Inverse transform a single feature column."""
    full = np.zeros((len(data), scaler.n_features_in_))
    full[:, col_index] = data
    inv = scaler.inverse_transform(full)[:, col_index]
    return inv


# Main Function
def main():
    # Define the time period (e.g., last 2 years)
    start_date = pd.to_datetime("2023-02-24").date()  # Modify to match available CSV data
    end_date = pd.to_datetime("2025-02-24").date()

    # Load Bitcoin data from CSV
    btc_data = load_bitcoin_data_from_csv(r"C:\Users\Sahil\OneDrive\Desktop\Capstone Project\Bitcoin Historical Data.csv")

    # Fetch other datasets
    trends_data = get_google_trends_data(start_date, end_date, keyword='bitcoin')
    sentiment_data = get_sentiment_data(start_date, end_date)

    # Merge the datasets
    merged_df = merge_data(btc_data, trends_data, sentiment_data)
    print("Merged Data Head:")
    print(merged_df.head())

    # Use BTC_Close as target; features: Trend and Sentiment.
    df_features = merged_df[['BTC_Close', 'Trend', 'Sentiment']]

    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_features)

    # Create sequences with a lookback window (e.g., 30 days)
    seq_length = 30
    X, y = create_sequences(scaled_data, seq_length)

    # Train-test split (80% training, 20% testing)
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build and summarize the model
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.summary()

    # Train the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Inverse scaling for the target (BTC_Close is at index 0)
    y_test_inv = invert_scale(scaler, y_test, col_index=0)
    y_pred_inv = invert_scale(scaler, y_pred.flatten(), col_index=0)

    # Evaluation metrics
    mape = mean_absolute_percentage_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))

    print(f"Test MAPE: {mape * 100:.2f}%")
    print(f"Test RMSE: {rmse:.2f}")

    # Plot true vs predicted Bitcoin prices
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_inv, label='True BTC Price')
    plt.plot(y_pred_inv, label='Predicted BTC Price')
    plt.title("Bitcoin Price Prediction using LSTM")
    plt.xlabel("Time Steps")
    plt.ylabel("BTC Price (USD)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
