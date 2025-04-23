# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:45:50 2025

@author: Sahil
"""

import pandas as pd
print(pd.__version__)
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
import os

# ---------------------------
# Data Collection Functions
# ---------------------------

def load_crypto_data_from_csv(file_path, crypto_name):
    """
    This function reads a CSV file that contains cryptocurrency data.
    It converts the date to a proper format and cleans up the price column.
    
    Parameters:
        file_path (str): The path where the CSV file is located.
        crypto_name (str): A short name for the crypto (e.g., 'BTC', 'ETH', 'XRP', 'ADA').
        
    Returns:
        A DataFrame with two columns: 'Date' and '<crypto_name>_Close'.
    """
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])  # Make sure the Date is recognized as a date.
    
    # Rename the price column to 'BTC_Close' (depending on the crypto).
    if 'Price' in df.columns:
        df.rename(columns={'Price': f'{crypto_name}_Close'}, inplace=True)
    else:
        df.rename(columns={'Close': f'{crypto_name}_Close'}, inplace=True)
    
    # Remove any commas in the price.
    df[f'{crypto_name}_Close'] = df[f'{crypto_name}_Close'].replace({',': ''}, regex=True)
    df[f'{crypto_name}_Close'] = pd.to_numeric(df[f'{crypto_name}_Close'], errors='coerce')
    return df[['Date', f'{crypto_name}_Close']]

def get_google_trends_data(start_date, end_date, keyword='bitcoin'):
    """
    This function gets Google Trends data for a given keyword.
    It returns a table with dates and a value called 'Trend'.
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
    This function simulates a daily sentiment score for Bitcoin using VADER.
    It picks random headlines for each day, calculates an average sentiment score, and returns a table.
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

# ---------------------------
# Data Merging & Preprocessing
# ---------------------------

def merge_crypto_data(data_frames):
    """
    This function merges multiple cryptocurrency data tables on the 'Date' column.
    It sorts the dates and fills in missing values.
    
    Parameters:
        data_frames (list): A list of DataFrames for different cryptocurrencies.
        
    Returns:
        A single merged DataFrame.
    """
    from functools import reduce
    merged_df = reduce(lambda left, right: pd.merge(left, right, on='Date', how='outer'), data_frames)
    merged_df.sort_values('Date', inplace=True)
    merged_df.fillna(method='ffill', inplace=True)
    merged_df.fillna(method='bfill', inplace=True)
    return merged_df

def merge_external_factors(crypto_df, trends_df, sentiment_df):
    """
    This function adds external data (Google Trends and Sentiment) to the cryptocurrency data.
    It merges them on the 'Date' column.
    """
    df = pd.merge(crypto_df, trends_df, on="Date", how="outer")
    df = pd.merge(df, sentiment_df, on="Date", how="outer")
    df.sort_values("Date", inplace=True)
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)
    return df

def create_sequences(data, seq_length):
    """
    Create sequences from the scaled data for training the model.
    Each sequence is a window of past days, and the target is the next day's value.
    
    Parameters:
        data (np.array): The scaled data.
        seq_length (int): How many past days to use for each sequence.
        
    Returns:
        Two arrays: X (the sequences) and y (the target values).
    """
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i, 0])  # We take the first column as the target.
    return np.array(X), np.array(y)

# ---------------------------
# LSTM Model & Training.
# ---------------------------

def build_lstm_model(input_shape):
    """
    Build an LSTM model with several layers to predict future prices.
    
    Parameters:
        input_shape (tuple): The shape of the input data (sequence length, number of features).
        
    Returns:
        A compiled LSTM model.
    """
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
    """
    Reverse the scaling transformation on a single feature column.
    
    Parameters:
        scaler: The MinMaxScaler used earlier.
        data (array): The scaled data.
        col_index (int): The index of the target column.
        
    Returns:
        The data transformed back to the original scale.
    """
    full = np.zeros((len(data), scaler.n_features_in_))
    full[:, col_index] = data
    inv = scaler.inverse_transform(full)[:, col_index]
    return inv

# ---------------------------
# Main Pipeline.
# ---------------------------

def main():
    # Define the date range for external factors.
    start_date = pd.to_datetime("2023-02-24").date()
    end_date = pd.to_datetime("2025-02-24").date()

    # Folder path where the CSV files are stored.
    base_path = r"~/Desktop/CapCodes"
    # Mapping crypto symbols to their CSV file names.
    file_map = {
        'BTC': 'Bitcoin Historical Data.csv',
        'ETH': 'Ethereum Historical Data.csv',
        'XRP': 'XRP Historical Data.csv',
        'ADA': 'Cardano Historical Data.csv'
    }
    crypto_list = ['BTC', 'ETH', 'XRP', 'ADA']
    
    # Load data for each crypto.
    crypto_dfs = []
    for crypto in crypto_list:
        file_path = os.path.join(base_path, file_map[crypto])
        try:
            df = load_crypto_data_from_csv(file_path, crypto)
            crypto_dfs.append(df)
            print(f"Loaded data for {crypto}")
        except Exception as e:
            print(f"Error loading {crypto}: {e}")
    
    # Merge the crypto data based on the 'Date' column.
    merged_crypto_df = merge_crypto_data(crypto_dfs)
    print("Merged Crypto Data Head:")
    print(merged_crypto_df.head())

    # Get Google Trends and sentiment data.
    trends_df = get_google_trends_data(start_date, end_date, keyword='bitcoin')
    sentiment_df = get_sentiment_data(start_date, end_date)
    
    # Merge the external data with the crypto data.
    merged_df = merge_external_factors(merged_crypto_df, trends_df, sentiment_df)
    print("Merged Data with External Factors Head:")
    print(merged_df.head())

    #Loop over each cryptocurrency to train a model.
    seq_length = 30  # We'll look at the past 30 days to predict the next day's price.
    
    for crypto in crypto_list:
        print(f"\nProcessing {crypto}...")
        target_col = f"{crypto}_Close"
        
        if target_col not in merged_df.columns:
            print(f"Column {target_col} not found. Skipping {crypto}.")
            continue
        
        # Use the crypto's closing price along with the Trend and Sentiment as features.
        df_features = merged_df[[target_col, 'Trend', 'Sentiment']].copy()
        
        # Scale the features so that they range between 0 and 1.
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df_features)
        
        # Create sequences of 30 days.
        X, y = create_sequences(scaled_data, seq_length)
        
        # Split data into training (80%) and testing (20%) sets.
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Build and show a summary of the LSTM model.
        model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
        model.summary()
        
        # Train the model.
        history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)
        
        # Make predictions on the test data.
        y_pred = model.predict(X_test)
        
        # Convert the predictions back to the original scale.
        y_test_inv = invert_scale(scaler, y_test, col_index=0)
        y_pred_inv = invert_scale(scaler, y_pred.flatten(), col_index=0)
        
        # Calculate how far off our predictions are.
        mape = mean_absolute_percentage_error(y_test_inv, y_pred_inv)
        rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
        print(f"{crypto} - Test MAPE: {mape * 100:.2f}%")
        print(f"{crypto} - Test RMSE: {rmse:.2f}")
        
        # Drawing a plot comparing actual prices to predicted prices.
        plt.figure(figsize=(12, 6))
        plt.plot(y_test_inv, label=f"True {crypto} Price")
        plt.plot(y_pred_inv, label=f"Predicted {crypto} Price")
        plt.title(f"{crypto} Price Prediction using LSTM")
        plt.xlabel("Time Steps")
        plt.ylabel(f"{crypto} Price (USD)")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()
