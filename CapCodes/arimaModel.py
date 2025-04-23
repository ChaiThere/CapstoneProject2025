# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 10:00:00 2025

@author: Kit Chai
"""

import pandas as pd
import numpy as np
from pytrends.request import TrendReq
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import random
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ---------------------------
# Data Collection Functions (assumed to be imported)
# ---------------------------
from data_utils import (
    load_crypto_data_from_csv,
    merge_crypto_data,
    get_google_trends_data,
    get_sentiment_data,
    merge_external_factors
)

# ---------------------------
# Main Pipeline
# ---------------------------

def add_features(df, target_col):
    df['log_return'] = np.log(df[target_col]).diff()
    df['rolling_mean_7'] = df[target_col].rolling(window=7).mean()
    df['rolling_std_7'] = df[target_col].rolling(window=7).std()
    df['rolling_mean_30'] = df[target_col].rolling(window=30).mean()
    df['rolling_std_30'] = df[target_col].rolling(window=30).std()
    return df.dropna()

def main():
    start_date = pd.to_datetime("2023-02-24").date()
    end_date = pd.to_datetime("2025-02-24").date()

    base_path = r"~/Desktop/CapCodes"
    file_map = {
        'BTC': 'Bitcoin Historical Data.csv',
        'ETH': 'Ethereum Historical Data.csv',
        'XRP': 'XRP Historical Data.csv',
        'ADA': 'Cardano Historical Data.csv'
    }
    crypto_list = ['BTC', 'ETH', 'XRP', 'ADA']

    crypto_dfs = []
    for crypto in crypto_list:
        file_path = os.path.expanduser(os.path.join(base_path, file_map[crypto]))
        try:
            df = load_crypto_data_from_csv(file_path, crypto)
            crypto_dfs.append(df)
            print(f"Loaded data for {crypto}")
        except Exception as e:
            print(f"Error loading {crypto}: {e}")

    merged_crypto_df = merge_crypto_data(crypto_dfs)
    print("Merged Crypto Data Head:")
    print(merged_crypto_df.head())

    trends_df = get_google_trends_data(start_date, end_date, keyword='bitcoin')
    sentiment_df = get_sentiment_data(start_date, end_date)

    merged_df = merge_external_factors(merged_crypto_df, trends_df, sentiment_df)
    print("Merged Data with External Factors Head:")
    print(merged_df.head())

    for crypto in crypto_list:
        print(f"\nProcessing {crypto}...")
        target_col = f"{crypto}_Close"

        if target_col not in merged_df.columns:
            print(f"Column {target_col} not found. Skipping {crypto}.")
            continue

        df = merged_df[['Date', target_col, 'Trend', 'Sentiment']].dropna()
        df.set_index('Date', inplace=True)

        # Add engineered features
        df = add_features(df, target_col)

        # Define y and exogenous variables
        y = df[target_col]
        exog = df[['Trend', 'Sentiment', 'log_return', 'rolling_mean_7', 'rolling_std_7', 'rolling_mean_30', 'rolling_std_30']]

        # Scale exogenous variables
        scaler = StandardScaler()
        exog_scaled = pd.DataFrame(scaler.fit_transform(exog), index=exog.index, columns=exog.columns)

        # Train/Test split
        train_size = int(len(df) * 0.8)
        train_y = y[:train_size]
        test_y = y[train_size:]
        train_exog = exog_scaled.iloc[:train_size]
        test_exog = exog_scaled.iloc[train_size:]

        # Fit ARIMAX model
        print(f"Running ARIMAX(2,1,2) for {crypto} with engineered features...")
        model = SARIMAX(train_y, exog=train_exog, order=(2,1,2))
        model_fit = model.fit(disp=False)
        forecast = model_fit.predict(start=len(train_y), end=len(df)-1, exog=test_exog)

        # Evaluation
        mae = mean_absolute_error(test_y, forecast)
        mape = mean_absolute_percentage_error(test_y, forecast)
        rmse = np.sqrt(mean_squared_error(test_y, forecast))
        r2 = r2_score(test_y, forecast)

        print(f"{crypto} - Test MAE: {mae:.2f}")
        print(f"{crypto} - Test MAPE: {mape * 100:.2f}%")
        print(f"{crypto} - Test RMSE: {rmse:.2f}")
        print(f"{crypto} - R^2 Score: {r2:.4f}")

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(test_y.index, test_y.values, label='True')
        plt.plot(test_y.index, forecast, label='Forecast')
        plt.title(f"{crypto} Price Forecast using ARIMAX with Features")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()