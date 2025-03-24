# -*- coding: utf-8 -*-
"""
Rolling Correlation & Exploratory Data Analysis for Cryptocurrencies
@Author: Sahil
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates  # <-- For date formatting
from pytrends.request import TrendReq
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
import os

# ---------------------------
# Data Loading & Processing
# ---------------------------
def load_crypto_data_from_csv(file_path, crypto_name):
    """Load and clean cryptocurrency data from CSV files."""
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
    df.dropna(subset=['Date'], inplace=True)
    df.sort_values(by='Date', inplace=True)
    df.rename(columns={'Price': f'{crypto_name}_Close'}, inplace=True)
    df[f'{crypto_name}_Close'] = pd.to_numeric(
        df[f'{crypto_name}_Close'].replace({',': ''}, regex=True), errors='coerce'
    )
    return df[['Date', f'{crypto_name}_Close']]

# Folder paths, file mapping, etc.
base_path = r"C:\Users\Sahil\OneDrive\Desktop\Capstone Project"
file_map = {
    'BTC': 'Bitcoin Historical Data.csv',
    'ETH': 'Ethereum Historical Data.csv',
    'XRP': 'XRP Historical Data.csv',
    'ADA': 'Cardano Historical Data.csv'
}

# Load & merge crypto data
crypto_dfs = []
for crypto, file_name in file_map.items():
    file_path = os.path.join(base_path, file_name)
    df = load_crypto_data_from_csv(file_path, crypto)
    crypto_dfs.append(df)

merged_crypto_df = crypto_dfs[0]
for df in crypto_dfs[1:]:
    merged_crypto_df = pd.merge(merged_crypto_df, df, on="Date", how="outer")

# Download & merge S&P 500 data from yfinance
sp500 = yf.download('^GSPC', start='2022-01-01', end='2023-12-31')
sp500.reset_index(inplace=True)
if isinstance(sp500.columns, pd.MultiIndex):
    sp500.columns = sp500.columns.get_level_values(0)
sp500.rename(columns={'Close': 'SP500'}, inplace=True)
sp500 = sp500[['Date', 'SP500']]

def get_google_trends_data(start_date, end_date, keyword='bitcoin'):
    """Fetch Google Trends data for Bitcoin."""
    pytrend = TrendReq(hl='en-US', tz=360)
    timeframe = f'{start_date.strftime("%Y-%m-%d")} {end_date.strftime("%Y-%m-%d")}'
    pytrend.build_payload([keyword], cat=0, timeframe=timeframe, geo='', gprop='')
    trend_data = pytrend.interest_over_time().reset_index()
    trend_data.rename(columns={keyword: 'Google_Trends', 'date': 'Date'}, inplace=True)
    return trend_data[['Date', 'Google_Trends']]

def get_sentiment_data(df):
    """Generate Bitcoin sentiment scores using VADER."""
    analyzer = SentimentIntensityAnalyzer()
    df['BTC_Sentiment'] = df['BTC_Close'].astype(str).apply(
        lambda x: analyzer.polarity_scores(x)['compound']
    )
    return df[['Date', 'BTC_Sentiment']]

start_date, end_date = pd.to_datetime("2022-01-01"), pd.to_datetime("2023-12-31")
google_trends_df = get_google_trends_data(start_date, end_date)

# Merge external data
merged_df = pd.merge(merged_crypto_df, google_trends_df, on='Date', how='left')
merged_df = pd.merge(merged_df, sp500, on='Date', how='left')
merged_df.fillna(method='ffill', inplace=True)
merged_df.fillna(method='bfill', inplace=True)

sentiment_df = get_sentiment_data(merged_df)
merged_df = pd.merge(merged_df, sentiment_df, on="Date", how="left")

# ----------------------------------------
# Set Date as Index & Rolling Correlation
# ----------------------------------------
merged_df.set_index('Date', inplace=True)

def calculate_rolling_correlation(df, window=30):
    df["Corr_BTC_ETH"] = df["BTC_Close"].rolling(window=window).corr(df["ETH_Close"])
    df["Corr_BTC_XRP"] = df["BTC_Close"].rolling(window=window).corr(df["XRP_Close"])
    df["Corr_BTC_ADA"] = df["BTC_Close"].rolling(window=window).corr(df["ADA_Close"])
    return df

merged_df = calculate_rolling_correlation(merged_df, window=30)

def plot_rolling_correlation(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["Corr_BTC_ETH"], label="BTC-ETH Correlation", color="blue", alpha=0.7)
    plt.plot(df.index, df["Corr_BTC_XRP"], label="BTC-XRP Correlation", color="red", alpha=0.7)
    plt.plot(df.index, df["Corr_BTC_ADA"], label="BTC-ADA Correlation", color="green", alpha=0.7)
    plt.axhline(0, color="black", linestyle="--", alpha=0.6)
    plt.title("Rolling 30-day Correlation Between Bitcoin & Altcoins")
    plt.xlabel("Date")
    plt.ylabel("Correlation")

    # Format x-axis to show Year Ticks (e.g., 2021, 2022)
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Tick every year
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.legend()
    plt.show()

# Plot Rolling Correlation
plot_rolling_correlation(merged_df)
