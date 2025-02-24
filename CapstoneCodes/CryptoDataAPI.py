# -*- coding: utf-8 -*-
"""
Created on Mondaay Feb 24 15:08:53 2025

@author: Sahil
"""


#Importing necessary libraries.
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
import logging
import argparse
import os
import io

# Setting up logging configuration.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_crypto_data(coin: str, currency: str = "usd", days: str = "365") -> pd.DataFrame:
    """
    Fetch historical market data for a given cryptocurrency from CoinGecko.
    Parameters:
        coin (str): CoinGecko coin id (e.g., "bitcoin", "ethereum", "ripple", "cardano")
        currency (str): Target fiat currency (default "usd")
        days (str): Number of days of historical data (or "max")
    Returns:
        pd.DataFrame: DataFrame with columns 'Date' and 'Price_<coin>'
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
    params = {"vs_currency": currency, "days": days, "interval": "daily"}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        logging.error(f"Error fetching data for {coin}: {e}")
        return pd.DataFrame()
    
    data = response.json()
    if "prices" not in data:
        logging.error(f"API response for {coin} does not contain 'prices'")
        return pd.DataFrame()
    
    df = pd.DataFrame(data["prices"], columns=["timestamp", f"Price_{coin}"])
    df["Date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.drop("timestamp", axis=1, inplace=True)
    return df[["Date", f"Price_{coin}"]]

def merge_crypto_data(coins: list, currency: str, days: str) -> pd.DataFrame:
    """
    Fetch and merge historical data for multiple cryptocurrencies.
    Parameters:
        coins (list): List of coin ids to fetch.
        currency (str): Target fiat currency.
        days (str): Number of days of historical data.
    Returns:
        pd.DataFrame: Merged DataFrame containing Date and price columns for each coin.
    """
    df_list = []
    for coin in coins:
        df = fetch_crypto_data(coin, currency, days)
        if not df.empty:
            if "Date" not in df.columns:
                logging.warning(f"{coin} dataframe missing 'Date' column. Resetting index.")
                df = df.reset_index()
            df_list.append(df)
            logging.info(f"Fetched data for {coin}")
        else:
            logging.warning(f"No data for {coin}")
    
    if not df_list:
        raise ValueError("No data fetched for any coin.")
    
    merged_df = reduce(lambda left, right: pd.merge(left, right, on="Date", how="outer"), df_list)
    merged_df.sort_values(by="Date", inplace=True)
    merged_df.fillna(method="ffill", inplace=True)
    merged_df.fillna(method="bfill", inplace=True)
    
    if "Date" not in merged_df.columns:
        merged_df = merged_df.reset_index()
    
    return merged_df

def plot_time_series(merged_df: pd.DataFrame, coins: list, save_plots: bool, output_dir: str):
    plt.figure(figsize=(14, 8))
    for coin in coins:
        plt.plot(merged_df["Date"], merged_df[f"Price_{coin}"], label=f"{coin.capitalize()} Price")
    plt.title("Cryptocurrency Prices Over Time")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    if save_plots:
        filepath = os.path.join(output_dir, "time_series.png")
        plt.savefig(filepath)
        logging.info(f"Time series plot saved to {filepath}")
    plt.show()

def plot_correlation_heatmap(merged_df: pd.DataFrame, save_plots: bool, output_dir: str):
    plt.figure(figsize=(10, 8))
    sns.heatmap(merged_df.drop("Date", axis=1).corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of Crypto Prices")
    if save_plots:
        filepath = os.path.join(output_dir, "correlation_heatmap.png")
        plt.savefig(filepath)
        logging.info(f"Correlation heatmap saved to {filepath}")
    plt.show()

def plot_pairplot(merged_df: pd.DataFrame, save_plots: bool, output_dir: str):
    pairplot = sns.pairplot(merged_df.drop("Date", axis=1))
    pairplot.fig.suptitle("Pairplot of Cryptocurrency Prices", y=1.02)
    if save_plots:
        filepath = os.path.join(output_dir, "pairplot.png")
        pairplot.savefig(filepath)
        logging.info(f"Pairplot saved to {filepath}")
    plt.show()

def plot_histograms(merged_df: pd.DataFrame, coins: list, save_plots: bool, output_dir: str):
    for coin in coins:
        plt.figure(figsize=(10, 6))
        sns.histplot(merged_df[f"Price_{coin}"], bins=30, kde=True)
        plt.title(f"Distribution of {coin.capitalize()} Price")
        plt.xlabel("Price (USD)")
        plt.ylabel("Frequency")
        if save_plots:
            filename = f"histogram_{coin}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath)
            logging.info(f"Histogram for {coin} saved to {filepath}")
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Cryptocurrency EDA with CoinGecko Data")
    parser.add_argument("--coins", nargs="+", default=["bitcoin", "ethereum", "ripple", "cardano"],
                        help="List of cryptocurrencies to analyze (default: bitcoin ethereum ripple cardano)")
    parser.add_argument("--currency", type=str, default="usd", help="Target fiat currency (default: usd)")
    parser.add_argument("--days", type=str, default="365", help="Number of days of historical data (or 'max')")
    parser.add_argument("--save_plots", action="store_true", help="Save plots to disk")
    parser.add_argument("--output_dir", type=str, default="plots", help="Directory to save plots")
    
    args = parser.parse_args()
    
    # Create output directory if required.
    if args.save_plots:
        os.makedirs(args.output_dir, exist_ok=True)
    
    sns.set_style("whitegrid")
    
    # Fetching and merging data.
    merged_df = merge_crypto_data(args.coins, args.currency, args.days)
    
    # Log basic data overview by capturing DataFrame info into a string.
    buffer = io.StringIO()
    merged_df.info(buf=buffer)
    info_str = buffer.getvalue()
    logging.info("Merged DataFrame Info:\n" + info_str)
    logging.info("Merged DataFrame Descriptive Statistics:\n" + merged_df.describe().to_string())
    
    # Generating Plots.
    plot_time_series(merged_df, args.coins, args.save_plots, args.output_dir)
    plot_correlation_heatmap(merged_df, args.save_plots, args.output_dir)
    plot_pairplot(merged_df, args.save_plots, args.output_dir)
    plot_histograms(merged_df, args.coins, args.save_plots, args.output_dir)

if __name__ == "__main__":
    main()
