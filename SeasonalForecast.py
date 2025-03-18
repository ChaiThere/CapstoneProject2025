import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
import os

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
    df['Date'] = pd.to_datetime(df['Date'])  # Ensure the Date is recognized as a date.
    
    if 'Price' in df.columns:
        df.rename(columns={'Price': f'{crypto_name}_Close'}, inplace=True)
    else:
        df.rename(columns={'Close': f'{crypto_name}_Close'}, inplace=True)
    
    df[f'{crypto_name}_Close'] = df[f'{crypto_name}_Close'].replace({',': ''}, regex=True)
    df[f'{crypto_name}_Close'] = pd.to_numeric(df[f'{crypto_name}_Close'], errors='coerce')
    return df[['Date', f'{crypto_name}_Close']]

crypto_dfs = []  # Initialize an empty list

# Define the base path where CSV files are stored
base_path = os.path.expanduser("~/Desktop/CapCodes")  # Ensure correct path resolution

# Define a mapping of cryptocurrency symbols to CSV filenames
file_map = {
    'BTC': 'Bitcoin Historical Data.csv',
    'ETH': 'Ethereum Historical Data.csv',
    'XRP': 'XRP Historical Data.csv',
    'ADA': 'Cardano Historical Data.csv'
}

# Load data for each cryptocurrency
for crypto, filename in file_map.items():
    file_path = os.path.join(base_path, filename)
    print(f"Checking file: {file_path}")  # Debugging line
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue  # Skip this file if it does not exist
    
    try:
        df = load_crypto_data_from_csv(file_path, crypto)
        if df.empty:
            print(f"Warning: {crypto} dataset is empty.")
        else:
            print(f"Loaded {crypto} data with {len(df)} rows.")
            crypto_dfs.append(df)
    except Exception as e:
        print(f"Error loading {crypto}: {e}")

# Now, `crypto_dfs` should contain DataFrames
if not crypto_dfs:
    print("No cryptocurrency data loaded. Check file paths and ensure CSVs exist.")
    exit()

def merge_crypto_data(data_frames):
    """
    Merges multiple cryptocurrency data tables on the 'Date' column.
    Sorts dates and fills missing values.
    
    Parameters:
        data_frames (list): A list of DataFrames for different cryptocurrencies.
        
    Returns:
        A single merged DataFrame.
    """
    merged_df = reduce(lambda left, right: pd.merge(left, right, on='Date', how='outer'), data_frames)
    merged_df.sort_values('Date', inplace=True)
    merged_df.fillna(method='ffill', inplace=True)
    merged_df.fillna(method='bfill', inplace=True)
    return merged_df

def plot_correlation_graph(merged_crypto_df, window=30):
    """
    Plots the rolling correlation between Bitcoin and altcoins to identify inverse relationships.
    Also generates a heatmap showing overall correlation and highlights significant inverse correlations.

    Parameters:
        merged_crypto_df (pd.DataFrame): The merged cryptocurrency data.
        window (int): Rolling window size for correlation calculation.
    """
    # Calculate rolling correlation with Bitcoin
    crypto_list = [col.replace('_Close', '') for col in merged_crypto_df.columns if col.endswith('_Close')]
    rolling_corrs = {}
    
    for crypto in crypto_list:
        if crypto != 'BTC':  # Compare altcoins with BTC
            rolling_corrs[crypto] = merged_crypto_df['BTC_Close'].rolling(window).corr(merged_crypto_df[f'{crypto}_Close'])
    
    # Plot Rolling Correlation
    plt.figure(figsize=(12, 6))
    for crypto, corr_series in rolling_corrs.items():
        plt.plot(corr_series, label=f'BTC vs {crypto}')
        
        # Highlight significant inverse correlations
        significant_points = corr_series[corr_series < -0.3]
        plt.scatter(significant_points.index, significant_points.values, color='red', label=f'Significant {crypto}', zorder=3)
    
    plt.axhline(0, color='black', linestyle='--', alpha=0.6)
    plt.title(f'Rolling {window}-Day Correlation Between Bitcoin and Altcoins')
    plt.xlabel('Date')
    plt.ylabel('Correlation')
    plt.legend()
    plt.show()
    
    # Identify and highlight significant inverse correlation periods
    inverse_corrs = {}
    for crypto, corr_series in rolling_corrs.items():
        inverse_corrs[crypto] = corr_series[corr_series < -0.3]  # Threshold for significant inverse correlation
    
    if any(len(v.dropna()) > 0 for v in inverse_corrs.values()):
        print("Significant inverse correlations detected:")
        for crypto, corr_series in inverse_corrs.items():
            if len(corr_series.dropna()) > 0:
                print(f"{crypto}: Inverse correlation periods detected.")
    else:
        print("No significant inverse correlation periods detected.")
    
    # Compute full correlation matrix
    correlation_matrix = merged_crypto_df[[f'{crypto}_Close' for crypto in crypto_list]].corr()
    
    # Plot heatmap of correlation
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
    plt.title('Cryptocurrency Correlation Matrix')
    plt.show()

# Call this function after merging the crypto data
df_crypto = merge_crypto_data(crypto_dfs)
plot_correlation_graph(df_crypto, window=30)

