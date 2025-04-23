import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
import os
import matplotlib.dates as mdates

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

def merge_crypto_data(data_frames):
    merged_df = reduce(lambda left, right: pd.merge(left, right, on='Date', how='outer'), data_frames)
    merged_df.sort_values('Date', inplace=True)
    merged_df.fillna(method='ffill', inplace=True)
    merged_df.fillna(method='bfill', inplace=True)
    return merged_df

def export_poster_graphs(df_crypto, output_dir="poster_graphs"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    crypto_list = [col.replace('_Close', '') for col in df_crypto.columns if col.endswith('_Close')]
    intervals = {"1 Month (30 days)": 30, "3 Months (90 days)": 90, "6 Months (180 days)": 180}

    for label, window in intervals.items():
        plt.figure(figsize=(14, 7))
        for crypto in crypto_list:
            if crypto != 'BTC':
                rolling_corr = df_crypto['BTC_Close'].rolling(window).corr(df_crypto[f'{crypto}_Close'])
                plt.plot(df_crypto['Date'], rolling_corr, label=f'BTC vs {crypto} ({label})')
                significant_points = rolling_corr[rolling_corr < -0.3]
                plt.scatter(df_crypto['Date'].iloc[significant_points.index], significant_points.values,
                            color='red', marker='o', zorder=3, label=f'Inverse points: BTC vs {crypto}')

        plt.axhline(0, color='black', linestyle='--', alpha=0.6)
        plt.title(f'Rolling Correlation Between BTC and Altcoins - {label}')
        plt.xlabel('Date')
        plt.ylabel('Correlation')
        plt.legend(loc='lower right', fontsize='small')
        plt.grid(True)
        plt.tight_layout()
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.xticks(rotation=45)

        filename = os.path.join(output_dir, f"rolling_correlation_{window}d.png")
        plt.savefig(filename, dpi=300)
        plt.close()

    # Correlation matrix heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_crypto[[f'{crypto}_Close' for crypto in crypto_list]].corr(), annot=True,
                cmap='coolwarm', linewidths=0.5, fmt='.2f')
    plt.title('Cryptocurrency Correlation Matrix')
    plt.tight_layout()
    heatmap_filename = os.path.join(output_dir, "correlation_matrix_heatmap.png")
    plt.savefig(heatmap_filename, dpi=300)
    plt.close()

# === Load data ===
base_path = os.path.expanduser("~/Desktop/CapCodes")
file_map = {
    'BTC': 'Bitcoin Historical Data.csv',
    'ETH': 'Ethereum Historical Data.csv',
    'XRP': 'XRP Historical Data.csv',
    'ADA': 'Cardano Historical Data.csv'
}

crypto_dfs = []
for crypto, filename in file_map.items():
    file_path = os.path.join(base_path, filename)
    if os.path.exists(file_path):
        df = load_crypto_data_from_csv(file_path, crypto)
        crypto_dfs.append(df)
    else:
        print(f"File not found: {file_path}")

if crypto_dfs:
    df_crypto = merge_crypto_data(crypto_dfs)
    export_poster_graphs(df_crypto)
else:
    print("No data loaded. Please check file paths.")
