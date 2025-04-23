import pandas as pd
import numpy as np
import random
from functools import reduce
from pytrends.request import TrendReq
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def load_crypto_data_from_csv(file_path, crypto_name):
    import pandas as pd
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])  # Ensure Date format

    if 'Price' in df.columns:
        df.rename(columns={'Price': f'{crypto_name}_Close'}, inplace=True)
    else:
        df.rename(columns={'Close': f'{crypto_name}_Close'}, inplace=True)

    df[f'{crypto_name}_Close'] = df[f'{crypto_name}_Close'].replace({',': ''}, regex=True)
    df[f'{crypto_name}_Close'] = pd.to_numeric(df[f'{crypto_name}_Close'], errors='coerce')
    return df[['Date', f'{crypto_name}_Close']]

def merge_crypto_data(data_frames):
    from functools import reduce
    merged_df = reduce(lambda left, right: pd.merge(left, right, on='Date', how='outer'), data_frames)
    merged_df.sort_values('Date', inplace=True)
    merged_df.fillna(method='ffill', inplace=True)
    merged_df.fillna(method='bfill', inplace=True)
    return merged_df

def get_google_trends_data(start_date, end_date, keyword='bitcoin'):
    from pytrends.request import TrendReq
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
    import pandas as pd
    import numpy as np
    import random
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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

def merge_external_factors(crypto_df, trends_df, sentiment_df):
    df = pd.merge(crypto_df, trends_df, on="Date", how="outer")
    df = pd.merge(df, sentiment_df, on="Date", how="outer")
    df.sort_values("Date", inplace=True)
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)
    return df
