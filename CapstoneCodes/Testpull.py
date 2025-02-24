import requests
import pandas as pd

def fetch_market_data(coin, currency="usd", days="365"):
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
    
    params = {
        "vs_currency": currency,
        "days": days,
        "interval": "daily"
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        print(f"Error fetching data for {coin}: {response.status_code}")
        print(response.text)
        return pd.DataFrame()

    data = response.json()
    
    if not all(key in data for key in ["prices", "market_caps", "total_volumes"]):
        print(f"Unexpected API response format for {coin}:")
        print(data)
        return pd.DataFrame()

    df = pd.DataFrame({
        "timestamp": [entry[0] for entry in data["prices"]],
        f"{coin}_open": [entry[1] for entry in data["prices"]],
        f"{coin}_high": [entry[1] for entry in data["prices"]],  # Placeholder, adjust if API supports
        f"{coin}_low": [entry[1] for entry in data["prices"]],   # Placeholder, adjust if API supports
        f"{coin}_close": [entry[1] for entry in data["prices"]],
        f"{coin}_volume": [entry[1] for entry in data["total_volumes"]],
        f"{coin}_market_cap": [entry[1] for entry in data["market_caps"]],
        f"{coin}_trade_count": [entry[1] for entry in data.get("trade_count", [[0, 0]] * len(data["prices"]))]  # Placeholder if trade count is supported
    })
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

# List of coins to fetch
tokens = ["bitcoin", "ethereum", "ripple", "cardano"]

# Fetch and save data
for token in tokens:
    data = fetch_market_data(token)
    if not data.empty:
        print(f"{token.capitalize()} data fetched")
        data.to_csv(f"{token}_market_data.csv", index=False)
        print(f"{token.capitalize()} data saved as CSV")
