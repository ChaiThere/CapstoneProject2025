import requests
import pandas as pd

def fetch_market_cap(coin, currency="usd", days="30"):
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
    
    # Ensure parameters are correctly passed
    params = {
        "vs_currency": currency,  # Required parameter
        "days": days,  # Number of days for historical data
        "interval": "daily"  # Ensures we get daily data
    }
    
    response = requests.get(url, params=params)
    
    # Check if the response is valid
    if response.status_code != 200:
        print(f"Error fetching data for {coin}: {response.status_code}")
        print(response.text)  # Print API error message
        return pd.DataFrame()  # Return an empty DataFrame

    data = response.json()
    
    # Check if 'prices' key exists
    if "prices" not in data:
        print(f"Unexpected API response format for {coin}:")
        print(data)  # Print the full API response for debugging
        return pd.DataFrame()

    # Convert to DataFrame
    prices = pd.DataFrame(data["prices"], columns=["timestamp", f"{coin}_price"])
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], unit="ms")  # Convert to readable date
    return prices

# Test the function
btc_data = fetch_market_cap("bitcoin")
eth_data = fetch_market_cap("ethereum")

if not btc_data.empty:
    print("Bitcoin data fetched")
if not eth_data.empty:
    print("Ethereum data fetched")

btc_data.to_csv("bitcoin_market_data.csv", index=False)
eth_data.to_csv("ethereum_market_data.csv", index=False)
print("Data saved as CSV")
