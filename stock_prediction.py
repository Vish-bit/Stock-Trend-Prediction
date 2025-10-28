import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from alpha_vantage.timeseries import TimeSeries
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os 

# --- Alpha Vantage API ---
api_key = os.getenv("API_KEY")
ts = TimeSeries(key=api_key, output_format="pandas")

print("Fetching stock data for AAPL...")
data, meta_data = ts.get_daily(symbol="AAPL", outputsize="full")
print("Data fetched successfully!")

# --- News API ---
news_url = "https://newsapi.org/v2/everything?q=Apple&apiKey=c888069b671f4ca0bb7c58deca582c42"
response = requests.get(news_url)
news_data = response.json()

articles = [article["title"] for article in news_data["articles"][:10]]
print("\nSample News Headlines:")
for headline in articles:
    print("•", headline)

# --- Sentiment Analysis ---
analyzer = SentimentIntensityAnalyzer()

def get_sentiment_score(text):
    return analyzer.polarity_scores(text)["compound"]

sentiment_scores = [get_sentiment_score(article) for article in articles]
print("\nSentiment Scores:", sentiment_scores)

# --- Stock Price Prediction ---
data["moving_avg"] = data["4. close"].rolling(window=20).mean()
data.dropna(inplace=True)

X = data[["moving_avg"]]
y = data["4. close"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# --- Visualization ---
plt.figure(figsize=(12, 4))
plt.plot(y_test.values, label="Actual Price", color="blue")
plt.plot(predictions, label="Predicted Price", color="orange")
plt.title("AAPL Stock Price Prediction (Random Forest)")
plt.xlabel("Samples")
plt.ylabel("Stock Price (USD)")
plt.legend()
plt.tight_layout()
plt.show()
