import streamlit as st
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import re
import pandas as pd

# Function to get stock details from Ape Wisdom website
def get_stock_details(ticker_symbol):
    url = f'https://apewisdom.io/stocks/{ticker_symbol}/'
    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, 'html.parser')
    details_div = soup.find('div', class_='row', style='font-size: 18px; margin-top: 20px; margin-bottom: 10px;')

    if details_div:
        details_text = details_div.get_text(separator="\n").strip()
    else:
        details_text = f"Details div not found for {ticker_symbol}."

    match = re.search(r'keywords: (.*)', details_text)
    if match:
        keywords = match.group(1).split(",")
        keywords = [keyword.strip().replace('"', '') for keyword in keywords]
    else:
        keywords = []

    stock_data = yf.Ticker(ticker_symbol)
    try:
        hist = stock_data.history(period="1d")
        if not hist.empty:
            current_price = hist['Close'].iloc[-1]
        else:
            current_price = 'Price not found'
    except Exception as e:
        current_price = f'Error fetching price: {e}'

    positive_percentage = None
    match = re.search(r'(\d+)% positive comments', details_text)
    if match:
        positive_percentage = int(match.group(1))

    return details_text, current_price, positive_percentage, keywords

# Function to fetch details for each ticker
def fetch_details(ticker):
    details, price, positive_pct, keywords = get_stock_details(ticker)
    return pd.Series({
        'current_price': price,
        'positive_sentiment_pct': positive_pct,
        'sentiment_details': details,
        'keywords': keywords
    })

# Streamlit app
st.title("Stock Data from Ape Wisdom")

url = 'https://apewisdom.io/api/v1.0/filter/all-stocks'

try:
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    # st.write("State of Wisdom API:" + response.status_code())
    # st.write("Response JSON:")
    # st.json(data)
    
    df = pd.DataFrame(data['results'])
    df[['current_price', 'positive_sentiment_pct', 'sentiment_details', 'keywords']] = df['ticker'].apply(fetch_details)
    
    st.write("Stock Data:")
    st.dataframe(df,use_container_width=True)

except requests.exceptions.RequestException as e:
    st.error(f"An error occurred: {e}")
