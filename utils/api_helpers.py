import requests
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import sys
import json  # Make sure this import is here
from utils.error_handler import handle_api_error

@handle_api_error
def fetch_market_sentiment(symbol):
    """
    Fetch market sentiment data from EODHD API using API key from secrets
    """
    api_key = st.secrets["eodhd_api_key"]
    base_url = f"https://eodhd.com/api/fundamentals/{symbol}.US"
    params = {
        'api_token': api_key,
        'fmt': 'json'
    }
    
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        if isinstance(data, dict):
            sentiment_score = data.get('AnalystRatings', {}).get('Rating', 3)
            
            sentiment_data = {
                'Date': pd.to_datetime(datetime.now().strftime('%Y-%m-%d')),
                'Sentiment': sentiment_score,
                'Normalised_Sentiment': (sentiment_score - 1) / 4
            }
            
            return pd.DataFrame([sentiment_data])
    
    return pd.DataFrame(columns=['Date', 'Sentiment', 'Normalised_Sentiment'])

@handle_api_error
def fetch_historical_prices(symbol, start_date, end_date):
    """
    Fetch historical price data from EODHD API using API key from secrets
    """
    api_key = st.secrets["eodhd_api_key"]
    base_url = f"https://eodhd.com/api/eod/{symbol}.US"
    params = {
        'api_token': api_key,
        'from': start_date.strftime('%Y-%m-%d'),
        'to': end_date.strftime('%Y-%m-%d'),
        'period': 'd',
        'fmt': 'json'
    }
    
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        if isinstance(data, list):
            df = pd.DataFrame(data)
            if not df.empty:
                # Standardise column names
                df.rename(columns={
                    'date': 'Date',
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                }, inplace=True)
                
                df['Date'] = pd.to_datetime(df['Date'])
                return df.sort_values('Date')
    
    return pd.DataFrame()

@handle_api_error
def fetch_news_data(symbol, start_date=None, end_date=None):
    """Fetch news data from EODHD API using API key from secrets"""
    api_key = st.secrets["eodhd_api_key"]
    base_url = "https://eodhd.com/api/news"
    
    # Convert dates to UTC format for API request
    if start_date:
        start_date = pd.to_datetime(start_date).tz_localize('UTC')
    if end_date:
        end_date = pd.to_datetime(end_date).tz_localize('UTC')
    
    params = {
        'api_token': api_key,
        's': symbol,
        'limit': 1000,
        'offset': 0
    }
    
    # Only add date parameters if they are provided
    if start_date:
        params['from'] = start_date.strftime('%Y-%m-%d')
    if end_date:
        params['to'] = end_date.strftime('%Y-%m-%d')
    
    # No need for try/except here - @handle_api_error already handles that
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        data = json.loads(response.text)
        
        if isinstance(data, list):
            news_data = []
            for item in data:
                # Basic data extraction
                date_str = item.get('date')
                if not date_str:
                    continue  # Skip items without dates
                    
                news_date = pd.to_datetime(date_str)
                # Handle timezone properly - check if already tz-aware
                if news_date.tzinfo is None:
                    news_date = news_date.tz_localize('UTC')
                    
                news_item = {
                    'Date': news_date,
                    'Title': item.get('title', ''),
                    'Text': item.get('text', ''),
                    'Source': item.get('source', ''),
                    'URL': item.get('link', '')
                }
                news_data.append(news_item)
            
            df = pd.DataFrame(news_data)
            if not df.empty:
                # Proper handling of timezone conversion
                df['Date'] = df['Date'].apply(
                    lambda dt: dt.tz_convert(None) if dt.tzinfo is not None else dt
                )
                df = df.sort_values('Date', ascending=True)
                return df
        
        st.warning(f"No news data found for {symbol} in the market data period")
    else:
        st.error(f"Error fetching news data: {response.status_code}")
    
    return pd.DataFrame()