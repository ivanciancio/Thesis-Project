import requests
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    """
    Fetch news data from EODHD API using API key from secrets
    """
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
        'offset': 0,
        'from': start_date.strftime('%Y-%m-%d') if start_date else None,
        'to': end_date.strftime('%Y-%m-%d') if end_date else None
    }
    
    try:
        response = requests.get(base_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list):
                news_data = []
                for item in data:
                    try:
                        # Handle already tz-aware timestamps properly
                        news_date = pd.to_datetime(item.get('date'))
                        if news_date.tzinfo is not None:
                            # Convert to UTC if it's tz-aware
                            news_date = news_date.tz_convert('UTC')
                        else:
                            # Localise to UTC if it's tz-naive
                            news_date = news_date.tz_localise('UTC')
                            
                        # Compare dates after converting to UTC
                        if (not start_date or news_date >= start_date) and \
                           (not end_date or news_date <= end_date):
                            news_item = {
                                'Date': news_date,
                                'Title': item.get('title', ''),
                                'Text': item.get('text', ''),
                                'Source': item.get('source', ''),
                                'URL': item.get('link', '')
                            }
                            news_data.append(news_item)
                    except Exception as e:
                        st.write(f"Error processing item: {str(e)}")
                        continue
                
                df = pd.DataFrame(news_data)
                if not df.empty:
                    # Convert all dates to naive timestamps after filtering
                    df['Date'] = df['Date'].dt.tz_convert(None)
                    df = df.sort_values('Date', ascending=True)
                    
                    st.write(f"Successfully processed {len(df)} news items")
                    return df
                
            return pd.DataFrame()
        else:
            st.error(f"Error fetching news data: {response.status_code}")
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Error in news request: {str(e)}")
        return pd.DataFrame()