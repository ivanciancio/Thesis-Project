import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def prepare_market_data(market_data):
    """Prepare market data for analysis"""
    if market_data is None or market_data.empty:
        return None
        
    df = market_data.copy()
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=5).std()
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    return df

def calculate_correlations(market_data, news_data, reddit_data):
    """Calculate correlations between different data sources"""
    if market_data is None or news_data is None:
        return None
        
    # Prepare daily sentiment scores
    news_sentiment = news_data.groupby(news_data['Date'].dt.date)['Sentiment Score'].mean()
    
    # Prepare market data
    market_returns = market_data.set_index('Date')['Returns']
    
    # Combine data
    combined_data = pd.DataFrame({
        'Market_Returns': market_returns,
        'News_Sentiment': news_sentiment
    })
    
    if reddit_data is not None and not reddit_data.empty:
        reddit_sentiment = reddit_data.groupby(reddit_data['Date'].dt.date)['Sentiment Score'].mean()
        combined_data['Reddit_Sentiment'] = reddit_sentiment
    
    return combined_data.corr()

def aggregate_sentiment_data(news_data, reddit_data):
    """Aggregate sentiment data from different sources"""
    aggregated_data = []
    
    if news_data is not None and not news_data.empty:
        news_daily = news_data.groupby(news_data['Date'].dt.date).agg({
            'Sentiment Score': 'mean',
            'Source': lambda x: 'News'
        }).reset_index()
        aggregated_data.append(news_daily)
    
    if reddit_data is not None and not reddit_data.empty:
        reddit_daily = reddit_data.groupby(reddit_data['Date'].dt.date).agg({
            'Sentiment Score': 'mean',
            'Source': lambda x: 'Reddit'
        }).reset_index()
        aggregated_data.append(reddit_daily)
    
    if aggregated_data:
        return pd.concat(aggregated_data, ignore_index=True)
    return None