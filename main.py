import warnings
import os

# Suppress PyTorch warnings
warnings.filterwarnings('ignore', message='.*torch.classes.*')
warnings.filterwarnings('ignore', message='.*Trying to instantiate.*')
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Get the absolute path of the project root directory and add it to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Import required functions from utils package
from utils import (
    fetch_market_sentiment,
    fetch_historical_prices,
    plot_market_data,
    prepare_market_data
)

  
def init_session_state():
    """Initialise session state variables"""
    if 'market_data' not in st.session_state:
        st.session_state.market_data = None
    if 'news_data' not in st.session_state:
        st.session_state.news_data = None
    if 'reddit_data' not in st.session_state:
        st.session_state.reddit_data = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None

def calculate_market_metrics(market_data):
    """Calculate market metrics"""
    metrics = {
        'price_change': (market_data['Close'].iloc[-1] - market_data['Close'].iloc[0]) / market_data['Close'].iloc[0] * 100,
        'avg_volume': market_data['Volume'].mean(),
        'volatility': market_data['Returns'].std() * 100 if 'Returns' in market_data.columns else None,
        'latest_price': market_data['Close'].iloc[-1]
    }
    return metrics

def display_market_metrics(metrics):
    """Display market metrics in Streamlit"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Price Change", f"{metrics['price_change']:.2f}%")
    with col2:
        st.metric("Average Volume", f"{metrics['avg_volume']:,.0f}")
    with col3:
        st.metric("Volatility", f"{metrics['volatility']:.2f}%" if metrics['volatility'] else "N/A")
    with col4:
        st.metric("Latest Price", f"${metrics['latest_price']:.2f}")

# Page configuration
st.set_page_config(
    page_title="Financial Market Sentiment Analysis",
    page_icon="📊",
    layout="wide"
)

# Initialise session state
init_session_state()

# Main title
st.title("Financial Market Sentiment Analysis")

# Market Data Analysis section
st.header("📈 Market Data Analysis")

# Input parameters
symbol = st.text_input("Enter Stock Symbol", "AAPL")
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
with col2:
    end_date = st.date_input("End Date", value=datetime.now())

if st.button("Fetch Market Data"):
    with st.spinner("Fetching market data..."):
        # Fetch market data
        market_data = fetch_historical_prices(symbol, start_date, end_date)
        market_sentiment = fetch_market_sentiment(symbol)
        
        if market_data is not None and not market_data.empty:
            # Prepare market data
            processed_data = prepare_market_data(market_data)
            st.session_state.market_data = processed_data
            st.session_state.symbol = symbol
            st.success("Market data fetched successfully!")
            
            # Display market data visualisation
            fig = plot_market_data(processed_data, market_sentiment)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display summary statistics
            st.subheader("Market Summary")
            metrics = calculate_market_metrics(processed_data)
            display_market_metrics(metrics)
            
            # Save market sentiment
            st.session_state.market_sentiment = market_sentiment
        else:
            st.error("Failed to fetch market data. Please check the symbol and try again.")
            