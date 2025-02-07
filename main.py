import streamlit as st
import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def init_session_state():
    """Initialize session state variables"""
    if 'market_data' not in st.session_state:
        st.session_state.market_data = None
    if 'news_data' not in st.session_state:
        st.session_state.news_data = None
    if 'reddit_data' not in st.session_state:
        st.session_state.reddit_data = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None

def main():
    st.set_page_config(
        page_title="Financial Market Sentiment Analysis",
        page_icon="📊",
        layout="wide"
    )
    
    init_session_state()
    
    st.title("Financial Market Sentiment Analysis")
    st.sidebar.title("Configuration")
    
    # API Configuration
    with st.sidebar.expander("API Configuration"):
        st.session_state.eodhd_api_key = st.text_input("EODHD API Key", type="password")
        st.session_state.reddit_client_id = st.text_input("Reddit Client ID", type="password")
        st.session_state.reddit_client_secret = st.text_input("Reddit Client Secret", type="password")

if __name__ == "__main__":
    main()