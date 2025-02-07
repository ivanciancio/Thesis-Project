# pages/3_correlation_analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
#from utils.visualization import plot_correlation_matrix

def interpret_correlation_strength(correlation):
    """Interpret the strength of a correlation coefficient"""
    abs_corr = abs(correlation)
    if np.isnan(abs_corr):
        return "insufficient data"
    elif abs_corr < 0.2:
        return "very weak"
    elif abs_corr < 0.4:
        return "weak"
    elif abs_corr < 0.6:
        return "moderate"
    elif abs_corr < 0.8:
        return "strong"
    else:
        return "very strong"

def interpret_correlation_direction(correlation):
    """Interpret the direction of a correlation coefficient"""
    if np.isnan(correlation):
        return ""
    return "positive" if correlation > 0 else "negative"

def align_dates_and_compute_returns(market_data, news_data, reddit_data=None):
    """
    Align dates between market data and sentiment data with better date handling
    """
    try:
        # Convert market data dates to date (without time)
        market_data = market_data.copy()
        market_data['Date'] = pd.to_datetime(market_data['Date']).dt.normalize()
        
        # Calculate returns
        market_data['Returns'] = market_data['Close'].pct_change()
        
        # Prepare news sentiment
        news_data = news_data.copy()
        news_data['Date'] = pd.to_datetime(news_data['Date']).dt.normalize()
        
        # Create daily aggregates
        daily_returns = market_data.groupby('Date')['Returns'].mean()
        daily_news = news_data.groupby('Date')['Sentiment Score'].mean()
        
        # Align the data on dates
        combined_data = pd.DataFrame({
            'Market_Returns': daily_returns,
            'News_Sentiment': daily_news
        })
        
        # Add Reddit data if available
        if reddit_data is not None and not reddit_data.empty:
            reddit_data = reddit_data.copy()
            reddit_data['Date'] = pd.to_datetime(reddit_data['Date']).dt.normalize()
            daily_reddit = reddit_data.groupby('Date')['Sentiment Score'].mean()
            combined_data['Reddit_Sentiment'] = daily_reddit
        
        # Remove any NaN values
        combined_data = combined_data.fillna(method='ffill').fillna(method='bfill')
        
        return combined_data
        
    except Exception as e:
        st.error(f"Error aligning dates: {str(e)}")
        return pd.DataFrame()

def plot_correlation_matrix(correlation_data):
    """Create correlation matrix visualization"""
    if correlation_data.empty:
        return None
        
    fig = go.Figure(data=go.Heatmap(
        z=correlation_data.values,
        x=correlation_data.columns,
        y=correlation_data.index,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        text=np.round(correlation_data.values, 3),
        texttemplate='%{text}',
        textfont={"size": 14},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Correlation Matrix",
        height=500,
        xaxis_title="",
        yaxis_title="",
        yaxis_autorange='reversed'
    )
    
    return fig

def display_correlation_summary(correlations):
    """Display summary of correlations with interpretation"""
    st.subheader("Correlation Summary")
    
    if 'News_Sentiment' in correlations.columns:
        news_corr = correlations.loc['Market_Returns', 'News_Sentiment']
        st.write("**Market Returns vs News Sentiment:**", f"{news_corr:.3f}")
        
        strength = interpret_correlation_strength(news_corr)
        direction = interpret_correlation_direction(news_corr)
        if strength != "insufficient data":
            st.write(f"There is a {strength} {direction} correlation between Market Returns and News Sentiment.")
        else:
            st.write("Insufficient data to determine correlation between Market Returns and News Sentiment.")
    
    if 'Reddit_Sentiment' in correlations.columns:
        reddit_corr = correlations.loc['Market_Returns', 'Reddit_Sentiment']
        st.write("\n**Market Returns vs Reddit Sentiment:**", f"{reddit_corr:.3f}")
        
        strength = interpret_correlation_strength(reddit_corr)
        direction = interpret_correlation_direction(reddit_corr)
        if strength != "insufficient data":
            st.write(f"There is a {strength} {direction} correlation between Market Returns and Reddit Sentiment.")
        else:
            st.write("Insufficient data to determine correlation between Market Returns and Reddit Sentiment.")

def correlation_analysis_page():
    st.title("🔗 Correlation Analysis")
    
    if not all(key in st.session_state for key in ['market_data', 'news_data']):
        st.warning("Please complete market data and sentiment analysis first")
        return
    
    try:
        # Get data from session state
        market_data = st.session_state.market_data
        news_data = st.session_state.news_data
        reddit_data = st.session_state.get('reddit_data')
        
        # Display data ranges
        st.subheader("Data Ranges")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"Market Data: From: {market_data['Date'].min().strftime('%Y-%m-%d')} To: {market_data['Date'].max().strftime('%Y-%m-%d')}")
        with col2:
            st.write(f"News Data: From: {news_data['Date'].min().strftime('%Y-%m-%d')} To: {news_data['Date'].max().strftime('%Y-%m-%d')}")
        if reddit_data is not None and not reddit_data.empty:
            with col3:
                st.write(f"Reddit Data: From: {reddit_data['Date'].min().strftime('%Y-%m-%d')} To: {reddit_data['Date'].max().strftime('%Y-%m-%d')}")
        
        # Align dates and compute correlations
        combined_data = align_dates_and_compute_returns(market_data, news_data, reddit_data)
        
        if combined_data.empty:
            st.warning("No overlapping data found between market returns and sentiment scores")
            return
            
        st.write(f"Number of aligned data points: {len(combined_data)}")
        
        if len(combined_data) < 2:
            st.warning("At least 2 data points are needed for correlation analysis")
            return
        
        # Calculate correlations
        correlations = combined_data.corr()
        
        # Display correlation matrix
        st.subheader("Correlation Matrix")
        fig = plot_correlation_matrix(correlations)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Display correlation summary
        st.subheader("Correlation Summary")
        
        for col1 in correlations.columns:
            for col2 in correlations.columns:
                if col1 < col2:  # Only show each pair once
                    corr_value = correlations.loc[col1, col2]
                    st.write(f"**{col1.replace('_', ' ')} vs {col2.replace('_', ' ')}:** {corr_value:.3f}")
                    
                    strength = abs(corr_value)
                    if strength < 0.3:
                        relationship = "weak"
                    elif strength < 0.7:
                        relationship = "moderate"
                    else:
                        relationship = "strong"
                        
                    direction = "positive" if corr_value > 0 else "negative"
                    st.write(f"There is a {relationship} {direction} correlation.")
                    st.write("")
        
        # Show the aligned data
        if st.button("View Aligned Data"):
            st.dataframe(combined_data)
        
        # Add download buttons
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Download Correlation Matrix",
                correlations.to_csv(),
                "correlation_matrix.csv",
                "text/csv",
                key='download-correlations'
            )
        with col2:
            st.download_button(
                "Download Aligned Data",
                combined_data.to_csv(),
                "aligned_data.csv",
                "text/csv",
                key='download-aligned-data'
            )
        
    except Exception as e:
        st.error(f"Error in correlation analysis: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    correlation_analysis_page()