# pages/3_correlation_analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats 
#from utils.visualisation import plot_correlation_matrix

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
    Align dates between market data and sentiment data with improved handling
    """
    try:
        # Create proper date indices
        market_data = market_data.copy()
        news_data = news_data.copy()
        
        # Convert all dates to datetime and normalise to remove time component
        market_data['Date'] = pd.to_datetime(market_data['Date']).dt.normalize()
        news_data['Date'] = pd.to_datetime(news_data['Date']).dt.normalize()
        
        # Calculate returns using closing prices
        market_data['Returns'] = market_data['Close'].pct_change()
        
        # Check for sentiment column names
        sentiment_column = 'Sentiment Score' if 'Sentiment Score' in news_data.columns else 'Sentiment_Score'
        reddit_sentiment_column = 'Sentiment_Score' if reddit_data is not None else None
        
        # Calculate daily sentiment averages for news
        daily_news = (news_data
            .groupby('Date')[sentiment_column]
            .agg(['mean', 'count'])
            .reset_index())
        daily_news.columns = ['Date', 'News_Sentiment', 'News_Count']
        
        # Create initial merged dataframe
        merged_data = pd.merge(
            market_data[['Date', 'Returns', 'Close']],
            daily_news,
            on='Date',
            how='outer'
        )
        
        # Add Reddit data if available
        if reddit_data is not None and not reddit_data.empty:
            reddit_data = reddit_data.copy()
            reddit_data['Date'] = pd.to_datetime(reddit_data['Date']).dt.normalize()
            
            daily_reddit = (reddit_data
                .groupby('Date')[reddit_sentiment_column]
                .agg(['mean', 'count'])
                .reset_index())
            daily_reddit.columns = ['Date', 'Reddit_Sentiment', 'Reddit_Count']
            
            merged_data = pd.merge(
                merged_data,
                daily_reddit,
                on='Date',
                how='outer'
            )
        
        # Sort by date
        merged_data = merged_data.sort_values('Date')
        
        # Calculate rolling metrics
        window = 3  # 3-day rolling window
        merged_data['Rolling_Returns'] = merged_data['Returns'].rolling(window).mean()
        merged_data['Rolling_News_Sentiment'] = merged_data['News_Sentiment'].rolling(window).mean()
        if 'Reddit_Sentiment' in merged_data.columns:
            merged_data['Rolling_Reddit_Sentiment'] = merged_data['Reddit_Sentiment'].rolling(window).mean()
        
        # Filter to overlapping date range
        merged_data = merged_data.dropna(subset=['Returns', 'News_Sentiment'])
        
        if len(merged_data) == 0:
            st.warning("No overlapping data found between market returns and sentiment scores")
            return pd.DataFrame()
            
        st.success(f"Number of aligned data points: {len(merged_data)}")
        return merged_data
        
    except Exception as e:
        st.error(f"Error aligning dates: {str(e)}")
        return pd.DataFrame()
    
def calculate_correlation_metrics(data):
    """Calculate enhanced correlation metrics with better handling of missing data"""
    metrics = {}
    
    try:
        # Ensure we have enough data points
        min_required_points = 3  # Minimum points needed for correlation
        
        # Calculate standard correlations
        correlation_columns = ['Returns', 'News_Sentiment']
        if 'Reddit_Sentiment' in data.columns:
            correlation_columns.append('Reddit_Sentiment')
            
        correlations = data[correlation_columns].corr()
        
        # Calculate rolling correlations
        window = 5  # 5-day rolling window
        
        # Market Returns vs News Sentiment
        news_data = data[['Returns', 'News_Sentiment']].dropna()
        if len(news_data) >= min_required_points:
            r_news = stats.pearsonr(news_data['Returns'], news_data['News_Sentiment'])
            metrics['news_correlation'] = {
                'coefficient': r_news[0],
                'p_value': r_news[1]
            }
        else:
            metrics['news_correlation'] = {
                'coefficient': np.nan,
                'p_value': np.nan
            }
        
        # Market Returns vs Reddit Sentiment
        if 'Reddit_Sentiment' in data.columns:
            reddit_data = data[['Returns', 'Reddit_Sentiment']].dropna()
            if len(reddit_data) >= min_required_points:
                r_reddit = stats.pearsonr(reddit_data['Returns'], reddit_data['Reddit_Sentiment'])
                metrics['reddit_correlation'] = {
                    'coefficient': r_reddit[0],
                    'p_value': r_reddit[1]
                }
            else:
                metrics['reddit_correlation'] = {
                    'coefficient': np.nan,
                    'p_value': np.nan
                }
        
        return correlations, metrics
        
    except Exception as e:
        st.error(f"Error calculating correlation metrics: {str(e)}")
        return pd.DataFrame(), {}    

def plot_correlation_matrix(correlation_data):
    """Create correlation matrix visualization with enhanced styling"""
    fig = go.Figure(data=go.Heatmap(
        z=correlation_data.values,
        x=correlation_data.columns,
        y=correlation_data.index,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        text=np.round(correlation_data.values, 3),
        texttemplate='%{text}',
        textfont={"size": 16},
        hoverongaps=False,
        customdata=np.round(correlation_data.values, 3),
        hovertemplate='%{x} vs %{y}<br>Correlation: %{customdata:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text="Correlation Matrix",
            font=dict(size=20)
        ),
        height=500,
        xaxis_title="",
        yaxis_title="",
        yaxis_autorange='reversed',
        xaxis=dict(side='top'),  # Move x-axis labels to top
        plot_bgcolor='white'
    )
    
    return fig

def display_correlation_summary(correlations, metrics):
    """Display correlation summary in a more organized layout"""
    st.subheader("Statistical Significance Analysis")
    
    # Create DataFrame for summary table
    summary_data = {
        'Comparison': [],
        'Correlation': [],
        'P-value': [],
        'Interpretation': []
    }
    
    # Add News Sentiment correlations
    if 'news_correlation' in metrics:
        summary_data['Comparison'].append('News Sentiment vs Returns')
        summary_data['Correlation'].append(f"{metrics['news_correlation']['coefficient']:.3f}")
        summary_data['P-value'].append(f"{metrics['news_correlation']['p_value']:.3f}")
        summary_data['Interpretation'].append(get_correlation_description(
            metrics['news_correlation']['coefficient']
        ))
    
    # Add Reddit Sentiment correlations
    if 'reddit_correlation' in metrics:
        summary_data['Comparison'].append('Reddit Sentiment vs Returns')
        summary_data['Correlation'].append(f"{metrics['reddit_correlation']['coefficient']:.3f}")
        summary_data['P-value'].append(f"{metrics['reddit_correlation']['p_value']:.3f}")
        summary_data['Interpretation'].append(get_correlation_description(
            metrics['reddit_correlation']['coefficient']
        ))
    
    # Add cross-platform correlation if available
    if ('News_Sentiment' in correlations.columns and 
        'Reddit_Sentiment' in correlations.columns):
        cross_corr = correlations.loc['News_Sentiment', 'Reddit_Sentiment']
        if not pd.isna(cross_corr):
            summary_data['Comparison'].append('News vs Reddit Sentiment')
            summary_data['Correlation'].append(f"{cross_corr:.3f}")
            summary_data['P-value'].append("N/A")  # No p-value for cross correlation
            summary_data['Interpretation'].append(get_correlation_description(cross_corr))
    
    # Create and display DataFrame
    df_summary = pd.DataFrame(summary_data)
    
    # Style the DataFrame for better presentation
    styled_df = df_summary.style.set_properties(**{
        'background-color': 'black',
        'color': 'white',
        'border': '1px solid gray'
    })
    
    # Display as a table
    st.table(df_summary.set_index('Comparison'))

def interpret_correlation(correlation):
    """Interpret correlation strength"""
    abs_corr = abs(correlation)
    if abs_corr < 0.2:
        strength = "weak"
    elif abs_corr < 0.4:
        strength = "moderate"
    elif abs_corr < 0.6:
        strength = "strong"
    else:
        strength = "very strong"
        
    direction = "positive" if correlation > 0 else "negative"
    return f"{strength} {direction}"

def get_correlation_description(correlation):
    """Get a descriptive interpretation of the correlation"""
    if pd.isna(correlation):
        return "Insufficient data"
        
    abs_corr = abs(correlation)
    if abs_corr < 0.2:
        strength = "weak"
    elif abs_corr < 0.4:
        strength = "moderate"
    elif abs_corr < 0.6:
        strength = "moderately strong"
    else:
        strength = "strong"
        
    direction = "positive" if correlation > 0 else "negative"
    return f"{strength} {direction} correlation"

def get_correlation_strength(correlation):
    """Get correlation strength category"""
    abs_corr = abs(correlation)
    if abs_corr < 0.3:
        return "weak"
    elif abs_corr < 0.7:
        return "moderate"
    else:
        return "strong"

def correlation_analysis_page():
    st.title("🔗 Correlation Analysis")
    
    # Check if required data exists in session state
    if 'market_data' not in st.session_state or st.session_state.market_data is None:
        st.warning("⚠️ Please fetch market data first in the Market Data page")
        st.info("➡️ Go to the Main page and fetch market data for your analysis")
        return
        
    if 'news_data' not in st.session_state or st.session_state.news_data is None:
        st.warning("⚠️ Please complete sentiment analysis first")
        st.info("➡️ Go to the Sentiment Analysis page and fetch news/reddit data")
        return
    
    try:
        # Get data from session state
        market_data = st.session_state.market_data
        news_data = st.session_state.news_data
        reddit_data = st.session_state.get('reddit_data')  # This might be None
        
        # Display data ranges
        st.subheader("Data Ranges")
        col1, col2, col3 = st.columns(3)
        
        # Safely display market data range
        with col1:
            if not market_data.empty and 'Date' in market_data.columns:
                st.write(f"Market Data: {market_data['Date'].min().strftime('%Y-%m-%d')} to {market_data['Date'].max().strftime('%Y-%m-%d')}")
            else:
                st.write("Market Data: No data available")
        
        # Safely display news data range
        with col2:
            if not news_data.empty and 'Date' in news_data.columns:
                st.write(f"News Data: {news_data['Date'].min().strftime('%Y-%m-%d')} to {news_data['Date'].max().strftime('%Y-%m-%d')}")
            else:
                st.write("News Data: No data available")
        
        # Safely display reddit data range if available
        with col3:
            if reddit_data is not None and not reddit_data.empty and 'Date' in reddit_data.columns:
                st.write(f"Reddit Data: {reddit_data['Date'].min().strftime('%Y-%m-%d')} to {reddit_data['Date'].max().strftime('%Y-%m-%d')}")
            else:
                st.write("Reddit Data: No data available")
        
        # Check if we have enough data to proceed
        if market_data.empty or news_data.empty:
            st.error("❌ Insufficient data for correlation analysis")
            return
            
        # Align dates and compute correlations
        combined_data = align_dates_and_compute_returns(market_data, news_data, reddit_data)
        
        if combined_data.empty:
            st.warning("⚠️ No overlapping data found between market returns and sentiment scores")
            return
            
        # Calculate correlation metrics
        correlations, metrics = calculate_correlation_metrics(combined_data)
        
        if correlations.empty:
            st.error("❌ Unable to calculate correlations with the available data")
            return
            
        # Display correlation matrix
        fig = plot_correlation_matrix(correlations)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display correlation summary
        display_correlation_summary(correlations, metrics)
        
        # Add data viewing and download options
        col1, col2 = st.columns(2)
        with col1:
            if st.button("View Aligned Data"):
                st.dataframe(combined_data)
        
        with col2:
            st.download_button(
                "Download Aligned Data",
                combined_data.to_csv(index=False),
                "aligned_data.csv",
                "text/csv",
                key='download-aligned-data'
            )
            
    except Exception as e:
        st.error(f"❌ Error in correlation analysis: {str(e)}")
        st.info("💡 Please ensure you have completed the following steps:")
        st.write("1. Fetched market data in the Main page")
        st.write("2. Completed sentiment analysis in the Sentiment Analysis page")
        if st.button("Show detailed error information"):
            st.exception(e)

if __name__ == "__main__":
    correlation_analysis_page()