# pages/3_correlation_analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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
    """Calculate enhanced correlation metrics"""
    metrics = {}
    
    try:
        # Calculate standard correlations
        correlation_columns = ['Returns', 'News_Sentiment']
        if 'Reddit_Sentiment' in data.columns:
            correlation_columns.append('Reddit_Sentiment')
            
        correlations = data[correlation_columns].corr()
        
        # Calculate rolling correlations
        window = 5  # 5-day rolling window
        rolling_corr_news = data['Returns'].rolling(window).corr(data['News_Sentiment'])
        metrics['rolling_corr_news'] = rolling_corr_news.mean()
        
        if 'Reddit_Sentiment' in data.columns:
            rolling_corr_reddit = data['Returns'].rolling(window).corr(data['Reddit_Sentiment'])
            metrics['rolling_corr_reddit'] = rolling_corr_reddit.mean()
        
        # Add correlation significance tests
        from scipy import stats
        
        # Market Returns vs News Sentiment
        r_news = stats.pearsonr(data['Returns'], data['News_Sentiment'])
        metrics['news_correlation'] = {
            'coefficient': r_news[0],
            'p_value': r_news[1]
        }
        
        # Market Returns vs Reddit Sentiment
        if 'Reddit_Sentiment' in data.columns:
            r_reddit = stats.pearsonr(data['Returns'], data['Reddit_Sentiment'])
            metrics['reddit_correlation'] = {
                'coefficient': r_reddit[0],
                'p_value': r_reddit[1]
            }
        
        return correlations, metrics
        
    except Exception as e:
        st.error(f"Error calculating correlation metrics: {str(e)}")
        st.write("Data columns available:", data.columns.tolist())
        return pd.DataFrame(), {}    

def plot_correlation_matrix(correlation_data):
    """Create correlation matrix visualisation"""
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

def display_correlation_summary(correlations, metrics):
    """Display correlation summary in a more organised layout"""
    st.subheader("Correlation Summary")

    # Create columns for different correlation categories
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Market Correlations")
        market_data = []
        # Add Returns vs News Sentiment correlation
        if 'news_correlation' in metrics:
            market_data.append({
                'Variable': 'News Sentiment',
                'Correlation': f"{metrics['news_correlation']['coefficient']:.3f}",
                'Interpretation': interpret_correlation(metrics['news_correlation']['coefficient'])
            })
        # Add Returns vs Reddit Sentiment correlation
        if 'reddit_correlation' in metrics:
            market_data.append({
                'Variable': 'Reddit Sentiment',
                'Correlation': f"{metrics['reddit_correlation']['coefficient']:.3f}",
                'Interpretation': interpret_correlation(metrics['reddit_correlation']['coefficient'])
            })
        
        if market_data:
            st.table(pd.DataFrame(market_data))
        else:
            st.write("No market correlations available")

    with col2:
        st.markdown("#### Sentiment Correlations")
        sentiment_data = []
        # Add News vs Reddit Sentiment correlation if available
        if 'News_Sentiment' in correlations.columns and 'Reddit_Sentiment' in correlations.columns:
            corr = correlations.loc['News_Sentiment', 'Reddit_Sentiment']
            sentiment_data.append({
                'Variable': 'News vs Reddit',
                'Correlation': f"{corr:.3f}",
                'Interpretation': interpret_correlation(corr)
            })
        
        if sentiment_data:
            st.table(pd.DataFrame(sentiment_data))
        else:
            st.write("No sentiment correlations available")

    # Statistical Significance section
    st.subheader("Statistical Significance")
    
    # Display p-values
    col1, col2 = st.columns(2)
    with col1:
        if 'news_correlation' in metrics:
            st.metric(
                "News Sentiment p-value",
                f"{metrics['news_correlation']['p_value']:.3f}"
            )
            
            # Add correlation interpretation
            corr = metrics['news_correlation']['coefficient']
            st.write(f"News Sentiment vs Returns: {corr:.3f}")
            st.write(get_correlation_description('News Sentiment vs Returns', corr))

    with col2:
        if 'reddit_correlation' in metrics:
            st.metric(
                "Reddit Sentiment p-value",
                f"{metrics['reddit_correlation']['p_value']:.3f}"
            )
            
            # Add correlation interpretation
            corr = metrics['reddit_correlation']['coefficient']
            st.write(f"Reddit Sentiment vs Returns: {corr:.3f}")
            st.write(get_correlation_description('Reddit Sentiment vs Returns', corr))

    # Cross-platform correlation if available
    if 'News_Sentiment' in correlations.columns and 'Reddit_Sentiment' in correlations.columns:
        st.write("News Sentiment vs Reddit Sentiment:", 
                f"{correlations.loc['News_Sentiment', 'Reddit_Sentiment']:.3f}")
        st.write(get_correlation_description(
            'News Sentiment vs Reddit Sentiment',
            correlations.loc['News_Sentiment', 'Reddit_Sentiment']
        ))

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

def get_correlation_description(pair, correlation):
    """Get a descriptive interpretation of the correlation"""
    if abs(correlation) < 0.2:
        strength = "weak"
    elif abs(correlation) < 0.4:
        strength = "moderate"
    elif abs(correlation) < 0.6:
        strength = "strong"
    else:
        strength = "very strong"
        
    direction = "positive" if correlation > 0 else "negative"
    return f"There is a {strength} {direction} correlation."

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
        
        # Calculate enhanced correlation metrics
        correlations, metrics = calculate_correlation_metrics(combined_data)
        
        # Display correlation matrix
        st.subheader("Correlation Matrix")
        fig = plot_correlation_matrix(correlations)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Display organised correlation summary
        display_correlation_summary(correlations, metrics)
        
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