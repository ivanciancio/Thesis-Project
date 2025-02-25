# 3_correlation_analysis.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from analysers.x_analyser import XAnalyser

def correlation_analysis_page():
    st.title("🔗 Correlation Analysis")
    
    # Check all required data is available
    if not all(key in st.session_state for key in ['market_data', 'news_data']):
        st.warning("⚠️ Please complete market and news analysis first")
        return
    
    # Display available data sources
    st.header("Available Data Sources")
    
    data_sources = {
        'Market Data': 'market_data',
        'News Data': 'news_data',
        'Reddit Data': 'reddit_data',
        'Twitter Data': 'twitter_data'
    }
    
    available_sources = []
    
    for name, key in data_sources.items():
        if key in st.session_state and st.session_state[key] is not None and not st.session_state[key].empty:
            data = st.session_state[key]
            st.success(f"✅ {name} available - Shape: {data.shape}")
            if 'Date' in data.columns:
                date_range = f"({pd.to_datetime(data['Date']).min().strftime('%Y-%m-%d')} to {pd.to_datetime(data['Date']).max().strftime('%Y-%m-%d')})"
                st.info(f"{name} date range: {date_range}")
            available_sources.append(key)
        else:
            st.warning(f"⚠️ {name} not available")
    
    if len(available_sources) >= 2:
        try:
            # Align dates and compute returns with debug info
            st.header("Data Processing")
            st.write("Aligning dates between different data sources...")
            
            market_data = st.session_state.market_data
            news_data = st.session_state.news_data
            
            # Process Twitter data if available
            twitter_data = None
            if 'twitter_data' in st.session_state and st.session_state.twitter_data is not None:
                twitter_data = st.session_state.twitter_data
                st.write("Twitter data columns:", twitter_data.columns.tolist())
            
            # Process Reddit data if available
            reddit_data = None
            if 'reddit_data' in st.session_state and st.session_state.reddit_data is not None:
                reddit_data = st.session_state.reddit_data
            
            combined_data = align_dates_and_compute_returns(
                market_data, news_data, reddit_data, twitter_data
            )
            
            if not combined_data.empty:
                st.success("✅ Data alignment successful")
                st.write("Combined Data Summary:")
                st.write("- Shape:", combined_data.shape)
                st.write("- Columns:", combined_data.columns.tolist())
                st.write("- Sample:", combined_data.head(3))
                
                # Calculate correlations
                st.header("Correlation Analysis")
                correlations, metrics = calculate_correlation_metrics(combined_data)
                
                if not correlations.empty:
                    # Display correlation matrix
                    st.subheader("Correlation Matrix")
                    fig = plot_correlation_matrix(correlations)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display statistical significance
                    st.subheader("Statistical Significance Analysis")
                    display_correlation_summary(correlations, metrics)
                    
                    # Add view/download options
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("View Aligned Data"):
                            st.dataframe(combined_data)
                    with col2:
                        st.download_button(
                            "Download Aligned Data",
                            combined_data.to_csv(index=False),
                            "aligned_data.csv",
                            "text/csv"
                        )
                else:
                    st.warning("Could not calculate correlations from the aligned data")
            else:
                st.warning("No overlapping data found between market returns and sentiment scores")
        except Exception as e:
            st.error(f"Error in correlation analysis: {str(e)}")
            st.exception(e)
    else:
        st.warning("Need at least two data sources for correlation analysis")

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

def align_dates_and_compute_returns(market_data, news_data, reddit_data=None, twitter_data=None):
    """Align dates between market data and sentiment data with improved debugging"""
    try:
        # Create copies of input data
        market_data = market_data.copy()
        news_data = news_data.copy()
        
        # Debug print column names
        print("Market data columns:", market_data.columns.tolist())
        print("News data columns:", news_data.columns.tolist())
        if twitter_data is not None:
            print("Twitter data columns:", twitter_data.columns.tolist())
        if reddit_data is not None:
            print("Reddit data columns:", reddit_data.columns.tolist())
        
        # Convert all dates to datetime for consistent handling
        market_data['Date'] = pd.to_datetime(market_data['Date'])
        news_data['Date'] = pd.to_datetime(news_data['Date'])
        
        # Extract date only (no time) for consistent grouping
        market_data['DateOnly'] = market_data['Date'].dt.date
        news_data['DateOnly'] = news_data['Date'].dt.date
        
        # Calculate returns if not already present
        if 'Returns' not in market_data.columns:
            market_data['Returns'] = market_data['Close'].pct_change()
        
        # Process news sentiment - check for both possible column names
        sentiment_col = None
        for col in news_data.columns:
            if 'sentiment' in col.lower() and 'score' in col.lower():
                sentiment_col = col
                break
        
        if sentiment_col is None:
            st.warning("Could not find sentiment score column in news data")
            return pd.DataFrame()
            
        news_grouped = news_data.groupby('DateOnly')[sentiment_col].mean().reset_index()
        news_grouped.columns = ['DateOnly', 'News_Sentiment']
        
        # Create base dataframe with market and news data
        base_data = pd.merge(
            market_data[['DateOnly', 'Returns', 'Close']],
            news_grouped,
            on='DateOnly',
            how='inner'
        )
        
        # Add Twitter data if available
        if twitter_data is not None and not twitter_data.empty:
            st.write("Processing Twitter data for correlation analysis...")
            # Convert Twitter dates
            twitter_data = twitter_data.copy()
            twitter_data['Date'] = pd.to_datetime(twitter_data['Date'])
            twitter_data['DateOnly'] = twitter_data['Date'].dt.date
            
            # Debug Twitter data
            st.write("Twitter date range:", twitter_data['Date'].min(), "to", twitter_data['Date'].max())
            
            # Find sentiment column - try both Sentiment_Score and other variations
            twitter_sentiment_col = None
            for col in twitter_data.columns:
                if 'sentiment' in col.lower() and 'score' in col.lower():
                    twitter_sentiment_col = col
                    st.write(f"Found Twitter sentiment column: {twitter_sentiment_col}")
                    break
            
            if twitter_sentiment_col is None:
                st.warning("Could not find sentiment score column in Twitter data")
                # Check if we can use raw Twitter data instead
                if 'twitter_raw_data' in st.session_state and not st.session_state.twitter_raw_data.empty:
                    st.write("Using raw Twitter data instead...")
                    raw_twitter = st.session_state.twitter_raw_data.copy()
                    raw_twitter['Date'] = pd.to_datetime(raw_twitter['Date'])
                    raw_twitter['DateOnly'] = raw_twitter['Date'].dt.date
                    
                    # Check for sentiment column in raw data
                    for col in raw_twitter.columns:
                        if 'sentiment' in col.lower() and 'score' in col.lower():
                            twitter_sentiment_col = col
                            st.write(f"Found Twitter sentiment column in raw data: {twitter_sentiment_col}")
                            
                            # Group Twitter data by date
                            twitter_grouped = raw_twitter.groupby('DateOnly')[twitter_sentiment_col].mean().reset_index()
                            twitter_grouped.columns = ['DateOnly', 'Twitter_Sentiment']
                            
                            # Display sample of Twitter sentiment data
                            st.write("Twitter sentiment sample:", twitter_grouped.head())
                            
                            # Merge Twitter data with base data
                            base_data = pd.merge(
                                base_data,
                                twitter_grouped,
                                on='DateOnly',
                                how='left'
                            )
                            
                            # Fill any missing Twitter sentiment with the mean
                            if 'Twitter_Sentiment' in base_data.columns:
                                twitter_mean = base_data['Twitter_Sentiment'].mean()
                                base_data['Twitter_Sentiment'].fillna(twitter_mean, inplace=True)
                            break
                    
                    if twitter_sentiment_col is None:
                        st.error("Could not find sentiment column in raw Twitter data either")
            else:
                # Group Twitter data by date
                twitter_grouped = twitter_data.groupby('DateOnly')[twitter_sentiment_col].mean().reset_index()
                twitter_grouped.columns = ['DateOnly', 'Twitter_Sentiment']
                
                # Display sample of Twitter sentiment data
                st.write("Twitter sentiment sample:", twitter_grouped.head())
                
                # Merge Twitter data
                base_data = pd.merge(
                    base_data,
                    twitter_grouped,
                    on='DateOnly',
                    how='left'
                )
                
                # Fill any missing Twitter sentiment with the mean
                if 'Twitter_Sentiment' in base_data.columns:
                    twitter_mean = base_data['Twitter_Sentiment'].mean()
                    base_data['Twitter_Sentiment'] = base_data['Twitter_Sentiment'].fillna(twitter_mean)
                    st.write("Twitter sentiment after merging:", base_data['Twitter_Sentiment'].head())
        
        # Add Reddit data if available
        if reddit_data is not None and not reddit_data.empty:
            # Convert Reddit dates
            reddit_data = reddit_data.copy()
            reddit_data['Date'] = pd.to_datetime(reddit_data['Date'])
            reddit_data['DateOnly'] = reddit_data['Date'].dt.date
            
            # Check which sentiment column to use
            reddit_sentiment_col = None
            for col in reddit_data.columns:
                if 'sentiment' in col.lower() and 'score' in col.lower():
                    reddit_sentiment_col = col
                    break
            
            if reddit_sentiment_col:
                # Group Reddit data by date
                reddit_grouped = reddit_data.groupby('DateOnly')[reddit_sentiment_col].mean().reset_index()
                reddit_grouped.columns = ['DateOnly', 'Reddit_Sentiment']
                
                # Merge Reddit data
                base_data = pd.merge(
                    base_data,
                    reddit_grouped,
                    on='DateOnly',
                    how='left'
                )
                
                # Fill any missing Reddit sentiment with the mean
                if 'Reddit_Sentiment' in base_data.columns:
                    reddit_mean = base_data['Reddit_Sentiment'].mean()
                    base_data['Reddit_Sentiment'] = base_data['Reddit_Sentiment'].fillna(reddit_mean)
            else:
                st.warning("Could not find sentiment column in Reddit data")
        
        # Create 'Date' column back from DateOnly for visualization
        base_data['Date'] = pd.to_datetime(base_data['DateOnly'])
        
        # Sort by date
        final_data = base_data.sort_values('Date')
        
        # Display the final data columns and sample
        st.write("Final aligned data columns:", final_data.columns.tolist())
        st.write("Final aligned data shape:", final_data.shape)
        st.write("Sample of final aligned data:", final_data.head(3))
        
        return final_data
        
    except Exception as e:
        st.error(f"Error in date alignment: {str(e)}")
        st.exception(e)
        return pd.DataFrame()

def calculate_correlation_metrics(data):
    """Calculate correlation metrics with p-values"""
    try:
        # Identify columns for correlation
        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # Filter for the columns we want in correlation analysis
        target_columns = []
        for col in numeric_columns:
            if any(keyword in col.lower() for keyword in ['returns', 'sentiment', 'close']):
                target_columns.append(col)
        
        if len(target_columns) < 2:
            st.error("Not enough suitable columns for correlation analysis")
            return pd.DataFrame(), pd.DataFrame()
        
        # Create copy of data for analysis
        analysis_data = data[target_columns].copy()
        
        # Calculate correlations
        correlations = analysis_data.corr(method='pearson')
        
        # Calculate metrics with p-values
        metrics_list = []
        p_values = np.zeros_like(correlations, dtype=float)
        
        for i in range(len(target_columns)):
            for j in range(len(target_columns)):
                if i != j:
                    col1, col2 = target_columns[i], target_columns[j]
                    valid_data = analysis_data[[col1, col2]].dropna()
                    
                    if len(valid_data) >= 3:
                        r_value, p_value = stats.pearsonr(valid_data[col1], valid_data[col2])
                        p_values[i, j] = p_value
                        
                        metrics_list.append({
                            'Comparison': f"{col1} vs {col2}",
                            'Correlation': f"{r_value:.4f}",
                            'P-value': f"{p_value:.4f}",
                            'Sample Size': len(valid_data),
                            'Significance': 'Significant' if p_value < 0.05 else 'Not Significant',
                            'Interpretation': f"{interpret_correlation_strength(r_value)} {interpret_correlation_direction(r_value)}"
                        })
        
        metrics_df = pd.DataFrame(metrics_list)
        
        # Add p-values to metrics
        metrics = {
            'p_values': p_values
        }
        
        return correlations, metrics_df
        
    except Exception as e:
        st.error(f"Error calculating correlations: {str(e)}")
        st.exception(e)
        return pd.DataFrame(), pd.DataFrame()

def plot_correlation_matrix(correlation_data):
    """Create correlation matrix visualization"""
    fig = go.Figure(data=go.Heatmap(
        z=correlation_data.values,
        x=correlation_data.columns,
        y=correlation_data.index,
        colorscale=[
            [0.0, '#FF4B4B'],    # Dark red for strong negative
            [0.5, '#FFFFFF'],    # White for no correlation
            [1.0, '#156FFF']     # Dark blue for strong positive
        ],
        zmin=-1,
        zmax=1,
        text=np.round(correlation_data.values, 3),
        texttemplate='%{text}',
        textfont={"size": 14, "color": "black"},
        showscale=True,
        colorbar=dict(
            title=dict(
                text="Correlation",
                side="right"
            ),
            ticktext=["-1", "-0.5", "0", "0.5", "1"],
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticks="outside"
        )
    ))

    # Update layout
    fig.update_layout(
        title={
            'text': 'Correlation Matrix',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        height=500,
        width=700,
        xaxis={'side': 'bottom', 'tickangle': 45},
        yaxis={'autorange': 'reversed'},
        margin=dict(t=100, l=100, r=100, b=100)
    )
    
    return fig

def display_correlation_summary(correlations, metrics_df):
    """Display enhanced correlation summary"""
    if not metrics_df.empty:
        # Add color coding based on p-value
        def color_p_value(val):
            try:
                p_val = float(val)
                if p_val < 0.01:
                    return 'background-color: #90EE90'  # Light green
                elif p_val < 0.05:
                    return 'background-color: #FFFFE0'  # Light yellow
                else:
                    return 'background-color: #FFB6C1'  # Light red
            except:
                return ''
        
        # Style the dataframe
        styled_df = metrics_df.style.map(color_p_value, subset=['P-value'])
        
        st.write("Color coding:")
        st.write("🟢 p < 0.01: Strong evidence")
        st.write("🟡 p < 0.05: Moderate evidence")
        st.write("🔴 p ≥ 0.05: Weak evidence")
        
        st.dataframe(styled_df)
    else:
        st.warning("No correlation data available for analysis")

def interpret_correlation(correlation):
    """Get detailed interpretation of correlation value"""
    if np.isnan(correlation):
        return "insufficient data"
        
    abs_corr = abs(correlation)
    direction = "positive" if correlation > 0 else "negative"
    
    if abs_corr < 0.2:
        strength = "very weak"
    elif abs_corr < 0.4:
        strength = "weak"
    elif abs_corr < 0.6:
        strength = "moderate"
    elif abs_corr < 0.8:
        strength = "strong"
    else:
        strength = "very strong"
        
    return f"{strength} {direction}"

if __name__ == "__main__":
    correlation_analysis_page()