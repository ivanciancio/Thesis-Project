import warnings
import os
import sys
from pathlib import Path
import logging

# Get the absolute path of the project root directory and add it to Python path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent  # Go up one level from the 'pages' directory
sys.path.insert(0, str(project_root))

# Now import from utils will work
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.api_helpers import fetch_news_data  # Remove fetch_news_data_fixed
from analysers.sentiment_analyser import FinancialSentimentAnalyser
from analysers.reddit_analyser import RedditAnalyser
from analysers.x_analyser import XAnalyser
from utils.visualisation import plot_model_comparison, plot_model_correlation_matrix
from datetime import datetime, timedelta
import logging
import numpy as np

def display_model_info(sentiment_analyser):
    """Display information about active sentiment models and their weights without using charts"""
    st.sidebar.header("🧠 Sentiment Models")
    
    # Show active models and weights
    models_info = {
        'textblob': {'name': 'TextBlob', 'color': '#3498db', 'active': sentiment_analyser.models.get('textblob', False)},
        'vader': {'name': 'VADER', 'color': '#2ecc71', 'active': sentiment_analyser.models.get('vader', False)},
        'finbert': {'name': 'FinBERT', 'color': '#e74c3c', 'active': sentiment_analyser.models.get('finbert', False)}
    }
    
    # Display active models
    st.sidebar.subheader("Active Models")
    active_models = []
    for model_id, info in models_info.items():
        if info['active']:
            active_models.append(f"✅ {info['name']}")
        else:
            active_models.append(f"❌ {info['name']}")
    
    st.sidebar.write('\n'.join(active_models))
    
    # Display model weights as text instead of chart
    st.sidebar.subheader("Model Weights")
    weights = sentiment_analyser.model_weights
    
    # Display weights as simple text
    for model_id, info in models_info.items():
        if info['active']:
            weight_pct = weights.get(model_id, 0) * 100
            st.sidebar.text(f"{info['name']}: {weight_pct:.1f}%")
    
    # Add custom weight adjustment
    st.sidebar.subheader("Adjust Weights")
    custom_weights = {}
    
    for model_id, info in models_info.items():
        if info['active']:
            custom_weights[model_id] = st.sidebar.slider(
                f"{info['name']} Weight",
                min_value=0.0,
                max_value=1.0,
                value=weights.get(model_id, 0.0),
                step=0.1,
                key=f"weight_{model_id}_{id(sentiment_analyser)}"  # Add unique key
            )
    
    if st.sidebar.button("Apply Custom Weights", key=f"apply_weights_{id(sentiment_analyser)}"):  # Add unique key
        # Normalize weights
        total = sum(custom_weights.values())
        if total > 0:
            for model in custom_weights:
                sentiment_analyser.model_weights[model] = custom_weights[model] / total
            st.sidebar.success("✅ Custom weights applied!")
            # Force re-normalization of weights
            sentiment_analyser._normalise_weights()

def plot_daily_sentiment(news_df):
    """Create daily sentiment visualisation"""
    # Ensure date column is datetime and strip timezone
    news_df['Date'] = pd.to_datetime(news_df['Date']).dt.tz_localize(None)
    
    # Extract the hour from the datetime for more granular visualisation
    news_df['Hour'] = news_df['Date'].dt.hour
    
    # Group by date and hour for more granular view
    # NOTE: Using the correct column name with space, not underscore
    hourly_sentiment = (news_df
        .groupby([news_df['Date'].dt.date, 'Hour'])
        .agg({
            'Sentiment Score': 'mean',  # This matches the actual column name
            'Title': 'count'
        })
        .reset_index()
    )
    
    # Convert back to datetime
    hourly_sentiment['DateTime'] = pd.to_datetime(
        hourly_sentiment['Date'].astype(str) + ' ' + 
        hourly_sentiment['Hour'].astype(str) + ':00:00'
    )
    
    # Create figure
    fig = go.Figure()
    
    # Add sentiment score line
    fig.add_trace(
        go.Scatter(
            x=hourly_sentiment['DateTime'],
            y=hourly_sentiment['Sentiment Score'],  # Correct column name here too
            mode='lines+markers',
            name='News Sentiment',
            line=dict(color='blue', width=2),
            marker=dict(
                size=8,
                symbol='circle',
                color='blue',
                line=dict(color='white', width=1)
            )
        )
    )
    
    # Add article count bars
    fig.add_trace(
        go.Bar(
            x=hourly_sentiment['DateTime'],
            y=hourly_sentiment['Title'],
            name='Number of Articles',
            yaxis='y2',
            marker_color='rgba(0,0,255,0.2)',
            width=3600000  # One hour width in milliseconds
        )
    )
    
    # Update layout
    fig.update_layout(
        title="Sentiment Analysis Over Time",
        xaxis=dict(
            title="Date",
            type='date',
            tickformat='%Y-%m-%d %H:%M',
            tickangle=45,
            tickmode='auto',
            nticks=20,
            showgrid=True,
            gridcolor='rgba(128,128,128,0.1)'
        ),
        yaxis=dict(
            title="Sentiment Score",
            range=[-1, 1],
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=True,
            zerolinecolor='rgba(128,128,128,0.4)',
            tickformat='.2f'
        ),
        yaxis2=dict(
            title="Number of Articles",
            overlaying='y',
            side='right',
            showgrid=False,
            range=[0, max(hourly_sentiment['Title']) * 1.2]
        ),
        height=500,
        showlegend=True,
        hovermode='x unified',
        plot_bgcolor='white',
        margin=dict(b=100),
        bargap=0
    )
    
    return fig

def plot_sentiment_comparison(news_data, reddit_data=None):
    """Create sentiment comparison visualisation"""
    fig = go.Figure()
    
    # Add news sentiment
    if news_data is not None and not news_data.empty:
        daily_news = news_data.groupby(news_data['Date'].dt.date).agg({
            'Sentiment_Score': 'mean',
            'Title': 'count'
        }).reset_index()
        
        fig.add_trace(
            go.Scatter(
                x=daily_news['Date'],
                y=daily_news['Sentiment_Score'],
                mode='lines+markers',
                name='News Sentiment',
                line=dict(color='blue')
            )
        )
    
    # Add Reddit sentiment if available
    if reddit_data is not None and not reddit_data.empty:
        daily_reddit = reddit_data.groupby(reddit_data['Date'].dt.date).agg({
            'Sentiment_Score': 'mean',
            'Text': 'count'
        }).reset_index()
        
        fig.add_trace(
            go.Scatter(
                x=daily_reddit['Date'],
                y=daily_reddit['Sentiment_Score'],
                mode='lines+markers',
                name='Reddit Sentiment',
                line=dict(color='orange')
            )
        )
    
    fig.update_layout(
        title="Sentiment Comparison Over Time",
        xaxis_title="Date",
        yaxis_title="Sentiment Score",
        height=500,
        showlegend=True,
        hovermode='x unified',
        yaxis=dict(
            tickformat='.2f',
            title_font=dict(size=14),
            gridcolor='rgba(128,128,128,0.2)',
        ),
        xaxis=dict(
            title_font=dict(size=14),
            gridcolor='rgba(128,128,128,0.2)',
        ),
        plot_bgcolor='white'
    )
    
    return fig

def news_analysis_tab():
    st.header("EODHD News Analysis")
    
    # Add tabs for different analysis views
    news_tabs = st.tabs(["Ensemble Analysis", "Model Comparison", "Raw Data"])
    
    if st.button("Fetch News Data"):
        with st.spinner("Fetching news data..."):
            try:
                # Get market data date range
                start_date = st.session_state.market_data['Date'].min()
                end_date = st.session_state.market_data['Date'].max()
                
                st.info(f"Fetching news for market data period: {start_date.date()} to {end_date.date()}")
                
                # Fetch news for the market data period
                news_data = fetch_news_data(
                    symbol=st.session_state.symbol,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if news_data is not None and not news_data.empty:
                    # Initialize sentiment analyzer
                    sentiment_analyser = FinancialSentimentAnalyser()
                    
                    # Process each news item with all models
                    analysed_news = []
                    total_news = len(news_data)
                    st.write(f"Analyzing {total_news} news items...")
                    progress_bar = st.progress(0)

                    for i, (_, row) in enumerate(news_data.iterrows()):
                        # Update progress
                        progress = min(i / total_news, 1.0)
                        progress_bar.progress(progress)
                        
                        text = f"{row['Title']} {row.get('Text', '')}"
                        # Get results from all models
                        sentiment_result = sentiment_analyser.analyse_sentiment(text, return_all_models=True)
                        
                        # Create base result
                        news_item = {
                            'Date': row['Date'],
                            'Title': row['Title'],
                            'Sentiment Score': sentiment_result['score'],  # Space, not underscore
                            'Sentiment': sentiment_result['sentiment'],
                            'Confidence': sentiment_result['confidence']
                        }
                        
                        # Add individual model scores if available
                        if 'individual_models' in sentiment_result:
                            for model, model_result in sentiment_result['individual_models'].items():
                                news_item[f'{model}_score'] = model_result['score']
                                news_item[f'{model}_sentiment'] = model_result['sentiment']
                                news_item[f'{model}_confidence'] = model_result['confidence']
                        
                        analysed_news.append(news_item)

                    # Clear progress bar when done
                    progress_bar.empty()
                    
                    # Create DataFrame with analysis results
                    news_df = pd.DataFrame(analysed_news)
                    
                    news_df['Date'] = pd.to_datetime(news_df['Date']).dt.tz_localize(None)
                    st.session_state.news_data = news_df
                    
                    # Store available models
                    available_models = []
                    for model in ['textblob', 'vader', 'finbert']:
                        if f'{model}_score' in news_df.columns:
                            available_models.append(model)
                    st.session_state.available_models = available_models
                    
                    # ENSEMBLE ANALYSIS TAB
                    with news_tabs[0]:
                        st.subheader("Ensemble Sentiment Analysis")
                        
                        # Show summary metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Average Sentiment", f"{news_df['Sentiment Score'].mean():.2f}")
                        with col2:
                            st.metric("Positive Articles", len(news_df[news_df['Sentiment'] == 'Positive']))
                        with col3:
                            st.metric("Negative Articles", len(news_df[news_df['Sentiment'] == 'Negative']))
                        
                        # Display sentiment trend
                        st.subheader("Sentiment Trend")
                        fig = plot_daily_sentiment(news_df)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # MODEL COMPARISON TAB
                    with news_tabs[1]:
                        st.subheader("Model Comparison")
                        
                        # Check if we have model-specific scores
                        if len(available_models) > 0:
                            # Show metrics for each model
                            model_metrics = []
                            for model in available_models:
                                metrics = {
                                    'Model': model.capitalize(),
                                    'Average Score': news_df[f'{model}_score'].mean(),
                                    'Std Dev': news_df[f'{model}_score'].std(),
                                    'Positive %': len(news_df[news_df[f'{model}_sentiment'] == 'Positive']) / len(news_df) * 100,
                                    'Negative %': len(news_df[news_df[f'{model}_sentiment'] == 'Negative']) / len(news_df) * 100,
                                    'Neutral %': len(news_df[news_df[f'{model}_sentiment'] == 'Neutral']) / len(news_df) * 100
                                }
                                model_metrics.append(metrics)
                            
                            # Add ensemble metrics
                            model_metrics.append({
                                'Model': 'Ensemble',
                                'Average Score': news_df['Sentiment Score'].mean(),
                                'Std Dev': news_df['Sentiment Score'].std(),
                                'Positive %': len(news_df[news_df['Sentiment'] == 'Positive']) / len(news_df) * 100,
                                'Negative %': len(news_df[news_df['Sentiment'] == 'Negative']) / len(news_df) * 100,
                                'Neutral %': len(news_df[news_df['Sentiment'] == 'Neutral']) / len(news_df) * 100
                            })
                            
                            # Display metrics table
                            metrics_df = pd.DataFrame(model_metrics)
                            metrics_df = metrics_df.set_index('Model')
                            st.dataframe(metrics_df.style.format({
                                'Average Score': '{:.3f}',
                                'Std Dev': '{:.3f}',
                                'Positive %': '{:.1f}%',
                                'Negative %': '{:.1f}%',
                                'Neutral %': '{:.1f}%'
                            }))
                            
                            # If plot_model_comparison function is available
                            try:
                                # Display model comparison plot
                                st.subheader("Model Score Comparison")
                                comp_fig = plot_model_comparison(news_df, available_models)
                                st.plotly_chart(comp_fig, use_container_width=True)
                                
                                # Display correlation matrix
                                st.subheader("Model Correlation Matrix")
                                corr_fig = plot_model_correlation_matrix(news_df, available_models)
                                st.plotly_chart(corr_fig)
                            except Exception as e:
                                st.warning(f"Could not generate model comparison plots: {str(e)}")
                        else:
                            st.warning("No model-specific data available for comparison.")
                    
                    # RAW DATA TAB
                    with news_tabs[2]:
                        st.subheader("News Articles")
                        
                        # Always include both basic and model columns
                        basic_columns = ['Date', 'Title', 'Sentiment', 'Sentiment Score']
                        model_columns = []
                        
                        # Get model columns
                        for model in available_models:
                            col_name = f"{model}_score"
                            if col_name in news_df.columns:
                                # Create nicely formatted column name
                                news_df[f"{model.capitalize()} Score"] = news_df[col_name]
                                model_columns.append(f"{model.capitalize()} Score")
                        
                        # Combine all columns
                        all_columns = basic_columns + model_columns
                        
                        # Create display dataframe with all columns
                        display_df = news_df[all_columns].copy()
                        
                        # Format date
                        display_df['Date'] = pd.to_datetime(display_df['Date']).dt.strftime('%Y-%m-%d %H:%M:%S')
                        
                        # Display dataframe
                        st.dataframe(
                            display_df.sort_values('Date', ascending=False),
                            use_container_width=True
                        )
                        
                        # Add download button
                        st.download_button(
                            "Download News Analysis Data",
                            news_df.to_csv(index=False),
                            "news_analysis.csv",
                            "text/csv",
                            key='download-news-data'
                        )
                else:
                    st.error(f"No news data found for {st.session_state.symbol} in the market data period")
                    
            except Exception as e:
                st.error(f"Error fetching news data: {str(e)}")
                st.exception(e)  # Show full exception for debugging

def x_analysis_tab():
    st.header("X (Twitter) Analysis")
    
    # Add tabs for different analysis views
    x_tabs = st.tabs(["Ensemble Analysis", "Model Comparison", "Raw Data"])
    
    if st.button("Fetch X Data"):
        with st.spinner("Fetching X data..."):
            try:
                # Get market data date range and ensure timezone-naive
                start_date = pd.to_datetime(st.session_state.market_data['Date'].min()).tz_localize(None)
                end_date = pd.to_datetime(st.session_state.market_data['Date'].max()).tz_localize(None)
                
                st.info(f"Fetching tweets for market data period: {start_date.date()} to {end_date.date()}")
                
                # Initialize X analyser
                x_analyser = XAnalyser()
                
                # Fetch X data
                twitter_data = x_analyser.fetch_twitter_data(
                    symbol=st.session_state.symbol,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if twitter_data is not None and not twitter_data.empty:
                    # Initialize sentiment analyzer
                    sentiment_analyser = FinancialSentimentAnalyser()
                    
                    # Analyze content
                    analysed_data = x_analyser.analyse_content(twitter_data, sentiment_analyser)
                    
                    if not analysed_data.empty:
                        # Save hourly aggregated data for visualization
                        st.session_state.twitter_data = analysed_data
                        
                        # Make raw tweet data available for correlation analysis
                        raw_twitter_available = 'twitter_raw_data' in st.session_state and not st.session_state.twitter_raw_data.empty
                        
                        # ENSEMBLE ANALYSIS TAB
                        with x_tabs[0]:
                            st.subheader("X Sentiment Analysis")
                            
                            # Calculate sentiment metrics
                            avg_sentiment = analysed_data['Sentiment_Score'].mean()
                            total_tweets = analysed_data['Tweet_Count'].sum()
                            
                            # Show summary metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(
                                    "Average Sentiment",
                                    f"{avg_sentiment:.2f}"
                                )
                                st.metric(
                                    "Total Tweets",
                                    total_tweets
                                )
                            
                            with col2:
                                st.metric(
                                    "Average Daily Tweets",
                                    f"{total_tweets / len(analysed_data):.1f}"
                                )
                                st.metric(
                                    "Days Analyzed",
                                    len(analysed_data)
                                )
                            
                            with col3:
                                st.metric(
                                    "Average Engagement",
                                    f"{analysed_data['Engagement_Mean'].mean():.1f}"
                                )
                                st.metric(
                                    "Total Engagement",
                                    int(analysed_data['Engagement_Total'].sum())
                                )
                            
                            # Display sentiment trend
                            st.subheader("Sentiment Trend")
                            fig = x_analyser.plot_sentiment_trend(analysed_data)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display engagement analysis
                            st.subheader("Engagement Analysis")
                            
                            # Create engagement metrics
                            eng_col1, eng_col2 = st.columns(2)
                            
                            with eng_col1:
                                engagement_corr = analysed_data['Sentiment_Score'].corr(
                                    analysed_data['Engagement_Mean']
                                )
                                st.metric(
                                    "Sentiment-Engagement Correlation",
                                    f"{engagement_corr:.2f}"
                                )
                            
                            with eng_col2:
                                st.write("Daily Statistics:")
                                daily_stats = analysed_data[[
                                    'Sentiment_Score',
                                    'Engagement_Mean',
                                    'Tweet_Count'
                                ]].describe()
                                st.dataframe(daily_stats)
                        
                        # MODEL COMPARISON TAB
                        with x_tabs[1]:
                            st.subheader("Model Comparison")
                            
                            if raw_twitter_available:
                                raw_tweets = st.session_state.twitter_raw_data
                                
                                # Check for available models
                                available_models = []
                                for model in ['textblob', 'vader', 'finbert']:
                                    if f'{model}_score' in raw_tweets.columns:
                                        available_models.append(model)
                                
                                if available_models:
                                    # Show metrics for each model
                                    model_metrics = []
                                    for model in available_models:
                                        metrics = {
                                            'Model': model.capitalize(),
                                            'Average Score': raw_tweets[f'{model}_score'].mean(),
                                            'Std Dev': raw_tweets[f'{model}_score'].std(),
                                            'Positive %': len(raw_tweets[raw_tweets[f'{model}_sentiment'] == 'Positive']) / len(raw_tweets) * 100,
                                            'Negative %': len(raw_tweets[raw_tweets[f'{model}_sentiment'] == 'Negative']) / len(raw_tweets) * 100,
                                            'Neutral %': len(raw_tweets[raw_tweets[f'{model}_sentiment'] == 'Neutral']) / len(raw_tweets) * 100
                                        }
                                        model_metrics.append(metrics)
                                    
                                    # Add ensemble metrics
                                    model_metrics.append({
                                        'Model': 'Ensemble',
                                        'Average Score': raw_tweets['Sentiment_Score'].mean(),
                                        'Std Dev': raw_tweets['Sentiment_Score'].std(),
                                        'Positive %': len(raw_tweets[raw_tweets['Sentiment'] == 'Positive']) / len(raw_tweets) * 100,
                                        'Negative %': len(raw_tweets[raw_tweets['Sentiment'] == 'Negative']) / len(raw_tweets) * 100,
                                        'Neutral %': len(raw_tweets[raw_tweets['Sentiment'] == 'Neutral']) / len(raw_tweets) * 100
                                    })
                                    
                                    # Display metrics table
                                    metrics_df = pd.DataFrame(model_metrics)
                                    metrics_df = metrics_df.set_index('Model')
                                    st.dataframe(metrics_df.style.format({
                                        'Average Score': '{:.3f}',
                                        'Std Dev': '{:.3f}',
                                        'Positive %': '{:.1f}%',
                                        'Negative %': '{:.1f}%',
                                        'Neutral %': '{:.1f}%'
                                    }))
                                    
                                    # Create model comparison visualization
                                    try:
                                        st.subheader("Model Score Comparison")
                                        # Create a simple line chart for each model's average daily sentiment
                                        model_data = {}
                                        
                                        # Add date column for x-axis
                                        model_data['Date'] = raw_tweets['Date'].dt.date
                                        
                                        # Add each model's score
                                        for model in available_models:
                                            model_data[model.capitalize()] = raw_tweets[f'{model}_score']
                                        
                                        # Add ensemble score
                                        model_data['Ensemble'] = raw_tweets['Sentiment_Score']
                                        
                                        # Convert to DataFrame
                                        model_df = pd.DataFrame(model_data)
                                        
                                        # Group by date and calculate mean
                                        daily_model_df = model_df.groupby('Date').mean().reset_index()
                                        
                                        # Create line chart using Plotly
                                        fig = go.Figure()
                                        
                                        for model in available_models:
                                            fig.add_trace(go.Scatter(
                                                x=daily_model_df['Date'],
                                                y=daily_model_df[model.capitalize()],
                                                mode='lines',
                                                name=model.capitalize()
                                            ))
                                        
                                        fig.add_trace(go.Scatter(
                                            x=daily_model_df['Date'],
                                            y=daily_model_df['Ensemble'],
                                            mode='lines',
                                            name='Ensemble',
                                            line=dict(width=3, dash='dash')
                                        ))
                                        
                                        fig.update_layout(
                                            title="Model Sentiment Comparison",
                                            xaxis_title="Date",
                                            yaxis_title="Sentiment Score",
                                            yaxis=dict(range=[-1, 1]),
                                            legend=dict(
                                                orientation="h",
                                                yanchor="bottom",
                                                y=1.02,
                                                xanchor="right",
                                                x=1
                                            )
                                        )
                                        
                                        st.plotly_chart(fig, use_container_width=True)
                                    except Exception as e:
                                        st.warning(f"Could not create visualization: {e}")
                                        
                                else:
                                    st.warning("No model-specific data available for comparison.")
                            else:
                                st.warning("No raw tweet data available for model comparison.")
                        
                        # RAW DATA TAB
                        with x_tabs[2]:
                            st.subheader("X (Twitter) Data")
                            
                            if raw_twitter_available:
                                raw_tweets = st.session_state.twitter_raw_data
                                
                                # Get available models
                                available_models = []
                                for model in ['textblob', 'vader', 'finbert']:
                                    if f'{model}_score' in raw_tweets.columns:
                                        available_models.append(model)
                                
                                # Basic columns
                                basic_columns = ['Date', 'Text', 'Sentiment', 'Sentiment_Score', 'Author', 'Likes', 'Retweets']
                                model_columns = []
                                
                                # Add model columns
                                for model in available_models:
                                    col_name = f"{model}_score"
                                    if col_name in raw_tweets.columns:
                                        # Create nicely formatted column name
                                        raw_tweets[f"{model.capitalize()} Score"] = raw_tweets[col_name]
                                        model_columns.append(f"{model.capitalize()} Score")
                                
                                # Combine all columns
                                all_columns = basic_columns + model_columns
                                
                                # Create display dataframe with all columns
                                display_df = raw_tweets[all_columns].copy()
                                
                                # Format date
                                display_df['Date'] = pd.to_datetime(display_df['Date']).dt.strftime('%Y-%m-%d %H:%M:%S')
                                
                                # Display dataframe
                                st.dataframe(
                                    display_df.sort_values('Date', ascending=False),
                                    use_container_width=True
                                )
                                
                                # Add download button
                                st.download_button(
                                    "Download X Analysis Data",
                                    raw_tweets.to_csv(index=False),
                                    "twitter_analysis.csv",
                                    "text/csv",
                                    key='download-twitter-data'
                                )
                            else:
                                # Show daily aggregated data instead
                                display_df = analysed_data.copy()
                                display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d %H:%M')
                                
                                # Select and reorder columns for display
                                display_columns = [
                                    'Date',
                                    'Tweet_Count',
                                    'Sentiment_Score',
                                    'Sentiment_Std',
                                    'Engagement_Mean',
                                    'Engagement_Total'
                                ]
                                
                                st.dataframe(
                                    display_df[display_columns].sort_values('Date', ascending=False),
                                    use_container_width=True
                                )
                                
                                # Add download button
                                st.download_button(
                                    "Download X Analysis Data",
                                    display_df.to_csv(index=False),
                                    "x_analysis.csv",
                                    "text/csv",
                                    key='download-x-data'
                                )
                    else:
                        st.warning("Could not analyse X content")
                else:
                    st.warning(f"No tweets found for {st.session_state.symbol} in the market data period")
                    
            except Exception as e:
                st.error(f"Error in X analysis: {str(e)}")
                st.exception(e)

def reddit_analysis_tab():
    st.header("Reddit Analysis")
    
    # Add tabs for different analysis views
    reddit_tabs = st.tabs(["Ensemble Analysis", "Model Comparison", "Raw Data"])
    
    if st.button("Fetch Reddit Data"):
        with st.spinner("Fetching Reddit data..."):
            try:
                # Initialize Reddit analyser
                reddit_analyser = RedditAnalyser()
                
                # Get market data date range
                start_date = st.session_state.market_data['Date'].min()
                end_date = st.session_state.market_data['Date'].max()
                
                # Fetch Reddit data
                reddit_data = reddit_analyser.fetch_reddit_data(
                    st.session_state.symbol,
                    start_date,
                    end_date
                )
                
                if reddit_data is not None and not reddit_data.empty:
                    # Initialize sentiment analyzer
                    sentiment_analyser = FinancialSentimentAnalyser()
                    
                    # Analyse sentiment
                    analysed_data = reddit_analyser.analyse_content(reddit_data, sentiment_analyser)
                    
                    if not analysed_data.empty:
                        st.session_state.reddit_data = analysed_data
                        
                        # ENSEMBLE ANALYSIS TAB
                        with reddit_tabs[0]:
                            # Show summary metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(
                                    "Total Posts",
                                    len(analysed_data[analysed_data['Type'] == 'post'])
                                )
                                st.metric(
                                    "Total Comments",
                                    len(analysed_data[analysed_data['Type'] == 'comment'])
                                )
                            
                            with col2:
                                avg_sentiment = analysed_data['Sentiment_Score'].mean()
                                st.metric(
                                    "Average Sentiment",
                                    f"{avg_sentiment:.2f}"
                                )
                                st.metric(
                                    "Positive Posts/Comments",
                                    len(analysed_data[analysed_data['Sentiment'] == 'Positive'])
                                )
                            
                            with col3:
                                st.metric(
                                    "Total Score",
                                    analysed_data['Score'].sum()
                                )
                                st.metric(
                                    "Negative Posts/Comments",
                                    len(analysed_data[analysed_data['Sentiment'] == 'Negative'])
                                )
                            
                            # Display sentiment distribution
                            st.subheader("Reddit Sentiment Distribution")
                            
                            # Create sentiment distribution plot
                            fig = go.Figure()
                            
                            # Add sentiment score distribution
                            fig.add_trace(go.Box(
                                y=analysed_data['Sentiment_Score'],
                                name='Sentiment Distribution',
                                boxpoints='all',
                                jitter=0.3,
                                pointpos=-1.8
                            ))
                            
                            fig.update_layout(
                                title="Sentiment Score Distribution",
                                yaxis_title="Sentiment Score",
                                height=400,
                                showlegend=False,
                                yaxis=dict(
                                    range=[-1, 1],
                                    zeroline=True,
                                    zerolinecolor='rgba(128,128,128,0.4)'
                                )
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display sentiment trend
                            st.subheader("Reddit Sentiment Trend")
                            
                            # Calculate daily sentiment
                            daily_sentiment = (analysed_data
                                .assign(Date=analysed_data['Date'].dt.date)
                                .groupby('Date')
                                .agg({
                                    'Sentiment_Score': 'mean',
                                    'Type': 'count'
                                })
                                .reset_index())
                            
                            # Create trend plot
                            fig = go.Figure()
                            
                            # Add sentiment line
                            fig.add_trace(
                                go.Scatter(
                                    x=daily_sentiment['Date'],
                                    y=daily_sentiment['Sentiment_Score'],
                                    mode='lines+markers',
                                    name='Reddit Sentiment',
                                    line=dict(color='orange', width=2),
                                    marker=dict(size=6)
                                )
                            )
                            
                            # Add post/comment count bars
                            fig.add_trace(
                                go.Bar(
                                    x=daily_sentiment['Date'],
                                    y=daily_sentiment['Type'],
                                    name='Number of Posts/Comments',
                                    yaxis='y2',
                                    marker_color='rgba(255,165,0,0.2)'
                                )
                            )
                            
                            fig.update_layout(
                                title="Reddit Sentiment Over Time",
                                xaxis_title="Date",
                                yaxis=dict(
                                    title="Sentiment Score",
                                    range=[-1, 1],
                                    gridcolor='rgba(128,128,128,0.2)',
                                    zeroline=True,
                                    zerolinecolor='rgba(128,128,128,0.4)'
                                ),
                                yaxis2=dict(
                                    title="Number of Posts/Comments",
                                    overlaying='y',
                                    side='right',
                                    showgrid=False
                                ),
                                height=400,
                                showlegend=True,
                                hovermode='x unified',
                                plot_bgcolor='white'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # MODEL COMPARISON TAB
                        with reddit_tabs[1]:
                            st.subheader("Model Comparison")
                            
                            # Store available models
                            available_models = []
                            for model in ['textblob', 'vader', 'finbert']:
                                if f'{model}_score' in analysed_data.columns:
                                    available_models.append(model)
                            
                            if available_models:
                                # Show metrics for each model
                                model_metrics = []
                                for model in available_models:
                                    metrics = {
                                        'Model': model.capitalize(),
                                        'Average Score': analysed_data[f'{model}_score'].mean(),
                                        'Std Dev': analysed_data[f'{model}_score'].std(),
                                        'Positive %': len(analysed_data[analysed_data[f'{model}_sentiment'] == 'Positive']) / len(analysed_data) * 100,
                                        'Negative %': len(analysed_data[analysed_data[f'{model}_sentiment'] == 'Negative']) / len(analysed_data) * 100,
                                        'Neutral %': len(analysed_data[analysed_data[f'{model}_sentiment'] == 'Neutral']) / len(analysed_data) * 100
                                    }
                                    model_metrics.append(metrics)
                                
                                # Add ensemble metrics
                                model_metrics.append({
                                    'Model': 'Ensemble',
                                    'Average Score': analysed_data['Sentiment_Score'].mean(),
                                    'Std Dev': analysed_data['Sentiment_Score'].std(),
                                    'Positive %': len(analysed_data[analysed_data['Sentiment'] == 'Positive']) / len(analysed_data) * 100,
                                    'Negative %': len(analysed_data[analysed_data['Sentiment'] == 'Negative']) / len(analysed_data) * 100,
                                    'Neutral %': len(analysed_data[analysed_data['Sentiment'] == 'Neutral']) / len(analysed_data) * 100
                                })
                                
                                # Display metrics table
                                metrics_df = pd.DataFrame(model_metrics)
                                metrics_df = metrics_df.set_index('Model')
                                st.dataframe(metrics_df.style.format({
                                    'Average Score': '{:.3f}',
                                    'Std Dev': '{:.3f}',
                                    'Positive %': '{:.1f}%',
                                    'Negative %': '{:.1f}%',
                                    'Neutral %': '{:.1f}%'
                                }))
                                
                                # Try to plot model comparison
                                try:
                                    st.subheader("Model Score Comparison")
                                    comp_fig = plot_model_comparison(analysed_data, available_models)
                                    st.plotly_chart(comp_fig, use_container_width=True)
                                    
                                    st.subheader("Model Correlation Matrix")
                                    corr_fig = plot_model_correlation_matrix(analysed_data, available_models)
                                    st.plotly_chart(corr_fig)
                                except Exception as e:
                                    st.warning(f"Could not generate model comparison plots: {str(e)}")
                            else:
                                st.warning("No model-specific data available for comparison.")
                        
                        # RAW DATA TAB
                        with reddit_tabs[2]:
                            st.subheader("Reddit Activity")
                            
                            # Basic columns
                            basic_columns = ['Date', 'Type', 'Text', 'Sentiment', 'Sentiment_Score', 'Score', 'URL']
                            model_columns = []
                            
                            # Get model columns
                            for model in available_models:
                                col_name = f"{model}_score"
                                if col_name in analysed_data.columns:
                                    # Create nicely formatted column name
                                    analysed_data[f"{model.capitalize()} Score"] = analysed_data[col_name]
                                    model_columns.append(f"{model.capitalize()} Score")
                            
                            # Combine all columns
                            all_columns = basic_columns + model_columns
                            
                            # Create display dataframe with all columns
                            display_df = analysed_data[all_columns].copy()
                            
                            # Format date
                            display_df['Date'] = pd.to_datetime(display_df['Date']).dt.strftime('%Y-%m-%d %H:%M:%S')
                            
                            # Display dataframe
                            st.dataframe(
                                display_df.sort_values('Date', ascending=False),
                                use_container_width=True
                            )
                            
                            # Add download button
                            st.download_button(
                                "Download Reddit Analysis Data",
                                analysed_data.to_csv(index=False),
                                "reddit_analysis.csv",
                                "text/csv",
                                key='download-reddit-data'
                            )
                    else:
                        st.warning("Could not analyse Reddit content")
                else:
                    st.warning("No Reddit data found for the specified period")
                    
            except Exception as e:
                st.error(f"Error in Reddit analysis: {str(e)}")
                st.exception(e)

def sentiment_analysis_page():
    st.title("🔍 Sentiment Analysis")
    
    if 'market_data' not in st.session_state or st.session_state.market_data is None:
        st.warning("Please fetch market data first in the Market Data page")
        return
    
    # Initialize sentiment analyzer immediately
    sentiment_analyser = FinancialSentimentAnalyser()
    
    # Display model information in sidebar right away
    display_model_info(sentiment_analyser)
    
    tabs = st.tabs(["EODHD News", "X (Twitter)", "Reddit Analysis"])
    
    with tabs[0]:
        news_analysis_tab()
    
    with tabs[1]:
        x_analysis_tab()
    
    with tabs[2]:
        reddit_analysis_tab()

if __name__ == "__main__":
    sentiment_analysis_page()