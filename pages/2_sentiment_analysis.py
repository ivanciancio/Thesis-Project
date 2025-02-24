import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.api_helpers import fetch_news_data
from analysers.sentiment_analyser import FinancialSentimentAnalyser
from analysers.reddit_analyser import RedditAnalyser
from analysers.x_analyser import XAnalyser
from datetime import datetime, timedelta


def plot_daily_sentiment(news_df):
    """Create daily sentiment visualisation"""
    # Ensure date column is datetime and strip timezone
    news_df['Date'] = pd.to_datetime(news_df['Date']).dt.tz_localize(None)
    
    # Extract the hour from the datetime for more granular visualisation
    news_df['Hour'] = news_df['Date'].dt.hour
    
    # Group by date and hour for more granular view
    hourly_sentiment = (news_df
        .groupby([news_df['Date'].dt.date, 'Hour'])
        .agg({
            'Sentiment Score': 'mean',
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
            y=hourly_sentiment['Sentiment Score'],
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
            'Sentiment Score': 'mean',
            'Title': 'count'
        }).reset_index()
        
        fig.add_trace(
            go.Scatter(
                x=daily_news['Date'],
                y=daily_news['Sentiment Score'],
                mode='lines+markers',
                name='News Sentiment',
                line=dict(color='blue')
            )
        )
    
    # Add Reddit sentiment if available
    if reddit_data is not None and not reddit_data.empty:
        daily_reddit = reddit_data.groupby(reddit_data['Date'].dt.date).agg({
            'Sentiment Score': 'mean',
            'Text': 'count'
        }).reset_index()
        
        fig.add_trace(
            go.Scatter(
                x=daily_reddit['Date'],
                y=daily_reddit['Sentiment Score'],
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
                    sentiment_analyser = FinancialSentimentAnalyser()
                    
                    # Process each news item
                    analysed_news = []
                    for _, row in news_data.iterrows():
                        text = f"{row['Title']} {row.get('Text', '')}"
                        sentiment_result = sentiment_analyser.analyse_sentiment(text)
                        analysed_news.append(sentiment_result)
                    
                    # Create DataFrame with analysis results
                    news_df = pd.DataFrame({
                        'Date': news_data['Date'],
                        'Title': news_data['Title'],
                        'Sentiment Score': [result['score'] for result in analysed_news],
                        'Sentiment': [result['sentiment'] for result in analysed_news]
                    })
                    
                    news_df['Date'] = pd.to_datetime(news_df['Date']).dt.tz_localize(None)
                    st.session_state.news_data = news_df
                    
                    # Display news sentiment analysis
                    st.subheader("News Sentiment Analysis")
                    
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
                    
                    # Display the news data table
                    st.subheader("News Articles")
                    display_df = news_df[['Date', 'Title', 'Sentiment', 'Sentiment Score']].copy()
                    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
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

def x_analysis_tab():
    st.header("X (Twitter) Analysis")
    
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
                    sentiment_analyser = FinancialSentimentAnalyser()
                    analysed_data = x_analyser.analyse_content(twitter_data, sentiment_analyser)
                    
                    if not analysed_data.empty:
                        # Save hourly aggregated data for visualization
                        st.session_state.twitter_data = analysed_data
                        
                        # Make raw tweet data available for correlation analysis (if created by analyse_content)
                        if 'twitter_raw_data' in st.session_state and not st.session_state.twitter_raw_data.empty:
                            st.success("✅ Raw tweet data also saved for correlation analysis")
                        
                        st.write("Twitter Data Summary:")
                        st.write("- Shape:", analysed_data.shape)
                        st.write("- Date range:", analysed_data['Date'].min(), "to", analysed_data['Date'].max())
                        
                        # Display X sentiment analysis
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
                            # Calculate correlation between sentiment and engagement
                            engagement_corr = analysed_data['Sentiment_Score'].corr(
                                analysed_data['Engagement_Mean']
                            )
                            st.metric(
                                "Sentiment-Engagement Correlation",
                                f"{engagement_corr:.2f}"
                            )
                        
                        with eng_col2:
                            # Daily statistics
                            st.write("Daily Statistics:")
                            daily_stats = analysed_data[[
                                'Sentiment_Score',
                                'Engagement_Mean',
                                'Tweet_Count'
                            ]].describe()
                            st.dataframe(daily_stats)
                        
                        # Display the daily aggregated data
                        st.subheader("Daily Twitter Activity")
                        display_df = analysed_data.copy()
                        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                        
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
    
    if st.button("Fetch Reddit Data"):
        with st.spinner("Fetching Reddit data..."):
            try:
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
                    # Analyse sentiment
                    sentiment_analyser = FinancialSentimentAnalyser()
                    analysed_data = reddit_analyser.analyse_content(reddit_data, sentiment_analyser)
                    
                    if not analysed_data.empty:
                        st.session_state.reddit_data = analysed_data
                        
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
                            avg_sentiment = analysed_data['Sentiment_Score'].mean()  # Changed from 'Sentiment Score'
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
                            y=analysed_data['Sentiment_Score'],  # Changed from 'Sentiment Score'
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
                                'Sentiment_Score': 'mean',  # Changed from 'Sentiment Score'
                                'Type': 'count'
                            })
                            .reset_index())
                        
                        # Create trend plot
                        fig = go.Figure()
                        
                        # Add sentiment line
                        fig.add_trace(
                            go.Scatter(
                                x=daily_sentiment['Date'],
                                y=daily_sentiment['Sentiment_Score'],  # Changed from 'Sentiment Score'
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
                        
                        # Display recent posts/comments
                        st.subheader("Recent Reddit Activity")
                        display_df = analysed_data[['Date', 'Type', 'Text', 'Sentiment', 'Sentiment_Score', 'Score', 'URL']].copy()  # Changed from 'Sentiment Score'
                        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
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
                st.exception(e)  # Show the full traceback

def sentiment_analysis_page():
    st.title("🔍 Sentiment Analysis")
    
    if 'market_data' not in st.session_state or st.session_state.market_data is None:
        st.warning("Please fetch market data first in the Market Data page")
        return
    
    tabs = st.tabs(["EODHD News", "X (Twitter)", "Reddit Analysis"])
    
    with tabs[0]:
        news_analysis_tab()
    
    with tabs[1]:
        x_analysis_tab()
    
    with tabs[2]:
        reddit_analysis_tab()

if __name__ == "__main__":
    sentiment_analysis_page()