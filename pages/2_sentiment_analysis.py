import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.api_helpers import fetch_news_data
from analyzers.sentiment_analyzer import FinancialSentimentAnalyzer
from analyzers.reddit_analyzer import RedditAnalyzer

def plot_daily_sentiment(news_df):
    """Create daily sentiment visualization"""
    # Convert dates to date (without time) and calculate daily averages
    daily_sentiment = (news_df
        .assign(Date=pd.to_datetime(news_df['Date']).dt.date)
        .groupby('Date')
        .agg({
            'Sentiment Score': 'mean',
            'Title': 'count'
        })
        .reset_index())
    
    # Create figure
    fig = go.Figure()
    
    # Add sentiment score line
    fig.add_trace(
        go.Scatter(
            x=daily_sentiment['Date'],
            y=daily_sentiment['Sentiment Score'],
            mode='lines+markers',
            name='News Sentiment',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        )
    )
    
    # Add article count bars
    fig.add_trace(
        go.Bar(
            x=daily_sentiment['Date'],
            y=daily_sentiment['Title'],
            name='Number of Articles',
            yaxis='y2',
            marker_color='rgba(0,0,255,0.2)'
        )
    )
    
    # Update layout
    fig.update_layout(
        title="Sentiment Analysis Over Time",
        xaxis_title="Date",
        yaxis=dict(
            title="Sentiment Score",
            range=[-1, 1],
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=True,
            zerolinecolor='rgba(128,128,128,0.4)'
        ),
        yaxis2=dict(
            title="Number of Articles",
            overlaying='y',
            side='right',
            showgrid=False
        ),
        height=500,
        showlegend=True,
        hovermode='x unified',
        plot_bgcolor='white'
    )
    
    return fig

def plot_sentiment_comparison(news_data, reddit_data=None):
    """Create sentiment comparison visualization"""
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
                    api_key=st.session_state.eodhd_api_key,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if news_data is not None and not news_data.empty:
                    sentiment_analyzer = FinancialSentimentAnalyzer()
                    
                    # Process each news item
                    analyzed_news = []
                    for _, row in news_data.iterrows():
                        text = f"{row['Title']} {row.get('Text', '')}"
                        sentiment_result = sentiment_analyzer.analyze_sentiment(text)
                        analyzed_news.append(sentiment_result)
                    
                    # Create DataFrame with analysis results
                    news_df = pd.DataFrame({
                        'Date': news_data['Date'],
                        'Title': news_data['Title'],
                        'Sentiment Score': [result['score'] for result in analyzed_news],
                        'Sentiment': [result['sentiment'] for result in analyzed_news]
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

def reddit_analysis_tab():
    st.header("Reddit Analysis")
    
    if not (st.session_state.get('reddit_client_id') and st.session_state.get('reddit_client_secret')):
        st.warning("Please provide Reddit API credentials in the sidebar")
        return
        
    if st.button("Fetch Reddit Data"):
        with st.spinner("Fetching Reddit data..."):
            try:
                reddit_analyzer = RedditAnalyzer(
                    st.session_state.reddit_client_id,
                    st.session_state.reddit_client_secret
                )
                
                # Get market data date range
                start_date = st.session_state.market_data['Date'].min()
                end_date = st.session_state.market_data['Date'].max()
                
                # Fetch Reddit data
                reddit_data = reddit_analyzer.fetch_reddit_data(
                    st.session_state.symbol,
                    start_date,
                    end_date
                )
                
                if reddit_data is not None and not reddit_data.empty:
                    # Analyze sentiment
                    sentiment_analyzer = FinancialSentimentAnalyzer()
                    analyzed_data = reddit_analyzer.analyze_content(reddit_data, sentiment_analyzer)
                    
                    if not analyzed_data.empty:
                        st.session_state.reddit_data = analyzed_data
                        
                        # Show summary metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "Total Posts",
                                len(analyzed_data[analyzed_data['Type'] == 'post'])
                            )
                            st.metric(
                                "Total Comments",
                                len(analyzed_data[analyzed_data['Type'] == 'comment'])
                            )
                        
                        with col2:
                            avg_sentiment = analyzed_data['Sentiment Score'].mean()
                            st.metric(
                                "Average Sentiment",
                                f"{avg_sentiment:.2f}"
                            )
                            st.metric(
                                "Positive Posts/Comments",
                                len(analyzed_data[analyzed_data['Sentiment'] == 'Positive'])
                            )
                        
                        with col3:
                            st.metric(
                                "Total Score",
                                analyzed_data['Score'].sum()
                            )
                            st.metric(
                                "Negative Posts/Comments",
                                len(analyzed_data[analyzed_data['Sentiment'] == 'Negative'])
                            )
                        
                        # Display sentiment distribution
                        st.subheader("Reddit Sentiment Distribution")
                        
                        # Create sentiment distribution plot
                        fig = go.Figure()
                        
                        # Add sentiment score distribution
                        fig.add_trace(go.Box(
                            y=analyzed_data['Sentiment Score'],
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
                        daily_sentiment = (analyzed_data
                            .assign(Date=analyzed_data['Date'].dt.date)
                            .groupby('Date')
                            .agg({
                                'Sentiment Score': 'mean',
                                'Type': 'count'
                            })
                            .reset_index())
                        
                        # Create trend plot
                        fig = go.Figure()
                        
                        # Add sentiment line
                        fig.add_trace(
                            go.Scatter(
                                x=daily_sentiment['Date'],
                                y=daily_sentiment['Sentiment Score'],
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
                        display_df = analyzed_data[['Date', 'Type', 'Text', 'Sentiment', 'Score', 'URL']].copy()
                        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
                        st.dataframe(
                            display_df.sort_values('Date', ascending=False),
                            use_container_width=True
                        )
                        
                        # Add download button
                        st.download_button(
                            "Download Reddit Analysis Data",
                            analyzed_data.to_csv(index=False),
                            "reddit_analysis.csv",
                            "text/csv",
                            key='download-reddit-data'
                        )
                    else:
                        st.warning("Could not analyze Reddit content")
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
    
    tabs = st.tabs(["EODHD News", "Reddit Analysis"])
    
    with tabs[0]:
        news_analysis_tab()
    
    with tabs[1]:
        reddit_analysis_tab()

if __name__ == "__main__":
    sentiment_analysis_page()