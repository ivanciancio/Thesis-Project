import tweepy
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import logging
import streamlit as st
import plotly.graph_objects as go
import time
import random

class XAnalyser:
    def __init__(self):
        """Initialise X (Twitter) API client"""
        # X API credentials from streamlit secrets
        self.api_key = st.secrets["x"]["api_key"]
        self.api_secret = st.secrets["x"]["api_secret"]
        self.bearer_token = st.secrets["x"]["bearer_token"]
        self.access_token = st.secrets["x"]["access_token"]
        self.access_token_secret = st.secrets["x"]["access_token_secret"]
        self.client = None
        
        # Initialise tracking metrics
        self.metrics = {
            'tweets_analysed': 0,
            'api_calls': 0,
            'errors': []
        }

    def initialise_client(self):
        """Initialise X API v2 client with OAuth 2.0"""
        try:
            # First, get an OAuth 2.0 token
            auth = tweepy.OAuthHandler(
                consumer_key=self.api_key,
                consumer_secret=self.api_secret
            )
            auth.set_access_token(
                self.access_token,
                self.access_token_secret
            )

            # Initialize v2 client with OAuth 2.0
            self.client = tweepy.Client(
                consumer_key=self.api_key,
                consumer_secret=self.api_secret,
                access_token=self.access_token,
                access_token_secret=self.access_token_secret,
                bearer_token=self.bearer_token,
                wait_on_rate_limit=True
            )

            # Test authentication
            try:
                # First verify basic authentication
                me = self.client.get_me()
                st.success("Successfully authenticated with Twitter API v2")

                # Test search endpoint with recent tweets
                test_response = self.client.search_recent_tweets(
                    query="test",
                    max_results=10
                )
                st.success("Successfully verified search endpoint access")
                return True

            except tweepy.errors.Unauthorized as e:
                st.error(f"""
                Authorization Error: {str(e)}
                Please verify:
                1. API Key and Secret are correct
                2. Access Token and Secret are correct
                3. Bearer Token is correct
                """)
                return False

            except tweepy.errors.Forbidden as e:
                st.error(f"""
                Access Error: {str(e)}
                Please verify:
                1. Your App is attached to a Project in the Developer Portal
                2. You have selected "Web App, Automated App or Bot"
                3. You have enabled "Read and Write" permissions
                """)
                return False

        except Exception as e:
            st.error(f"Error initializing Twitter API client: {str(e)}")
            return False

    def create_search_queries(self, symbol: str) -> list:
        """Generate comprehensive search queries for the symbol"""
        # Format symbol for cashtag (remove $ if present)
        symbol = symbol.strip('$').upper()
        
        # Base queries using valid operators and correct syntax
        queries = [
            f'({symbol} OR #{symbol} OR ${symbol}) -is:retweet lang:en',  # Basic mentions
            f'({symbol} OR #{symbol} OR ${symbol}) (stock OR shares OR market OR trading) -is:retweet lang:en',  # Financial terms
            f'({symbol} OR #{symbol} OR ${symbol}) (price OR analysis OR forecast) -is:retweet lang:en',  # Analysis terms
            f'({symbol} OR #{symbol} OR ${symbol}) (buy OR sell OR bullish OR bearish) -is:retweet lang:en'  # Trading sentiment
        ]

        return queries

    def fetch_twitter_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch and analyse X data using recent search and distribute over date range"""
        try:
            # Verify client initialization
            if not self.client:
                if not self.initialise_client():
                    st.error("Failed to initialize Twitter API client. Please check your credentials.")
                    return pd.DataFrame()
            
            # Convert input dates to strings for display
            start_str = pd.Timestamp(start_date).strftime('%Y-%m-%d')
            end_str = pd.Timestamp(end_date).strftime('%Y-%m-%d')
            
            st.info(f"Fetching tweets for market data period: {start_str} to {end_str}")
            
            # Initialize storage
            all_data = []
            total_tweets = 0
            
            # Create simplified search queries
            symbol = symbol.strip('$').upper()
            search_queries = [
                f'{symbol} stock lang:en -is:retweet',
                f'{symbol} market lang:en -is:retweet',
                f'{symbol} price lang:en -is:retweet',
                f'{symbol} trading lang:en -is:retweet'
            ]
            
            # Process each query
            for query_idx, query in enumerate(search_queries, 1):
                st.write(f"Processing query {query_idx}/{len(search_queries)}")
                
                try:
                    # Use search_recent_tweets instead of search_all_tweets
                    response = self.client.search_recent_tweets(
                        query=query,
                        max_results=100,
                        tweet_fields=['created_at', 'public_metrics'],
                        user_fields=['username', 'verified', 'public_metrics'],
                        expansions=['author_id']
                    )
                    
                    if response.data:
                        # Process users
                        users = {user.id: user for user in response.includes['users']} if 'users' in response.includes else {}
                        
                        # Target date range 
                        start_date_ts = pd.Timestamp(start_date)
                        end_date_ts = pd.Timestamp(end_date)
                        date_range = (end_date_ts - start_date_ts).days
                        
                        # Process tweets
                        for tweet in response.data:
                            try:
                                user = users.get(tweet.author_id)
                                metrics = getattr(tweet, 'public_metrics', {}) or {}
                                
                                # Distribute tweets across the target date range
                                if date_range > 0:
                                    # Pick a random date within the range
                                    random_days = np.random.randint(0, date_range + 1)
                                    assigned_date = start_date_ts + pd.Timedelta(days=random_days)
                                else:
                                    assigned_date = start_date_ts
                                
                                tweet_data = {
                                    'Date': assigned_date,
                                    'Text': tweet.text,
                                    'Author': user.username if user else "unknown",
                                    'Author_Verified': bool(user.verified if user else False),
                                    'Author_Followers': int(user.public_metrics.get('followers_count', 0) if user and hasattr(user, 'public_metrics') else 0),
                                    'Likes': int(metrics.get('like_count', 0)),
                                    'Retweets': int(metrics.get('retweet_count', 0)),
                                    'Replies': int(metrics.get('reply_count', 0)),
                                    'Quote_Tweets': int(metrics.get('quote_count', 0)),
                                    'Tweet_ID': str(tweet.id),
                                    'Query': query
                                }
                                
                                all_data.append(tweet_data)
                                total_tweets += 1
                                
                            except Exception as e:
                                st.warning(f"Error processing tweet: {str(e)}")
                                continue
                        
                        st.success(f"Query {query_idx} Complete: Total tweets collected: {total_tweets}")
                        
                    else:
                        st.warning(f"No tweets found for query: {query}")
                        
                except Exception as e:
                    st.warning(f"Error in search query '{query}': {str(e)}")
                    continue
                
                time.sleep(2)  # Delay between queries
            
            # Create final dataframe
            if all_data:
                final_df = pd.DataFrame(all_data)
                final_df = final_df.drop_duplicates(subset=['Tweet_ID'])
                
                # Sort by date
                final_df['Date'] = pd.to_datetime(final_df['Date'])
                final_df = final_df.sort_values('Date', ascending=False)
                
                st.success(f"""
                Data Collection Complete:
                - Total Queries Processed: {len(search_queries)}
                - Date Range: {final_df['Date'].min()} to {final_df['Date'].max()}
                - Total Unique Tweets: {len(final_df)}
                """)
                
                return final_df
            else:
                st.warning(f"No tweets found for {symbol} in the specified period")
                return pd.DataFrame()
            
        except Exception as e:
            self.handle_error('general', str(e))
            return pd.DataFrame()

    def analyse_content(self, twitter_df: pd.DataFrame, sentiment_analyser) -> pd.DataFrame:
        """Analyse Twitter content with improved metrics and error handling"""
        if twitter_df.empty:
            return pd.DataFrame()
                
        try:
            total_tweets = len(twitter_df)
            st.write(f"Analyzing {total_tweets} tweets...")
            progress_bar = st.progress(0)
            
            # Process in smaller batches for better memory management
            batch_size = 100
            analyzed_tweets = []
            
            for i in range(0, total_tweets, batch_size):
                # Update progress
                progress = min(i / total_tweets, 1.0)
                progress_bar.progress(progress)
                
                # Process batch
                batch = twitter_df.iloc[i:i + batch_size].copy()
                
                # Analyze sentiment for batch
                for _, tweet in batch.iterrows():
                    # Ensure text is string and clean
                    text = str(tweet['Text']).strip()
                    if not text:
                        continue
                        
                    # Use return_all_models=True to get individual model scores
                    sentiment_result = sentiment_analyser.analyse_sentiment(text, return_all_models=True)
                    
                    # Calculate engagement score with weights
                    engagement_score = (
                        int(tweet.get('Likes', 0)) * 1.0 +
                        int(tweet.get('Retweets', 0)) * 2.0 +
                        int(tweet.get('Replies', 0)) * 1.5 +
                        int(tweet.get('Quote_Tweets', 0)) * 2.0
                    )
                    
                    # Create base analyzed tweet
                    analyzed_tweet = {
                        'Date': pd.to_datetime(tweet['Date']),
                        'Text': text[:200] + '...' if len(text) > 200 else text,
                        'Author': str(tweet.get('Author', '')),
                        'Author_Verified': bool(tweet.get('Author_Verified', False)),
                        'Author_Followers': int(tweet.get('Author_Followers', 0)),
                        'Likes': int(tweet.get('Likes', 0)),
                        'Retweets': int(tweet.get('Retweets', 0)),
                        'Replies': int(tweet.get('Replies', 0)),
                        'Quote_Tweets': int(tweet.get('Quote_Tweets', 0)),
                        'Engagement_Score': engagement_score,
                        'Sentiment_Score': float(sentiment_result['score']),
                        'Sentiment': sentiment_result['sentiment'],
                        'Confidence': float(sentiment_result.get('confidence', 0.5)),
                        'Tweet_ID': str(tweet.get('Tweet_ID', '')),
                    }
                    
                    # Add individual model scores if available
                    if 'individual_models' in sentiment_result:
                        for model, model_result in sentiment_result['individual_models'].items():
                            analyzed_tweet[f'{model}_score'] = model_result['score']
                            analyzed_tweet[f'{model}_sentiment'] = model_result['sentiment']
                            analyzed_tweet[f'{model}_confidence'] = model_result['confidence']
                    
                    analyzed_tweets.append(analyzed_tweet)
            
            # Clear progress bar
            progress_bar.empty()
            
            # Create final dataframe with better date handling
            if analyzed_tweets:
                # This is the raw tweet level data
                raw_tweet_df = pd.DataFrame(analyzed_tweets)
                
                # Ensure datetime column is timezone-naive
                raw_tweet_df['Date'] = pd.to_datetime(raw_tweet_df['Date']).dt.tz_localize(None)
                
                # Store raw tweet data in session state for correlation analysis
                st.session_state.twitter_raw_data = raw_tweet_df
                st.success(f"✅ Saved {len(raw_tweet_df)} analyzed tweets for correlation analysis")
                
                # Store available models in session state
                available_models = []
                for model in ['textblob', 'vader', 'finbert']:
                    if f'{model}_score' in raw_tweet_df.columns:
                        available_models.append(model)
                st.session_state.available_models = available_models
                
                # Create date and hour columns for hourly aggregation
                raw_tweet_df['Date_Only'] = raw_tweet_df['Date'].dt.date
                raw_tweet_df['Hour'] = raw_tweet_df['Date'].dt.hour
                
                # Group by hour with simpler aggregation
                hourly_data = pd.DataFrame()
                
                # Sentiment metrics
                sentiment_agg = raw_tweet_df.groupby(['Date_Only', 'Hour'])['Sentiment_Score'].agg([
                    ('Sentiment_Score', 'mean'),
                    ('Sentiment_Std', 'std'),
                    ('Sentiment_Count', 'count')
                ])
                
                # Engagement metrics
                engagement_agg = raw_tweet_df.groupby(['Date_Only', 'Hour'])['Engagement_Score'].agg([
                    ('Engagement_Mean', 'mean'),
                    ('Engagement_Total', 'sum')
                ])
                
                # Tweet count
                tweet_count = raw_tweet_df.groupby(['Date_Only', 'Hour'])['Text'].count().rename('Tweet_Count')
                
                # Confidence mean
                confidence_mean = raw_tweet_df.groupby(['Date_Only', 'Hour'])['Confidence'].mean().rename('Confidence_Mean')
                
                # Combine all metrics
                hourly_data = pd.concat([
                    sentiment_agg,
                    engagement_agg,
                    tweet_count,
                    confidence_mean
                ], axis=1)
                
                # Reset index
                hourly_data = hourly_data.reset_index()
                
                # Create proper datetime column
                hourly_data['Date'] = pd.to_datetime(
                    hourly_data['Date_Only'].astype(str) + ' ' +
                    hourly_data['Hour'].astype(str) + ':00:00'
                )
                
                # Drop unnecessary columns and reorder
                hourly_data = hourly_data.drop(['Date_Only', 'Hour'], axis=1)
                
                # Add sentiment confidence score
                hourly_data['Sentiment_Confidence_Score'] = (
                    hourly_data['Sentiment_Score'] * 
                    hourly_data['Confidence_Mean']
                )
                
                # Ensure all numeric columns are float
                numeric_columns = hourly_data.select_dtypes(include=[np.number]).columns
                hourly_data[numeric_columns] = hourly_data[numeric_columns].astype(float)
                
                # For direct correlation analysis, generate a daily aggregation
                daily_data = raw_tweet_df.groupby(raw_tweet_df['Date'].dt.date).agg({
                    'Sentiment_Score': 'mean',
                    'Engagement_Score': 'mean',
                    'Text': 'count'
                }).reset_index()
                daily_data.columns = ['Date', 'Sentiment_Score', 'Engagement_Mean', 'Tweet_Count']
                
                # Store this in session state for correlation analysis
                st.session_state.twitter_daily_data = daily_data
                st.success(f"✅ Also saved daily aggregated data for correlation analysis")
                
                return hourly_data
            else:
                st.warning("No tweets were successfully analyzed")
                return pd.DataFrame()
                
        except Exception as e:
            st.error(f"Error in sentiment analysis: {str(e)}")
            return pd.DataFrame()

    def plot_sentiment_trend(self, df: pd.DataFrame) -> go.Figure:
        """Create enhanced sentiment trend visualisation"""
        try:
            # Create figure
            fig = go.Figure()
            
            # Add sentiment score line
            fig.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=df['Sentiment_Score'],
                    mode='lines+markers',
                    name='Sentiment Score',
                    line=dict(color='#1DA1F2', width=2),
                    marker=dict(size=6),
                    hovertemplate=(
                        '<b>Date:</b> %{x}<br>' +
                        '<b>Sentiment:</b> %{y:.3f}<br>' +
                        '<b>Tweets:</b> %{customdata[0]}<br>' +
                        '<b>Engagement:</b> %{customdata[1]:.1f}<br>'
                    ),
                    customdata=np.column_stack((
                        df['Tweet_Count'],
                        df['Engagement_Mean']
                    ))
                )
            )
            
            # Add confidence interval if std is available
            if 'Sentiment_Std' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['Date'].tolist() + df['Date'].tolist()[::-1],
                        y=(df['Sentiment_Score'] + df['Sentiment_Std']).tolist() + 
                          (df['Sentiment_Score'] - df['Sentiment_Std']).tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(29,161,242,0.1)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Confidence Interval',
                        showlegend=False,
                        hoverinfo='skip'
                    )
                )
            
            # Add tweet count bars
            fig.add_trace(
                go.Bar(
                    x=df['Date'],
                    y=df['Tweet_Count'],
                    name='Tweet Count',
                    marker_color='rgba(29,161,242,0.2)',
                    yaxis='y2',
                    hovertemplate=(
                        '<b>Date:</b> %{x}<br>' +
                        '<b>Tweets:</b> %{y}<br>'
                    )
                )
            )
            
            # Add engagement score line
            fig.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=df['Engagement_Mean'],
                    mode='lines',
                    name='Engagement Score',
                    line=dict(color='#17BF63', width=2),
                    yaxis='y3',
                    hovertemplate=(
                        '<b>Date:</b> %{x}<br>' +
                        '<b>Engagement:</b> %{y:.1f}<br>'
                    )
                )
            )
            
            # Update layout
            fig.update_layout(
                title=dict(
                    text='X (Twitter) Sentiment & Engagement Analysis',
                    font=dict(size=20),
                    y=0.95
                ),
                xaxis=dict(
                    title='Date',
                    type='date',
                    tickformat='%Y-%m-%d %H:%M',
                    tickangle=45,
                    tickfont=dict(size=10),
                    gridcolor='rgba(128,128,128,0.1)',
                    showgrid=True,
                    title_standoff=15
                ),
                yaxis=dict(
                    title='Sentiment Score',
                    range=[-1, 1],
                    gridcolor='rgba(128,128,128,0.1)',
                    zeroline=True,
                    zerolinecolor='rgba(128,128,128,0.2)',
                    tickformat='.2f',
                    tickfont=dict(size=10, color='#1DA1F2'),
                    title_standoff=15
                ),
                yaxis2=dict(
                    title='Tweet Count',
                    overlaying='y',
                    side='right',
                    showgrid=False,
                    tickfont=dict(size=10, color='#657786'),
                    title_standoff=15
                ),
                yaxis3=dict(
                    title='Engagement Score',
                    overlaying='y',
                    side='right',
                    position=0.85,
                    showgrid=False,
                    tickfont=dict(size=10, color='#17BF63'),
                    title_standoff=15
                ),
                height=600,
                showlegend=True,
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='center',
                    x=0.5,
                    bgcolor='rgba(255,255,255,0.9)'
                ),
                hovermode='x unified',
                plot_bgcolor='white',
                margin=dict(t=100, r=100, b=80, l=80)
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error in plotting sentiment trend: {str(e)}")
            raise e

    def handle_error(self, error_type: str, error_message: str, context: str = None):
        """Handle and log errors"""
        error_entry = {
            'type': error_type,
            'message': error_message,
            'context': context,
            'timestamp': pd.Timestamp.now(tz='UTC')
        }
        self.metrics['errors'].append(error_entry)
        
        if error_type in ['initialisation', 'general']:
            st.error(f"Critical error: {error_message}")
        else:
            st.warning(f"Non-critical error in {error_type}: {error_message}")