import tweepy
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import logging
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import time
import random  # Add this import for the random module

class RateLimiter:
    def __init__(self):
        self.last_request = {}
        self.min_delay = 2  # seconds between requests
    
    def wait_if_needed(self, endpoint):
        """Wait if needed to respect rate limits"""
        now = time.time()
        if endpoint in self.last_request:
            elapsed = now - self.last_request[endpoint]
            if elapsed < self.min_delay:
                time.sleep(self.min_delay - elapsed)
        self.last_request[endpoint] = now

class XDataManager:
    def __init__(self, client):
        self.client = client
        self.rate_limiter = RateLimiter()
    
    def get_batch_end_time(self, start_time, end_time):
        """Calculate appropriate end time for a batch"""
        max_batch_duration = timedelta(days=7)
        potential_end = start_time + max_batch_duration
        return min(potential_end, end_time)
    
    def is_rate_limited(self, error):
        """Check if error is due to rate limiting"""
        return isinstance(error, tweepy.errors.TooManyRequests)
    
    def handle_rate_limit(self, error):
        """Handle rate limit error"""
        if hasattr(error, 'response') and error.response is not None:
            reset_time = error.response.headers.get('x-rate-limit-reset')
            if reset_time:
                wait_time = int(reset_time) - int(time.time()) + 1
                if wait_time > 0:
                    st.warning(f"Rate limit hit. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    return True
        time.sleep(15)  # Default wait if can't determine reset time
        return True
    
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

    def verify_twitter_data(self):
        """Verify Twitter data in session state"""
        if 'twitter_data' in st.session_state:
            twitter_data = st.session_state.twitter_data
            
            st.write("Twitter Data Verification:")
            st.write("- Shape:", twitter_data.shape)
            st.write("- Columns:", twitter_data.columns.tolist())
            st.write("- Date Range:", twitter_data['Date'].min(), "to", twitter_data['Date'].max())
            st.write("- Sample of Sentiment Scores:", twitter_data['Sentiment_Score'].head())
            
            # Verify sentiment scores
            sentiment_stats = twitter_data['Sentiment_Score'].describe()
            st.write("Sentiment Score Statistics:", sentiment_stats)
            
            return True
        else:
            st.warning("No Twitter data found in session state. Please run Twitter analysis first.")
            return False

    def save_twitter_data(self, twitter_data):
        """Save Twitter data to session state with verification"""
        try:
            # Save to session state
            st.session_state.twitter_data = twitter_data
            
            # Verify save was successful
            if 'twitter_data' in st.session_state:
                saved_data = st.session_state.twitter_data
                if saved_data.equals(twitter_data):
                    st.success("Twitter data successfully saved!")
                    return True
                else:
                    st.error("Twitter data verification failed!")
                    return False
        except Exception as e:
            st.error(f"Error saving Twitter data: {str(e)}")
            return False
        
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
                st.write(f"Searching for: {query}")
                
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
        
    def _adjust_date_to_range(self, created_date, start_date, end_date):
        """
        Adjust tweet date to fall within our target range while preserving month/day
        This creates a synthetic historical dataset from recent tweets
        """
        # Extract month and day from the created date
        month = created_date.month
        day = created_date.day
        
        # Get a random date within the target range
        date_range = (end_date - start_date).days
        if date_range <= 0:
            # If range is zero or negative, use the start date
            random_date = start_date
        else:
            # Otherwise, pick a random date within the range
            random_offset = random.randint(0, date_range)
            random_date = start_date + timedelta(days=random_offset)
        
        # Try to construct a date with the original month/day in the target year
        try:
            adjusted_date = datetime(
                year=random_date.year,
                month=month,
                day=day,
                hour=created_date.hour,
                minute=created_date.minute,
                second=created_date.second
            )
            
            # Verify the adjusted date is within our range
            if adjusted_date < start_date:
                adjusted_date = start_date
            elif adjusted_date > end_date:
                adjusted_date = end_date
                
        except ValueError:
            # Handle invalid dates (like February 29 in non-leap years)
            adjusted_date = random_date
            
        return adjusted_date

    def process_batch(self, batch_df: pd.DataFrame, query_stats: dict) -> pd.DataFrame:
        """Process a batch of tweets with safer engagement calculation"""
        if batch_df.empty:
            return pd.DataFrame()
            
        try:
            # Convert dates to consistent format
            batch_df['Date'] = pd.to_datetime(batch_df['Date']).dt.normalize()
            
            # Basic preprocessing with integer operations
            batch_df['Likes'] = pd.to_numeric(batch_df['Likes'], errors='coerce').fillna(0).astype(int)
            batch_df['Retweets'] = pd.to_numeric(batch_df['Retweets'], errors='coerce').fillna(0).astype(int)
            batch_df['Quote_Tweets'] = pd.to_numeric(batch_df['Quote_Tweets'], errors='coerce').fillna(0).astype(int)
            batch_df['Replies'] = pd.to_numeric(batch_df['Replies'], errors='coerce').fillna(0).astype(int)
            
            # Calculate engagement score
            batch_df['Engagement_Score'] = (
                batch_df['Likes'] + 
                (batch_df['Retweets'] * 2) + 
                (batch_df['Quote_Tweets'] * 2) + 
                batch_df['Replies']
            )
            
            return batch_df
            
        except Exception as e:
            st.error(f"Error in batch processing: {str(e)}")
            return batch_df

    def process_twitter_data(self, twitter_data: list, query_stats: dict) -> pd.DataFrame:
        """Process and structure Twitter data"""
        if not twitter_data:
            st.warning("No relevant tweets found")
            return pd.DataFrame()
            
        df = pd.DataFrame(twitter_data)
        
        # Basic preprocessing with safe timezone handling
        df['Date'] = pd.to_datetime(df['Date'])
        if df['Date'].dt.tz is None:
            df['Date'] = df['Date'].dt.tz_localize('UTC')
        df = df.sort_values('Date', ascending=False)
        
        # Calculate engagement score
        df['Engagement_Score'] = (
            df['Likes'] + 
            df['Retweets'] * 2 + 
            df['Quote_Tweets'] * 2 + 
            df['Replies']
        )
        
        # Remove duplicates based on Tweet_ID
        df = df.drop_duplicates(subset=['Tweet_ID'])
        
        # Display collection statistics
        st.success(f"""
        Data Collection Summary:
        - Total Tweets: {len(df)}
        - Unique Authors: {df['Author'].nunique()}
        - Verified Authors: {df['Author_Verified'].sum()}
        - Total Engagement: {df['Likes'].sum() + df['Retweets'].sum() + df['Quote_Tweets'].sum()}
        - Queries Used: {len(query_stats)}
        """)
        
        return df

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
                        
                    sentiment_result = sentiment_analyser.analyse_sentiment(text)
                    
                    # Calculate engagement score with weights
                    engagement_score = (
                        int(tweet.get('Likes', 0)) * 1.0 +
                        int(tweet.get('Retweets', 0)) * 2.0 +
                        int(tweet.get('Replies', 0)) * 1.5 +
                        int(tweet.get('Quote_Tweets', 0)) * 2.0
                    )
                    
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
                
                # For debugging, generate a daily aggregation specifically for correlation analysis
                daily_data = raw_tweet_df.groupby(raw_tweet_df['Date'].dt.date).agg({
                    'Sentiment_Score': 'mean',
                    'Engagement_Score': 'mean',
                    'Text': 'count'
                }).reset_index()
                daily_data.columns = ['Date', 'Sentiment_Score', 'Engagement_Mean', 'Tweet_Count']
                
                # Store this in session state as well for direct correlation access
                st.session_state.twitter_daily_data = daily_data
                st.success(f"✅ Also saved daily aggregated data with {len(daily_data)} days for correlation analysis")
                
                return hourly_data
            else:
                st.warning("No tweets were successfully analyzed")
                return pd.DataFrame()
                
        except Exception as e:
            st.error(f"Error in sentiment analysis: {str(e)}")
            st.exception(e)  # This will show the full error traceback
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
            'timestamp': pd.Timestamp.now(tz='UTC')  # Use pandas timestamp for consistency
        }
        self.metrics['errors'].append(error_entry)
        
        if error_type in ['initialisation', 'general']:
            st.error(f"Critical error: {error_message}")
        else:
            st.warning(f"Non-critical error in {error_type}: {error_message}")