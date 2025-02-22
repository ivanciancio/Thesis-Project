import tweepy
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import logging
import streamlit as st
import plotly.graph_objects as go
import time

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
        """Initialise X API v2 client with error handling"""
        try:
            # Initialise v2 client
            self.client = tweepy.Client(
                bearer_token=self.bearer_token,
                consumer_key=self.api_key,
                consumer_secret=self.api_secret,
                access_token=self.access_token,
                access_token_secret=self.access_token_secret,
                wait_on_rate_limit=True
            )
            return True
        except Exception as e:
            st.error(f"Error initialising X API: {str(e)}")
            self.metrics['errors'].append(('initialisation', str(e)))
            return False

    def create_search_queries(self, symbol: str) -> list:
        """Generate comprehensive search queries for the symbol"""
        # Base queries using logical operators for better filtering
        queries = [
            f'"{symbol} stock" OR "{symbol} shares"',  # Exact phrase matching
            f'"{symbol} trading" OR "{symbol} price"',
            f'"{symbol} market" OR "{symbol} investor"',
            f'"{symbol} forecast" OR "{symbol} analysis"'
        ]
        # Add common query parameters
        params = "-is:retweet lang:en"
        return [f"({q}) {params}" for q in queries]

    def fetch_twitter_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch and analyse X data using API v2 in batches"""
        if not self.client and not self.initialise_client():
            return pd.DataFrame()

        try:
            # Convert input dates to pandas Timestamps
            start_date = pd.Timestamp(start_date)
            end_date = pd.Timestamp(end_date)
            
            # Ensure timezone awareness using UTC
            if start_date.tz is None:
                start_date = start_date.tz_localize('UTC')
            if end_date.tz is None:
                end_date = end_date.tz_localize('UTC')
            
            st.info(f"Fetching X data from {start_date} to {end_date}")
            
            # Initialize storage for all batches
            all_batches = []
            total_tweets = 0
            
            # Get the current API time restriction
            current_time = pd.Timestamp.now(tz='UTC')
            oldest_allowed_time = current_time - pd.Timedelta(days=7)
            
            try:
                test_response = self.client.search_recent_tweets(query="test", max_results=10)
            except Exception as e:
                error_msg = str(e)
                time_match = re.search(r"must be on or after (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}Z)", error_msg)
                if time_match:
                    oldest_allowed_time = pd.Timestamp(time_match.group(1), tz='UTC')
            
            # Adjust start and end dates based on API limitations
            effective_start = max(start_date, oldest_allowed_time)
            effective_end = min(end_date, current_time)
            
            if effective_end <= effective_start:
                st.warning("No data available within the API time restrictions")
                return pd.DataFrame()
            
            # Calculate batch periods (7-day windows)
            batch_periods = []
            current_end = effective_end
            
            while current_end > effective_start:
                batch_start = max(current_end - pd.Timedelta(days=7), effective_start)
                batch_periods.append((batch_start, current_end))
                current_end = batch_start - pd.Timedelta(seconds=1)
            
            # Display batch information
            st.write(f"Total number of batches to process: {len(batch_periods)}")
            st.write(f"Note: Due to X API limitations, we can only fetch tweets from {oldest_allowed_time}")
            st.write(f"Will fetch tweets from {effective_start} to {effective_end}")
            
            # Process each batch
            search_queries = self.create_search_queries(symbol)
            
            for batch_idx, (batch_start, batch_end) in enumerate(batch_periods, 1):
                st.write(f"\nProcessing batch {batch_idx}/{len(batch_periods)}")
                st.write(f"Fetching tweets from: {batch_start} to {batch_end}")
                
                batch_data = []
                query_stats = {}
                
                for query in search_queries:
                    st.write(f"Searching for: {query}")
                    query_stats[query] = query_stats.get(query, 0)
                    
                    try:
                        tweets = tweepy.Paginator(
                            self.client.search_recent_tweets,
                            query=query,
                            start_time=batch_start,
                            end_time=batch_end,
                            tweet_fields=[
                                'created_at',
                                'public_metrics',
                                'lang',
                                'conversation_id',
                                'context_annotations'
                            ],
                            user_fields=[
                                'username',
                                'public_metrics',
                                'verified'
                            ],
                            expansions=['author_id'],
                            max_results=100,
                            limit=10
                        )
                        
                        for response in tweets:
                            if response.data:
                                users = {user.id: user for user in response.includes['users']} if 'users' in response.includes else {}
                                
                                for tweet in response.data:
                                    self.metrics['tweets_analysed'] += 1
                                    
                                    user = users.get(tweet.author_id, None)
                                    
                                    tweet_data = {
                                        'Date': tweet.created_at,
                                        'Text': tweet.text,
                                        'Author': user.username if user else None,
                                        'Author_Verified': user.verified if user else False,
                                        'Likes': tweet.public_metrics['like_count'],
                                        'Retweets': tweet.public_metrics['retweet_count'],
                                        'Replies': tweet.public_metrics['reply_count'],
                                        'Quote_Tweets': tweet.public_metrics['quote_count'],
                                        'Author_Followers': user.public_metrics['followers_count'] if user else 0,
                                        'Query': query,
                                        'Tweet_ID': tweet.id,
                                        'URL': f"https://twitter.com/{user.username if user else 'twitter'}/status/{tweet.id}",
                                        'Batch': batch_idx
                                    }
                                    
                                    if hasattr(tweet, 'context_annotations'):
                                        context_domains = []
                                        for annotation in tweet.context_annotations:
                                            if 'domain' in annotation:
                                                context_domains.append(annotation['domain']['name'])
                                        tweet_data['Context'] = ', '.join(context_domains)
                                    
                                    batch_data.append(tweet_data)
                                    query_stats[query] += 1
                                    
                    except Exception as e:
                        st.warning(f"Error in search query '{query}': {str(e)}")
                        continue
                    
                    time.sleep(2)  # Respect rate limits between queries
                
                # Process batch data
                if batch_data:
                    batch_df = pd.DataFrame(batch_data)
                    batch_df = self.process_batch(batch_df, query_stats)
                    all_batches.append(batch_df)
                    total_tweets += len(batch_df)
                    
                    st.success(f"""
                    Batch {batch_idx} Complete:
                    - Tweets in this batch: {len(batch_df)}
                    - Total tweets so far: {total_tweets}
                    """)
                
                time.sleep(5)  # Add delay between batches
            
            # Combine all batches
            if all_batches:
                final_df = pd.concat(all_batches, ignore_index=True)
                final_df = final_df.drop_duplicates(subset=['Tweet_ID'])
                
                st.success(f"""
                Data Collection Complete:
                - Total Batches: {len(batch_periods)}
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

    def process_batch(self, batch_df: pd.DataFrame, query_stats: dict) -> pd.DataFrame:
        """Process a single batch of Twitter data"""
        if batch_df.empty:
            return pd.DataFrame()
        
        # Basic preprocessing with safe timezone handling
        batch_df['Date'] = pd.to_datetime(batch_df['Date'])
        if batch_df['Date'].dt.tz is None:
            batch_df['Date'] = batch_df['Date'].dt.tz_localize('UTC')
        batch_df = batch_df.sort_values('Date', ascending=False)
        
        # Calculate engagement score
        batch_df['Engagement_Score'] = (
            batch_df['Likes'] + 
            batch_df['Retweets'] * 2 + 
            batch_df['Quote_Tweets'] * 2 + 
            batch_df['Replies']
        )
        
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
        """Analyse X content using provided sentiment analyser"""
        if twitter_df.empty:
            return pd.DataFrame()
                
        try:
            # Analyse sentiment for each tweet
            analysed_data = []
            total_tweets = len(twitter_df)
            
            # Create progress text
            progress_text = st.empty()
            progress_text.text('Starting sentiment analysis...')
            
            for idx, row in twitter_df.iterrows():
                try:
                    # Update progress
                    if idx % 5 == 0:  # Update every 5 tweets to avoid too many updates
                        progress_text.text(f'Analyzing tweets... {idx}/{total_tweets}')
                    
                    sentiment_result = sentiment_analyser.analyse_sentiment(row['Text'])
                    
                    analysed_item = {
                        'Date': row['Date'],
                        'Text': row['Text'],
                        'Author': row['Author'],
                        'Author_Verified': row['Author_Verified'],
                        'Author_Followers': row['Author_Followers'],
                        'Likes': row['Likes'],
                        'Retweets': row['Retweets'],
                        'Replies': row['Replies'],
                        'Quote_Tweets': row['Quote_Tweets'],
                        'Engagement_Score': row['Engagement_Score'],
                        'URL': row['URL'],
                        'Sentiment_Score': sentiment_result['score'],
                        'Sentiment': sentiment_result['sentiment'],
                        'Confidence': sentiment_result.get('confidence', 0.5)
                    }
                    
                    if 'Context' in row:
                        analysed_item['Context'] = row['Context']
                    
                    analysed_data.append(analysed_item)
                    
                except Exception as e:
                    st.warning(f"Error analyzing tweet {idx}: {str(e)}")
                    continue
            
            progress_text.text('Sentiment analysis completed!')
            
            # Create the final dataframe
            result_df = pd.DataFrame(analysed_data)
            
            if not result_df.empty:
                st.success(f"""
                Sentiment Analysis Complete:
                - Total Tweets Analyzed: {len(result_df)}
                - Average Sentiment Score: {result_df['Sentiment_Score'].mean():.2f}
                - Positive Tweets: {len(result_df[result_df['Sentiment_Score'] > 0])}
                - Negative Tweets: {len(result_df[result_df['Sentiment_Score'] < 0])}
                """)
            
            return result_df
                
        except Exception as e:
            st.error(f"Error in sentiment analysis: {str(e)}")
            return pd.DataFrame()

    def plot_sentiment_trend(self, df: pd.DataFrame) -> go.Figure:
        """Create sentiment trend visualisation"""
        try:
            # Calculate hourly sentiment and engagement
            df['date'] = df['Date'].dt.date
            df['hour'] = df['Date'].dt.hour
            
            hourly_data = df.groupby(['date', 'hour']).agg({
                'Sentiment_Score': 'mean',
                'Text': 'count',
                'Engagement_Score': 'mean'
            }).reset_index()
            
            # Convert to datetime safely with timezone handling
            hourly_data['DateTime'] = pd.to_datetime(
                hourly_data['date'].astype(str) + ' ' + 
                hourly_data['hour'].astype(str) + ':00:00'
            ).dt.tz_localize('UTC')
            
            # Create figure with secondary y-axis
            fig = go.Figure()
            
            # Add sentiment score line
            fig.add_trace(
                go.Scatter(
                    x=hourly_data['DateTime'],
                    y=hourly_data['Sentiment_Score'],
                    mode='lines+markers',
                    name='Tweet Sentiment',
                    line=dict(color='#1DA1F2', width=2),  # Twitter blue
                    marker=dict(size=6)
                )
            )
            
            # Add tweet count bars
            fig.add_trace(
                go.Bar(
                    x=hourly_data['DateTime'],
                    y=hourly_data['Text'],
                    name='Number of Tweets',
                    marker_color='rgba(29,161,242,0.2)',  # Light Twitter blue
                    yaxis='y2'
                )
            )
            
            # Add engagement score line
            fig.add_trace(
                go.Scatter(
                    x=hourly_data['DateTime'],
                    y=hourly_data['Engagement_Score'],
                    mode='lines',
                    name='Engagement Score',
                    line=dict(color='#17BF63', width=2),  # Twitter green
                    yaxis='y3'
                )
            )
            
            # Update layout
            fig.update_layout(
                title="X (Twitter) Sentiment & Engagement Analysis",
                xaxis=dict(
                    title="Date",
                    type='date',
                    tickformat='%Y-%m-%d %H:%M',
                    tickangle=45
                ),
                yaxis=dict(
                    title="Sentiment Score",
                    range=[-1, 1],
                    gridcolor='rgba(128,128,128,0.2)',
                    zeroline=True,
                    zerolinecolor='rgba(128,128,128,0.4)',
                    tickfont=dict(color='#1DA1F2')
                ),
                yaxis2=dict(
                    title="Number of Tweets",
                    overlaying='y',
                    side='right',
                    showgrid=False,
                    tickfont=dict(color='#657786')  # Twitter grey
                ),
                yaxis3=dict(
                    title="Engagement Score",
                    overlaying='y',
                    side='right',
                    position=0.85,
                    showgrid=False,
                    tickfont=dict(color='#17BF63')
                ),
                height=600,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor='rgba(255,255,255,0.8)'
                ),
                hovermode='x unified',
                plot_bgcolor='white'
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error in X analysis: {str(e)}")
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