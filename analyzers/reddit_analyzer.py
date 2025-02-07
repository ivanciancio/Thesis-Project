import praw
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

class RedditAnalyzer:
    def __init__(self, client_id, client_secret, user_agent="FinancialSentimentBot/1.0"):
        """Initialize Reddit analyzer with API credentials"""
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self.reddit = None
        self.subreddits = ['wallstreetbets', 'stocks', 'investing', 'stockmarket']

    def initialize_reddit(self):
        """Initialize Reddit API client"""
        try:
            self.reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent
            )
            return True
        except Exception as e:
            st.error(f"Error initializing Reddit API: {e}")
            return False

    def fetch_reddit_data(self, symbol, start_date, end_date):
        """
        Fetch Reddit posts and comments from relevant financial subreddits
        
        Args:
            symbol (str): Stock symbol
            start_date (pd.Timestamp): Start date
            end_date (pd.Timestamp): End date
        """
        if not self.reddit and not self.initialize_reddit():
            return pd.DataFrame()

        # Convert timestamps to datetime.datetime for comparison
        start_datetime = pd.Timestamp(start_date).to_pydatetime()
        end_datetime = pd.Timestamp(end_date).to_pydatetime()
        
        reddit_data = []
        
        try:
            for subreddit_name in self.subreddits:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Search for posts containing the symbol
                search_query = f"({symbol} OR ${symbol})"
                posts = subreddit.search(search_query, sort='new', time_filter='month')
                
                for post in posts:
                    post_datetime = datetime.fromtimestamp(post.created_utc)
                    
                    # Check if post is within date range
                    if start_datetime <= post_datetime <= end_datetime:
                        post_data = {
                            'Date': post_datetime,
                            'Title': post.title,
                            'Text': post.selftext,
                            'Score': post.score,
                            'Comments': post.num_comments,
                            'Subreddit': subreddit_name,
                            'Type': 'post',
                            'URL': f"https://reddit.com{post.permalink}"
                        }
                        reddit_data.append(post_data)
                        
                        # Get top comments
                        post.comments.replace_more(limit=0)
                        for comment in post.comments.list()[:10]:  # Get top 10 comments
                            comment_datetime = datetime.fromtimestamp(comment.created_utc)
                            if start_datetime <= comment_datetime <= end_datetime:
                                comment_data = {
                                    'Date': comment_datetime,
                                    'Title': '[Comment]',
                                    'Text': comment.body,
                                    'Score': comment.score,
                                    'Comments': 0,
                                    'Subreddit': subreddit_name,
                                    'Type': 'comment',
                                    'URL': f"https://reddit.com{comment.permalink}"
                                }
                                reddit_data.append(comment_data)
            
            # Convert to DataFrame and sort by date
            df = pd.DataFrame(reddit_data)
            if not df.empty:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date', ascending=False)
            
            return df
            
        except Exception as e:
            st.error(f"Error fetching Reddit data: {e}")
            return pd.DataFrame()

    def analyze_content(self, reddit_df, sentiment_analyzer):
        """Analyze Reddit content using provided sentiment analyzer"""
        if reddit_df.empty:
            return pd.DataFrame()
            
        try:
            # Analyze sentiment for each post/comment
            analyzed_data = []
            
            for _, row in reddit_df.iterrows():
                text = f"{row['Title']} {row['Text']}"
                sentiment_result = sentiment_analyzer.analyze_sentiment(text)
                
                analyzed_item = {
                    'Date': row['Date'],
                    'Type': row['Type'],
                    'Subreddit': row['Subreddit'],
                    'Score': row['Score'],
                    'Comments': row['Comments'],
                    'Text': row['Text'][:200] + '...' if len(row['Text']) > 200 else row['Text'],
                    'URL': row['URL'],
                    'Sentiment Score': sentiment_result['score'],
                    'Sentiment': sentiment_result['sentiment']
                }
                analyzed_data.append(analyzed_item)
            
            return pd.DataFrame(analyzed_data)
            
        except Exception as e:
            st.error(f"Error analyzing Reddit content: {e}")
            return pd.DataFrame()

    def display_analysis(self, reddit_df):
        """Display Reddit analysis results"""
        if reddit_df.empty:
            st.warning("No Reddit data available")
            return

        # Display summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Posts", len(reddit_df[reddit_df['Type'] == 'post']))
            st.metric("Total Comments", len(reddit_df[reddit_df['Type'] == 'comment']))
            
        with col2:
            avg_sentiment = reddit_df['Sentiment Score'].mean()
            st.metric("Average Sentiment", f"{avg_sentiment:.2f}")
            st.metric("Positive Posts/Comments", len(reddit_df[reddit_df['Sentiment'] == 'Positive']))
            
        with col3:
            st.metric("Total Score", reddit_df['Score'].sum())
            st.metric("Negative Posts/Comments", len(reddit_df[reddit_df['Sentiment'] == 'Negative']))

        # Display sentiment distribution
        st.subheader("Sentiment Distribution by Subreddit")
        fig = go.Figure()
        
        for subreddit in reddit_df['Subreddit'].unique():
            subreddit_data = reddit_df[reddit_df['Subreddit'] == subreddit]
            
            fig.add_trace(go.Box(
                y=subreddit_data['Sentiment Score'],
                name=subreddit,
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ))
        
        fig.update_layout(
            title="Sentiment Distribution Across Subreddits",
            yaxis_title="Sentiment Score",
            boxmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display recent posts/comments
        st.subheader("Recent Reddit Activity")
        display_df = reddit_df[['Date', 'Subreddit', 'Type', 'Text', 'Sentiment', 'Score', 'URL']]
        st.dataframe(
            display_df.sort_values('Date', ascending=False),
            use_container_width=True
        )