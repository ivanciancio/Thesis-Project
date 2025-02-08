import praw
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timezone

class RedditAnalyzer:
    def __init__(self):
        """Initialize Reddit analyzer using credentials from secrets.toml"""
        self.client_id = st.secrets["reddit"]["client_id"]
        self.client_secret = st.secrets["reddit"]["client_secret"]
        self.user_agent = st.secrets["reddit"].get("user_agent", "FinancialSentimentBot/1.0")
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
            start_date (datetime/Timestamp): Start date
            end_date (datetime/Timestamp): End date
        """
        if not self.reddit and not self.initialize_reddit():
            return pd.DataFrame()

        # Convert timestamps to UTC timestamps for Reddit API
        start_timestamp = pd.Timestamp(start_date).replace(tzinfo=timezone.utc).timestamp()
        end_timestamp = pd.Timestamp(end_date).replace(tzinfo=timezone.utc).timestamp()
        
        st.info(f"Fetching Reddit data from {start_date} to {end_date}")
        
        reddit_data = []
        total_posts_checked = 0
        
        try:
            for subreddit_name in self.subreddits:
                st.write(f"Searching in r/{subreddit_name}...")
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Create multiple search queries to increase chances of finding relevant posts
                search_queries = [
                    f'"{symbol}"',  # Exact match
                    f"${symbol}",   # Stock symbol format
                    f"{symbol} stock",
                    f"{symbol} price",
                    f"{symbol} analysis"
                ]
                
                for query in search_queries:
                    posts = subreddit.search(
                        query,
                        sort='new',
                        time_filter='year',  # Expanded time filter
                        limit=100  # Increased limit
                    )
                    
                    for post in posts:
                        total_posts_checked += 1
                        post_timestamp = post.created_utc
                        
                        # Check if post is within date range
                        if start_timestamp <= post_timestamp <= end_timestamp:
                            # Check if post is actually about the stock (to avoid false positives)
                            post_text = f"{post.title.lower()} {post.selftext.lower()}"
                            if (f"${symbol.lower()}" in post_text or 
                                f" {symbol.lower()} " in post_text or 
                                symbol.lower() in post_text.split()):
                                
                                post_data = {
                                    'Date': datetime.fromtimestamp(post_timestamp),
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
                                try:
                                    post.comments.replace_more(limit=0)
                                    for comment in post.comments.list()[:5]:  # Reduced to top 5 comments
                                        comment_timestamp = comment.created_utc
                                        if start_timestamp <= comment_timestamp <= end_timestamp:
                                            comment_data = {
                                                'Date': datetime.fromtimestamp(comment_timestamp),
                                                'Title': '[Comment]',
                                                'Text': comment.body,
                                                'Score': comment.score,
                                                'Comments': 0,
                                                'Subreddit': subreddit_name,
                                                'Type': 'comment',
                                                'URL': f"https://reddit.com{comment.permalink}"
                                            }
                                            reddit_data.append(comment_data)
                                except Exception as e:
                                    st.warning(f"Couldn't fetch comments for a post: {str(e)}")
                                    continue
            
            # Convert to DataFrame and sort by date
            df = pd.DataFrame(reddit_data)
            
            if not df.empty:
                st.success(f"Found {len(df)} relevant posts/comments out of {total_posts_checked} posts checked")
                df['Date'] = pd.to_datetime(df['Date'])
                return df.sort_values('Date', ascending=False)
            else:
                st.warning(f"No relevant posts found after checking {total_posts_checked} posts")
                # Add debugging information
                st.info(f"""
                Search details:
                - Date range: {start_date} to {end_date}
                - Symbol: {symbol}
                - Subreddits searched: {', '.join(self.subreddits)}
                - Total posts checked: {total_posts_checked}
                Try adjusting the date range or check if there might be alternative symbols used for this stock.
                """)
                return pd.DataFrame()
            
        except Exception as e:
            st.error(f"Error fetching Reddit data: {str(e)}")
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