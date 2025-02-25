import praw
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timezone
from typing import Dict, List, Optional
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

class RedditAnalyser:
    def __init__(self):
        """Initialise Reddit analyser with enhanced capabilities"""
        # Reddit API credentials
        self.client_id = st.secrets["reddit"]["client_id"]
        self.client_secret = st.secrets["reddit"]["client_secret"]
        self.user_agent = st.secrets["reddit"].get("user_agent", "FinancialSentimentBot/1.0")
        self.reddit = None
        
        # Platform configuration
        self.subreddits = {
            'primary': ['wallstreetbets', 'stocks', 'investing', 'stockmarket'],
            'secondary': ['options', 'SecurityAnalysis', 'StockMarket'],
            'crypto': ['cryptocurrency', 'CryptoMarkets']
        }
        
        # Sentiment analysis thresholds
        self.sentiment_thresholds = {
            'strong_positive': 0.5,
            'positive': 0.1,
            'negative': -0.1,
            'strong_negative': -0.5
        }
        
        # Initialise tracking metrics
        self.metrics = {
            'posts_analysed': 0,
            'comments_analysed': 0,
            'api_calls': 0,
            'errors': []
        }

    def initialise_reddit(self) -> bool:
        """Initialise Reddit API client with error handling"""
        try:
            self.reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent
            )
            return True
        except Exception as e:
            st.error(f"Error initialising Reddit API: {str(e)}")
            self.metrics['errors'].append(('initialisation', str(e)))
            return False

    def create_search_queries(self, symbol: str) -> List[str]:
        """Generate comprehensive search queries for the symbol"""
        return [
            f'"{symbol}"',          # Exact match
            f"${symbol}",           # Stock symbol format
            f"{symbol} stock",      
            f"{symbol} price",
            f"{symbol} analysis",
            f"{symbol} DD",         # Due Diligence
            f"{symbol} short",
            f"{symbol} long",
            f"{symbol} earnings"
        ]

    def fetch_reddit_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch and analyse Reddit posts and comments with enhanced error handling
        and comprehensive data collection
        """
        if not self.reddit and not self.initialise_reddit():
            return pd.DataFrame()

        # Convert timestamps for Reddit API
        start_timestamp = pd.Timestamp(start_date).replace(tzinfo=timezone.utc).timestamp()
        end_timestamp = pd.Timestamp(end_date).replace(tzinfo=timezone.utc).timestamp()
        
        st.info(f"Fetching Reddit data from {start_date} to {end_date}")
        
        reddit_data = []
        subreddit_stats = {}
        
        try:
            # Analyse primary and secondary subreddits
            for category, subreddit_list in self.subreddits.items():
                for subreddit_name in subreddit_list:
                    st.write(f"Searching in r/{subreddit_name}...")
                    subreddit = self.reddit.subreddit(subreddit_name)
                    subreddit_stats[subreddit_name] = {'posts': 0, 'comments': 0}
                    
                    search_queries = self.create_search_queries(symbol)
                    
                    for query in search_queries:
                        try:
                            posts = subreddit.search(
                                query,
                                sort='new',
                                time_filter='year',
                                limit=200
                            )
                            
                            for post in posts:
                                self.metrics['posts_analysed'] += 1
                                post_timestamp = post.created_utc
                                
                                if start_timestamp <= post_timestamp <= end_timestamp:
                                    # Validate post relevance
                                    if self.validate_post_relevance(post, symbol):
                                        post_data = self.extract_post_data(post, subreddit_name, category)
                                        reddit_data.append(post_data)
                                        subreddit_stats[subreddit_name]['posts'] += 1
                                        
                                        # Fetch and analyse comments
                                        comment_data = self.fetch_post_comments(
                                            post, symbol, subreddit_name, 
                                            category, start_timestamp, end_timestamp
                                        )
                                        reddit_data.extend(comment_data)
                                        subreddit_stats[subreddit_name]['comments'] += len(comment_data)
                                        
                        except Exception as e:
                            self.handle_error('search', str(e), subreddit_name)
                            continue
            
            # Convert to DataFrame and process
            df = self.process_reddit_data(reddit_data, subreddit_stats)
            return df
            
        except Exception as e:
            self.handle_error('general', str(e))
            return pd.DataFrame()

    def validate_post_relevance(self, post: praw.models.Submission, symbol: str) -> bool:
        """Validate if post is relevant to the symbol"""
        text = f"{post.title.lower()} {post.selftext.lower()}"
        
        # Check for symbol mentions
        symbol_lower = symbol.lower()
        if (f"${symbol_lower}" in text or 
            f" {symbol_lower} " in text or 
            symbol_lower in text.split()):
            
            # Additional validation criteria
            if post.score < -5:  # Filter out heavily downvoted posts
                return False
            if len(post.selftext.split()) < 5 and post.num_comments < 2:  # Filter low-quality posts
                return False
                
            return True
            
        return False

    def extract_post_data(self, post: praw.models.Submission, subreddit_name: str, category: str) -> Dict:
        """Extract and structure post data"""
        return {
            'Date': datetime.fromtimestamp(post.created_utc),
            'Title': post.title,
            'Text': post.selftext,
            'Score': post.score,
            'Comments': post.num_comments,
            'Awards': len(post.all_awardings) if hasattr(post, 'all_awardings') else 0,
            'Upvote_Ratio': post.upvote_ratio if hasattr(post, 'upvote_ratio') else None,
            'Subreddit': subreddit_name,
            'Category': category,
            'Type': 'post',
            'URL': f"https://reddit.com{post.permalink}",
            'Author': str(post.author),
            'Is_Original_Content': post.is_original_content if hasattr(post, 'is_original_content') else False
        }

    def fetch_post_comments(self, post: praw.models.Submission, symbol: str, 
                          subreddit_name: str, category: str,
                          start_timestamp: float, end_timestamp: float) -> List[Dict]:
        """Fetch and analyse post comments"""
        comment_data = []
        try:
            post.comments.replace_more(limit=0)
            # Sort comments by score and get top ones
            comments = sorted(post.comments.list()[:10], key=lambda x: x.score, reverse=True)
            
            for comment in comments:
                self.metrics['comments_analysed'] += 1
                comment_timestamp = comment.created_utc
                
                if (start_timestamp <= comment_timestamp <= end_timestamp and
                    self.validate_comment_relevance(comment, symbol)):
                    
                    comment_dict = {
                        'Date': datetime.fromtimestamp(comment_timestamp),
                        'Title': '[Comment]',
                        'Text': comment.body,
                        'Score': comment.score,
                        'Comments': len(comment.replies) if hasattr(comment, 'replies') else 0,
                        'Awards': len(comment.all_awardings) if hasattr(comment, 'all_awardings') else 0,
                        'Subreddit': subreddit_name,
                        'Category': category,
                        'Type': 'comment',
                        'URL': f"https://reddit.com{comment.permalink}",
                        'Author': str(comment.author),
                        'Parent_Type': 'post' if comment.parent_id.startswith('t3_') else 'comment'
                    }
                    comment_data.append(comment_dict)
                    
        except Exception as e:
            self.handle_error('comments', str(e))
            
        return comment_data

    def validate_comment_relevance(self, comment: praw.models.Comment, symbol: str) -> bool:
        """Validate if comment is relevant and meets quality criteria"""
        if len(comment.body.split()) < 5:  # Filter very short comments
            return False
        if comment.score < -2:  # Filter heavily downvoted comments
            return False
            
        # Check for symbol mentions
        symbol_lower = symbol.lower()
        text = comment.body.lower()
        return (f"${symbol_lower}" in text or 
                f" {symbol_lower} " in text or 
                symbol_lower in text.split())

    def process_reddit_data(self, reddit_data: List[Dict], subreddit_stats: Dict) -> pd.DataFrame:
        """Process and structure Reddit data"""
        if not reddit_data:
            st.warning("No relevant posts found")
            return pd.DataFrame()
            
        df = pd.DataFrame(reddit_data)
        
        # Display collection statistics
        st.success(f"""
        Data Collection Summary:
        - Total Posts: {len(df[df['Type'] == 'post'])}
        - Total Comments: {len(df[df['Type'] == 'comment'])}
        - Subreddits Analysed: {len(subreddit_stats)}
        - Total Items Analysed: {self.metrics['posts_analysed'] + self.metrics['comments_analysed']}
        """)
        
        # Basic preprocessing
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date', ascending=False)
        
        return df

    def analyse_content(self, reddit_df, sentiment_analyser):
        """Analyse Reddit content using provided sentiment analyser"""
        if reddit_df.empty:
            return pd.DataFrame()
                
        try:
            # Analyse sentiment for each post/comment
            analysed_data = []
                
            for _, row in reddit_df.iterrows():
                text = f"{row['Title']} {row['Text']}"
                sentiment_result = sentiment_analyser.analyse_sentiment(text)
                    
                analysed_item = {
                    'Date': row['Date'],
                    'Type': row['Type'],
                    'Subreddit': row['Subreddit'],
                    'Score': row['Score'],
                    'Comments': row['Comments'],
                    'Text': row['Text'][:200] + '...' if len(row['Text']) > 200 else row['Text'],
                    'URL': row['URL'],
                    'Sentiment_Score': sentiment_result['score'],
                    'Sentiment': sentiment_result['sentiment'],
                    'Confidence': sentiment_result.get('confidence', 0.5),
                    'Polarisation': sentiment_result.get('polarisation_metrics', {}).get('std', 0)
                }
                analysed_data.append(analysed_item)
                
            return pd.DataFrame(analysed_data)
                
        except Exception as e:
            st.error(f"Error analysing Reddit content: {e}")
            return pd.DataFrame()

    def display_analysis(self, reddit_df: pd.DataFrame):
        """Display comprehensive Reddit analysis results"""
        if reddit_df.empty:
            st.warning("No Reddit data available for analysis")
            return

        # Create tabs for different analysis views
        tabs = st.tabs(["Overview", "Sentiment Analysis", "Platform Comparison", "Detailed Data"])

        with tabs[0]:
            self.display_overview_metrics(reddit_df)

        with tabs[1]:
            self.display_sentiment_analysis(reddit_df)

        with tabs[2]:
            self.display_platform_comparison(reddit_df)

        with tabs[3]:
            self.display_detailed_data(reddit_df)

    def display_overview_metrics(self, df: pd.DataFrame):
        """Display overview metrics and statistics"""
        st.subheader("Overview Metrics")
        
        # Create three columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Posts", len(df[df['Type'] == 'post']))
            st.metric("Total Comments", len(df[df['Type'] == 'comment']))
            st.metric("Unique Authors", df['Author'].nunique())
            
        with col2:
            avg_sentiment = df['Sentiment_Score'].mean()
            st.metric("Average Sentiment", f"{avg_sentiment:.2f}")
            st.metric("Positive Content", len(df[df['Sentiment'] == 'Positive']))
            st.metric("Negative Content", len(df[df['Sentiment'] == 'Negative']))
            
        with col3:
            st.metric("Total Engagement", df['Score'].sum())
            st.metric("Average Comments", int(df[df['Type'] == 'post']['Comments'].mean()))
            st.metric("Highly Discussed Posts", len(df[df['Comments'] > df['Comments'].quantile(0.9)]))

        # Display time series of activity
        st.subheader("Activity Over Time")
        daily_activity = df.resample('D', on='Date').size()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_activity.index,
            y=daily_activity.values,
            mode='lines+markers',
            name='Daily Activity'
        ))
        
        fig.update_layout(
            title="Daily Reddit Activity",
            xaxis_title="Date",
            yaxis_title="Number of Posts/Comments",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def display_sentiment_analysis(self, df: pd.DataFrame):
        """Display detailed sentiment analysis"""
        st.subheader("Sentiment Analysis")
        
        # Sentiment distribution by subreddit
        fig = go.Figure()
        
        for subreddit in df['Subreddit'].unique():
            subreddit_data = df[df['Subreddit'] == subreddit]
            
            fig.add_trace(go.Box(
                y=subreddit_data['Sentiment_Score'],
                name=subreddit,
                boxpoints='outliers',
                jitter=0.3,
                pointpos=-1.8
            ))
        
        fig.update_layout(
            title="Sentiment Distribution Across Subreddits",
            yaxis_title="Sentiment Score",
            boxmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def display_platform_comparison(self, df: pd.DataFrame):
        """Display platform comparison analysis"""
        st.subheader("Platform Comparison")

        # Calculate platform-level metrics
        platform_metrics = df.groupby('Subreddit').agg({
            'Sentiment_Score': ['mean', 'std'],
            'Score': 'sum',
            'Comments': 'mean',
            'Polarisation': 'mean'
        }).round(3)

        st.write("Platform Comparison Metrics:")
        st.dataframe(platform_metrics)

        # Create platform correlation heatmap
        platform_correlations = df.pivot_table(
            values='Sentiment_Score',
            index='Date',
            columns='Subreddit',
            aggfunc='mean'
        ).corr()

        fig = go.Figure(data=go.Heatmap(
            z=platform_correlations,
            x=platform_correlations.columns,
            y=platform_correlations.index,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            colorbar=dict(
                title=dict(
                    text="Correlation",
                    side="right"
                )
            )
        ))

        fig.update_layout(
            title="Cross-Platform Sentiment Correlation",
            xaxis_title="Subreddit",
            yaxis_title="Subreddit"
        )

        st.plotly_chart(fig, use_container_width=True)

    def display_detailed_data(self, df: pd.DataFrame):
        """Display detailed data view"""
        st.subheader("Detailed Data Analysis")

        # Filter controls
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_subreddits = st.multiselect(
                "Select Subreddits",
                options=df['Subreddit'].unique(),
                default=df['Subreddit'].unique()
            )
        with col2:
            selected_sentiment = st.multiselect(
                "Select Sentiment",
                options=df['Sentiment'].unique(),
                default=df['Sentiment'].unique()
            )
        with col3:
            selected_type = st.multiselect(
                "Select Type",
                options=df['Type'].unique(),
                default=df['Type'].unique()
            )

        # Apply filters
        filtered_df = df[
            (df['Subreddit'].isin(selected_subreddits)) &
            (df['Sentiment'].isin(selected_sentiment)) &
            (df['Type'].isin(selected_type))
        ]

        # Display filtered data
        st.dataframe(
            filtered_df[[
                'Date', 'Subreddit', 'Type', 'Text', 
                'Sentiment', 'Sentiment_Score', 'Score', 
                'Comments', 'URL'
            ]].sort_values('Date', ascending=False),
            use_container_width=True
        )

    def handle_error(self, error_type: str, error_message: str, context: str = None):
        """Handle and log errors"""
        error_entry = {
            'type': error_type,
            'message': error_message,
            'context': context,
            'timestamp': datetime.now()
        }
        self.metrics['errors'].append(error_entry)
        
        if error_type in ['initialisation', 'general']:
            st.error(f"Critical error: {error_message}")
        else:
            st.warning(f"Non-critical error in {error_type}: {error_message}")