import streamlit as st
from analysers.trend_analyser import SentimentTrendAnalyser
import plotly.graph_objects as go
import pandas as pd

def display_trend_analysis(trend_analysis):
    """Display trend analysis results with error handling"""
    if not trend_analysis:
        st.warning("⚠️ No trend analysis results available")
        return
    
    try:
        # Create metrics display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Trend Direction",
                trend_analysis.get('trend_direction', 'N/A'),
                delta=f"{trend_analysis.get('trend_strength', 0):.2f}"
            )
        
        with col2:
            st.metric(
                "Trend Confidence",
                f"{trend_analysis.get('trend_confidence', 0):.1f}%"
            )
        
        with col3:
            st.metric(
                "Volatility",
                f"{trend_analysis.get('volatility', 0):.3f}"
            )
        
        # Display moving averages comparison
        st.subheader("Moving Averages")
        ma5 = trend_analysis.get('moving_average_5')
        ma20 = trend_analysis.get('moving_average_20')
        
        if ma5 is not None:
            st.write(f"Short-term MA (5): {ma5:.2f}")
        if ma20 is not None:
            st.write(f"Long-term MA (20): {ma20:.2f}")
        
        if trend_analysis.get('outlier_count', 0) > 0:
            st.warning(f"⚠️ Found {trend_analysis['outlier_count']} significant outliers in the data")
            
    except Exception as e:
        st.error(f"❌ Error displaying trend analysis: {str(e)}")

def display_recommendations(recommendations):
    """Display trend recommendations"""
    if not recommendations:
        return
    
    st.subheader("📋 Recommendations")
    for i, rec in enumerate(recommendations, 1):
        st.write(f"{i}. {rec}")

def create_trend_visualisation(market_data, news_sentiment, reddit_sentiment=None, twitter_sentiment=None):
    """Create improved price vs sentiment visualisation with Twitter data"""
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(
        go.Scatter(
            x=market_data['Date'],
            y=market_data['Close'],
            name='Price',
            line=dict(color='blue', width=2),
            yaxis='y'
        )
    )
    
    # Add sentiment lines for each source
    if not news_sentiment.empty:
        daily_news = news_sentiment.groupby(news_sentiment['Date'].dt.date)['Sentiment Score'].mean()
        fig.add_trace(
            go.Scatter(
                x=daily_news.index,
                y=daily_news.values,
                name='News Sentiment',
                line=dict(color='red', width=2),
                yaxis='y2'
            )
        )
    
    if reddit_sentiment is not None and not reddit_sentiment.empty:
        daily_reddit = reddit_sentiment.groupby(reddit_sentiment['Date'].dt.date)['Sentiment_Score'].mean()
        fig.add_trace(
            go.Scatter(
                x=daily_reddit.index,
                y=daily_reddit.values,
                name='Reddit Sentiment',
                line=dict(color='orange', width=2),
                yaxis='y2'
            )
        )
    
    if twitter_sentiment is not None and not twitter_sentiment.empty:
        daily_twitter = twitter_sentiment.groupby(twitter_sentiment['Date'].dt.date)['Sentiment_Score'].mean()
        fig.add_trace(
            go.Scatter(
                x=daily_twitter.index,
                y=daily_twitter.values,
                name='Twitter Sentiment',
                line=dict(color='purple', width=2),
                yaxis='y2'
            )
        )
    
    # Update layout
    fig.update_layout(
        title='Price vs Sentiment Trends',
        xaxis=dict(title='Date'),
        yaxis=dict(
            title='Price ($)',
            titlefont=dict(color='blue'),
            tickfont=dict(color='blue')
        ),
        yaxis2=dict(
            title='Sentiment Score',
            titlefont=dict(color='red'),
            tickfont=dict(color='red'),
            overlaying='y',
            side='right',
            range=[-1, 1]
        ),
        height=600,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def trend_interpretation_page():
    st.title("📊 Trend Interpretation")
    
    # Check if required data exists
    if not all(key in st.session_state for key in ['market_data', 'news_data']):
        st.warning("⚠️ Please complete market and sentiment analysis first")
        return
    
    try:
        trend_analyser = SentimentTrendAnalyser()
        
        # Analyze market trends
        market_trend = trend_analyser.analyse_trends(st.session_state.market_data['Close'])
        
        # Analyze sentiment trends for each source
        sentiment_trends = {}
        if st.session_state.news_data is not None:
            sentiment_trends['news'] = trend_analyser.analyse_trends(
                st.session_state.news_data['Sentiment Score']
            )
        
        if 'reddit_data' in st.session_state and st.session_state.reddit_data is not None:
            sentiment_trends['reddit'] = trend_analyser.analyse_trends(
                st.session_state.reddit_data['Sentiment_Score']
            )
            
        if 'twitter_data' in st.session_state and st.session_state.twitter_data is not None:
            sentiment_trends['twitter'] = trend_analyser.analyse_trends(
                st.session_state.twitter_data['Sentiment_Score']
            )
        
        # Display trend analysis for each source
        st.subheader("Market Trends")
        display_trend_analysis(market_trend)
        
        # Display sentiment trends in columns
        cols = st.columns(len(sentiment_trends))
        for i, (source, trend) in enumerate(sentiment_trends.items()):
            with cols[i]:
                st.subheader(f"{source.title()} Sentiment Trends")
                display_trend_analysis(trend)
        
        # Create and display visualization
        fig = create_trend_visualisation(
            st.session_state.market_data,
            st.session_state.news_data,
            st.session_state.get('reddit_data'),
            st.session_state.get('twitter_data')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Get and display recommendations
        recommendations = trend_analyser.get_trend_recommendations(
            market_trend,
            sentiment_trends
        )
        if recommendations:
            display_recommendations(recommendations)
            
    except Exception as e:
        st.error(f"Error in trend analysis: {str(e)}")

def generate_trend_report(market_trend, sentiment_trend, recommendations):
    """Generate a detailed trend report with error handling"""
    try:
        report = []
        report.append("FINANCIAL MARKET TREND ANALYSIS REPORT")
        report.append("=" * 40 + "\n")
        
        if market_trend:
            report.append("MARKET TRENDS")
            report.append("-" * 20)
            report.append(f"Direction: {market_trend.get('trend_direction', 'N/A')}")
            report.append(f"Strength: {market_trend.get('trend_strength', 0):.3f}")
            report.append(f"Confidence: {market_trend.get('trend_confidence', 0):.1f}%")
            report.append(f"Volatility: {market_trend.get('volatility', 0):.3f}\n")
        
        if sentiment_trend:
            report.append("SENTIMENT TRENDS")
            report.append("-" * 20)
            report.append(f"Direction: {sentiment_trend.get('trend_direction', 'N/A')}")
            report.append(f"Strength: {sentiment_trend.get('trend_strength', 0):.3f}")
            report.append(f"Confidence: {sentiment_trend.get('trend_confidence', 0):.1f}%")
            report.append(f"Volatility: {sentiment_trend.get('volatility', 0):.3f}\n")
        
        if recommendations:
            report.append("RECOMMENDATIONS")
            report.append("-" * 20)
            for i, rec in enumerate(recommendations, 1):
                report.append(f"{i}. {rec}")
        
        return "\n".join(report)
        
    except Exception as e:
        return f"Error generating report: {str(e)}"

if __name__ == "__main__":
    trend_interpretation_page()