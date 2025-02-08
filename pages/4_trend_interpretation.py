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

def create_trend_visualisation(market_data, sentiment_data):
    """Create improved price vs sentiment visualisation"""
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
    
    # Add sentiment line
    if not sentiment_data.empty:
        # Calculate daily sentiment
        daily_sentiment = (sentiment_data
            .groupby(sentiment_data['Date'].dt.date)
            .agg({
                'Sentiment Score': 'mean'
            })
            .reset_index())
        daily_sentiment['Date'] = pd.to_datetime(daily_sentiment['Date'])
        
        fig.add_trace(
            go.Scatter(
                x=daily_sentiment['Date'],
                y=daily_sentiment['Sentiment Score'],
                name='Sentiment',
                line=dict(color='red', width=2),
                yaxis='y2'
            )
        )
    
    # Update layout
    fig.update_layout(
        title='Price vs Sentiment Trends',
        xaxis=dict(
            title=dict(  # Changed here
                text='Date',  # Changed here
            ),
            gridcolor='rgba(128,128,128,0.2)',
            gridwidth=1,
            showgrid=True
        ),
        yaxis=dict(
            title=dict(  # Changed here
                text='Price ($)',  # Changed here
                font=dict(color='blue')  # Changed here - titlefont becomes font under title
            ),
            tickfont=dict(color='blue'),
            gridcolor='rgba(128,128,128,0.2)',
            gridwidth=1,
            showgrid=True
        ),
        yaxis2=dict(
            title=dict(  # Changed here
                text='Sentiment Score',  # Changed here
                font=dict(color='red')  # Changed here - titlefont becomes font under title
            ),
            tickfont=dict(color='red'),
            overlaying='y',
            side='right',
            range=[-1, 1],  # Fix sentiment range
            showgrid=False
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
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        hovermode='x unified'
    )
    
    # Add range slider
    fig.update_xaxes(rangeslider_visible=True)
    
    return fig

def trend_interpretation_page():
    st.title("📊 Trend Interpretation")
    
    # Check if required data exists in session state
    if 'market_data' not in st.session_state or st.session_state.market_data is None:
        st.warning("⚠️ Please fetch market data first in the Market Data page")
        st.info("➡️ Go to the Main page and fetch market data for your analysis")
        return
        
    if 'news_data' not in st.session_state or st.session_state.news_data is None:
        st.warning("⚠️ Please complete sentiment analysis first")
        st.info("➡️ Go to the Sentiment Analysis page and fetch news data")
        return
    
    try:
        trend_analyser = SentimentTrendAnalyser()
        
        # Safely get market data and check for required columns
        market_data = st.session_state.market_data
        if market_data is None or market_data.empty or 'Close' not in market_data.columns:
            st.error("❌ Invalid or missing market data. Please ensure market data is properly loaded.")
            return
            
        # Safely get sentiment data
        news_data = st.session_state.news_data
        if news_data is None or news_data.empty or 'Sentiment Score' not in news_data.columns:
            st.error("❌ Invalid or missing sentiment data. Please ensure sentiment analysis is completed.")
            return
        
        # Analyse market trends
        market_trend = trend_analyser.analyse_trends(market_data['Close'])
        if market_trend is None:
            st.error("❌ Unable to analyze market trends with the available data")
            return
            
        # Analyse sentiment trends
        sentiment_trend = trend_analyser.analyse_trends(news_data['Sentiment Score'])
        if sentiment_trend is None:
            st.error("❌ Unable to analyze sentiment trends with the available data")
            return
        
        # Display trend analysis
        st.subheader("Market Trends")
        display_trend_analysis(market_trend)
        
        st.subheader("Sentiment Trends")
        display_trend_analysis(sentiment_trend)
        
        # Create and display visualisation
        try:
            fig = create_trend_visualisation(market_data, news_data)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as viz_error:
            st.warning(f"⚠️ Unable to create visualization: {str(viz_error)}")
        
        # Get and display recommendations
        try:
            recommendations = trend_analyser.get_trend_recommendations(market_trend, sentiment_trend)
            if recommendations:
                display_recommendations(recommendations)
        except Exception as rec_error:
            st.warning(f"⚠️ Unable to generate recommendations: {str(rec_error)}")
        
        # Add trend summary for download
        if st.button("Generate Trend Report"):
            try:
                report = generate_trend_report(market_trend, sentiment_trend, recommendations)
                st.download_button(
                    "Download Trend Report",
                    report,
                    "trend_report.txt",
                    "text/plain",
                    key='download-trend-report'
                )
            except Exception as report_error:
                st.error(f"❌ Error generating report: {str(report_error)}")
                
    except Exception as e:
        st.error(f"❌ Error in trend analysis: {str(e)}")
        st.info("💡 Please ensure you have completed the following steps:")
        st.write("1. Fetched market data in the Main page")
        st.write("2. Completed sentiment analysis in the Sentiment Analysis page")
        if st.button("Show detailed error information"):
            st.exception(e)

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