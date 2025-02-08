import streamlit as st
from analysers.trend_analyser import SentimentTrendAnalyser
import plotly.graph_objects as go
import pandas as pd

def display_trend_analysis(trend_analysis):
    """Display trend analysis results"""
    if not trend_analysis:
        st.error("No trend analysis results available")
        return
    
    st.subheader("Trend Analysis Results")
    
    # Create metrics display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Trend Direction",
            trend_analysis['trend_direction'],
            delta=f"{trend_analysis['trend_strength']:.2f}"
        )
    
    with col2:
        st.metric(
            "Trend Confidence",
            f"{trend_analysis['trend_confidence']:.1f}%"
        )
    
    with col3:
        st.metric(
            "Volatility",
            f"{trend_analysis['volatility']:.3f}"
        )
    
    # Display moving averages comparison
    st.subheader("Moving Averages")
    st.write(f"Short-term MA (5): {trend_analysis['moving_average_5']:.2f}")
    st.write(f"Long-term MA (20): {trend_analysis['moving_average_20']:.2f}")
    
    if trend_analysis.get('outlier_count', 0) > 0:
        st.warning(f"Found {trend_analysis['outlier_count']} significant outliers in the data")

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
    
    if not all(key in st.session_state for key in ['market_data', 'news_data']):
        st.warning("Please complete previous analyses first")
        return
    
    trend_analyser = SentimentTrendAnalyser()
    
    # Analyse market trends
    market_trend = trend_analyser.analyse_trends(st.session_state.market_data['Close'])
    
    # Analyse sentiment trends
    sentiment_trend = trend_analyser.analyse_trends(st.session_state.news_data['Sentiment Score'])
    
    # Display trend analysis
    st.subheader("Market Trends")
    display_trend_analysis(market_trend)
    
    st.subheader("Sentiment Trends")
    display_trend_analysis(sentiment_trend)
    
    # Create and display visualisation
    fig = create_trend_visualisation(
        st.session_state.market_data,
        st.session_state.news_data
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Get and display recommendations
    recommendations = trend_analyser.get_trend_recommendations(market_trend, sentiment_trend)
    display_recommendations(recommendations)
    
    # Add trend summary for download
    if st.button("Generate Trend Report"):
        report = generate_trend_report(market_trend, sentiment_trend, recommendations)
        st.download_button(
            "Download Trend Report",
            report,
            "trend_report.txt",
            "text/plain",
            key='download-trend-report'
        )

def generate_trend_report(market_trend, sentiment_trend, recommendations):
    """Generate a detailed trend report"""
    report = []
    report.append("FINANCIAL MARKET TREND ANALYSIS REPORT")
    report.append("=" * 40 + "\n")
    
    report.append("MARKET TRENDS")
    report.append("-" * 20)
    report.append(f"Direction: {market_trend['trend_direction']}")
    report.append(f"Strength: {market_trend['trend_strength']:.3f}")
    report.append(f"Confidence: {market_trend['trend_confidence']:.1f}%")
    report.append(f"Volatility: {market_trend['volatility']:.3f}\n")
    
    report.append("SENTIMENT TRENDS")
    report.append("-" * 20)
    report.append(f"Direction: {sentiment_trend['trend_direction']}")
    report.append(f"Strength: {sentiment_trend['trend_strength']:.3f}")
    report.append(f"Confidence: {sentiment_trend['trend_confidence']:.1f}%")
    report.append(f"Volatility: {sentiment_trend['volatility']:.3f}\n")
    
    report.append("RECOMMENDATIONS")
    report.append("-" * 20)
    for i, rec in enumerate(recommendations, 1):
        report.append(f"{i}. {rec}")
    
    return "\n".join(report)

if __name__ == "__main__":
    trend_interpretation_page()