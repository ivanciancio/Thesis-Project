import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def plot_market_data(market_data, market_sentiment=None):
    """Create market data visualisation"""
    fig = make_subplots(
        rows=2, 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('Price', 'Volume')
    )
    
    # Price candlestick
    fig.add_trace(
        go.Candlestick(
            x=market_data['Date'],
            open=market_data['Open'],
            high=market_data['High'],
            low=market_data['Low'],
            close=market_data['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Volume bars
    fig.add_trace(
        go.Bar(
            x=market_data['Date'],
            y=market_data['Volume'],
            name='Volume'
        ),
        row=2, col=1
    )
    
    # Add moving averages if available
    if 'MA_5' in market_data.columns:
        fig.add_trace(
            go.Scatter(
                x=market_data['Date'],
                y=market_data['MA_5'],
                name='MA 5',
                line=dict(color='orange')
            ),
            row=1, col=1
        )
    
    if 'MA_20' in market_data.columns:
        fig.add_trace(
            go.Scatter(
                x=market_data['Date'],
                y=market_data['MA_20'],
                name='MA 20',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        title_text="Market Data Analysis"
    )
    
    # Update y-axes labels
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

def plot_sentiment_comparison(news_data, reddit_data=None):
    """Create sentiment comparison visualisation with improved formatting"""
    fig = go.Figure()
    
    # Add news sentiment
    if news_data is not None and not news_data.empty:
        # Proper date handling and aggregation
        daily_news = (news_data
            .assign(Date=news_data['Date'].dt.date)
            .groupby('Date')
            .agg({
                'Sentiment Score': 'mean',
                'Title': 'count'
            })
            .reset_index())
        
        # Add news sentiment line
        fig.add_trace(
            go.Scatter(
                x=daily_news['Date'],
                y=daily_news['Sentiment Score'],
                mode='lines+markers',
                name='News Sentiment',
                line=dict(color='blue', width=2),
                marker=dict(size=6),
                hovertemplate=(
                    '<b>Date:</b> %{x}<br>' +
                    '<b>Sentiment:</b> %{y:.3f}<br>' +
                    '<b>Articles:</b> %{text}<br>'
                ),
                text=daily_news['Title']
            )
        )
        
        # Add article count as bars on secondary y-axis
        fig.add_trace(
            go.Bar(
                x=daily_news['Date'],
                y=daily_news['Title'],
                name='Number of Articles',
                yaxis='y2',
                marker_color='rgba(0,0,255,0.2)',
                hovertemplate=(
                    '<b>Date:</b> %{x}<br>' +
                    '<b>Articles:</b> %{y}<br>'
                )
            )
        )
    
    # Add Reddit sentiment if available
    if reddit_data is not None and not reddit_data.empty:
        daily_reddit = (reddit_data
            .assign(Date=reddit_data['Date'].dt.date)
            .groupby('Date')
            .agg({
                'Sentiment Score': 'mean',
                'Text': 'count'
            })
            .reset_index())
        
        fig.add_trace(
            go.Scatter(
                x=daily_reddit['Date'],
                y=daily_reddit['Sentiment Score'],
                mode='lines+markers',
                name='Reddit Sentiment',
                line=dict(color='orange', width=2),
                marker=dict(size=6),
                hovertemplate=(
                    '<b>Date:</b> %{x}<br>' +
                    '<b>Sentiment:</b> %{y:.3f}<br>' +
                    '<b>Posts:</b> %{text}<br>'
                ),
                text=daily_reddit['Text']
            )
        )
    
    # Update layout with better formatting
    fig.update_layout(
        title=dict(
            text="Sentiment Analysis Over Time",
            font=dict(size=20),
            y=0.95
        ),
        xaxis=dict(
            title="Date",
            title_font=dict(size=14),
            gridcolor='rgba(128,128,128,0.2)',
            showgrid=True,
            title_standoff=15
        ),
        yaxis=dict(
            title="Sentiment Score",
            title_font=dict(size=14),
            gridcolor='rgba(128,128,128,0.2)',
            showgrid=True,
            range=[-1, 1],  # Set fixed range for sentiment scores
            zeroline=True,
            zerolinecolor='rgba(128,128,128,0.4)',
            title_standoff=15
        ),
        yaxis2=dict(
            title="Number of Articles",
            title_font=dict(size=14),
            overlaying='y',
            side='right',
            showgrid=False,
            title_standoff=15
        ),
        height=500,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        hovermode='x unified',
        plot_bgcolor='white',
        margin=dict(t=80, r=80)
    )
    
    return fig

def plot_correlation_matrix(correlation_data):
    """Create correlation matrix visualisation"""
    fig = go.Figure(data=go.Heatmap(
        z=correlation_data.values,
        x=correlation_data.columns,
        y=correlation_data.index,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        text=np.round(correlation_data.values, 3),
        texttemplate='%{text}',
        textfont={"size": 12},
        hoverongaps=False,
    ))
    
    fig.update_layout(
        title="Correlation Matrix",
        height=500,
        xaxis_title="",
        yaxis_title="",
        yaxis_autorange='reversed'
    )
    
    return fig

def plot_enhanced_correlation_matrix(correlations, metrics):
    """Create enhanced correlation matrix visualisation"""
    fig = go.Figure(data=go.Heatmap(
        z=correlations.values,
        x=correlations.columns,
        y=correlations.index,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        text=np.round(correlations.values, 3),
        texttemplate='%{text}<br>p=%{customdata}',
        customdata=np.round(metrics['p_values'], 3),
        textfont={"size": 14},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Correlation Matrix with Statistical Significance",
        height=600,
        xaxis_title="",
        yaxis_title="",
        yaxis_autorange='reversed'
    )
    
    return fig