import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st



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

def plot_model_comparison(data_df, model_columns):
    """Create visualization comparing different sentiment models"""
    try:
        # Ensure we're working with a copy to avoid modifying original
        df = data_df.copy()
        
        # Ensure Date column is properly formatted
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df['Date_Only'] = df['Date'].dt.date
        else:
            st.warning("Date column not found in data")
            return None
        
        # Create model data dictionary
        model_data = {}
        model_data['Date'] = df['Date_Only']  # Use date only for grouping
        
        # Add scores for each model
        for model in model_columns:
            col_name = f'{model}_score'
            if col_name in df.columns:
                model_data[model.capitalize()] = pd.to_numeric(df[col_name], errors='coerce')
            else:
                st.warning(f"Column {col_name} not found in data")
                return None
                
        # Add ensemble sentiment score
        ensemble_col = 'Sentiment_Score'
        if ensemble_col in df.columns:
            model_data['Ensemble'] = pd.to_numeric(df[ensemble_col], errors='coerce')
        else:
            # Try alternate column name with space
            ensemble_col = 'Sentiment Score'
            if ensemble_col in df.columns:
                model_data['Ensemble'] = pd.to_numeric(df[ensemble_col], errors='coerce')
            else:
                st.warning(f"Ensemble score column not found in data")
                return None
        
        # Convert to DataFrame
        model_df = pd.DataFrame(model_data)
        
        # Group by date and calculate mean
        daily_model_df = model_df.groupby('Date').mean().reset_index()
        
        # Create line chart using Plotly
        fig = go.Figure()
        
        # Add lines for each model
        colors = {
            'Textblob': '#3498db',  # Blue
            'Vader': '#2ecc71',     # Green
            'Finbert': '#e74c3c',   # Red
            'Ensemble': '#9b59b6'   # Purple
        }
        
        for model in model_columns:
            model_name = model.capitalize()
            if model_name in daily_model_df.columns:
                fig.add_trace(go.Scatter(
                    x=daily_model_df['Date'],
                    y=daily_model_df[model_name],
                    mode='lines',
                    name=model_name,
                    line=dict(
                        color=colors.get(model_name, '#666'), 
                        width=2
                    )
                ))
        
        # Add ensemble line
        if 'Ensemble' in daily_model_df.columns:
            fig.add_trace(go.Scatter(
                x=daily_model_df['Date'],
                y=daily_model_df['Ensemble'],
                mode='lines',
                name='Ensemble',
                line=dict(
                    color=colors['Ensemble'],
                    width=3, 
                    dash='dash'
                )
            ))
        
        # Update layout
        fig.update_layout(
            title="Model Sentiment Comparison",
            xaxis_title="Date",
            yaxis_title="Sentiment Score",
            yaxis=dict(
                range=[-1, 1],
                zeroline=True,
                zerolinecolor='rgba(128,128,128,0.4)'
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode='x unified',
            plot_bgcolor='white'
        )
        
        return fig
        
    except Exception as e:
        st.warning(f"Could not generate model comparison plots: {e}")
        return None


def plot_model_correlation_matrix(data_df, model_columns):
    """Create correlation matrix for different sentiment models"""
    try:
        # Create a copy of the data
        df = data_df.copy()
        
        # Extract model scores with explicit conversion to numeric
        model_data = pd.DataFrame()
        
        # Get scores for each model and ensure they're numeric
        for model in model_columns:
            col_name = f'{model}_score'
            if col_name in df.columns:
                model_data[model.capitalize()] = pd.to_numeric(df[col_name], errors='coerce')
            else:
                st.warning(f"Column {col_name} not found in data")
                return None
        
        # Add the ensemble score - try both possible column names
        if 'Sentiment_Score' in df.columns:
            model_data['Ensemble'] = pd.to_numeric(df['Sentiment_Score'], errors='coerce')
        elif 'Sentiment Score' in df.columns:
            model_data['Ensemble'] = pd.to_numeric(df['Sentiment Score'], errors='coerce')
        else:
            st.warning("Ensemble score column not found in data")
            return None
        
        # Drop rows with NA values
        model_data = model_data.dropna()
        
        if len(model_data) == 0:
            st.warning("No valid data for correlation calculation after removing NA values")
            return None
            
        # Calculate correlation matrix
        corr_matrix = model_data.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            text=np.round(corr_matrix.values, 3),
            texttemplate='%{text}',
            textfont={"size": 14},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Model Correlation Matrix",
            height=450,
            width=450,
            xaxis_title="",
            yaxis_title="",
            yaxis_autorange='reversed'
        )
        
        return fig
        
    except Exception as e:
        st.warning(f"Could not generate correlation matrix: {e}")
        return None