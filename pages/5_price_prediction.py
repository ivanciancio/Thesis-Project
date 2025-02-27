import os
import sys
from pathlib import Path

# Get the absolute path to the project root directory
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent  # This goes up one level from the 'pages' directory

# Add the project root to the Python path
sys.path.append(str(project_root))

# Now continue with the imports
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from analysers.prediction_analyser import MarketPredictionAnalyser, MultiModelPredictionAnalyser

def prepare_data_for_prediction(market_data, news_data, reddit_data=None, twitter_data=None):
    """Prepare data for prediction with improved column detection and error handling"""
    try:
        # Create copies of input data
        market_data = market_data.copy()
        news_data = news_data.copy()
        
        # Ensure datetime columns
        market_data['Date'] = pd.to_datetime(market_data['Date'])
        news_data['Date'] = pd.to_datetime(news_data['Date'])
        
        # Convert to date for merging
        market_data['DateOnly'] = market_data['Date'].dt.date
        news_data['DateOnly'] = news_data['Date'].dt.date
        
        # Find sentiment column in news data
        news_sentiment_col = None
        for col in news_data.columns:
            if 'sentiment' in col.lower() and 'score' in col.lower():
                news_sentiment_col = col
                break
                
        if news_sentiment_col is None:
            st.error("Could not find sentiment score column in news data")
            return None
        
        # Calculate market features
        market_data['Returns'] = market_data['Close'].pct_change()
        market_data['Volatility'] = market_data['Returns'].rolling(window=2, min_periods=1).std()
        market_data['MA5'] = market_data['Close'].rolling(window=3, min_periods=1).mean()
        market_data['price_momentum'] = market_data['Close'].pct_change(2)
        market_data['volume_momentum'] = market_data['Volume'].pct_change(2)
        market_data['recent_trend'] = market_data['Close'].pct_change(2).rolling(window=2, min_periods=1).mean()
        
        # Calculate daily news sentiment
        daily_news = news_data.groupby('DateOnly')[news_sentiment_col].mean().reset_index()
        daily_news.columns = ['DateOnly', 'News_Sentiment']
        
        # Merge market data with news sentiment
        combined_data = pd.merge(market_data, daily_news, on='DateOnly', how='left')
        
        # Fill missing sentiment values with 0
        combined_data['News_Sentiment'] = combined_data['News_Sentiment'].fillna(0)
        
        # Add Twitter data if available
        if twitter_data is not None and not twitter_data.empty:
            twitter_data = twitter_data.copy()
            
            # Find Twitter sentiment column
            twitter_sentiment_col = None
            for col in twitter_data.columns:
                if 'sentiment' in col.lower() and 'score' in col.lower():
                    twitter_sentiment_col = col
                    break
            
            if twitter_sentiment_col:
                # Convert date and create DateOnly
                twitter_data['Date'] = pd.to_datetime(twitter_data['Date'])
                twitter_data['DateOnly'] = twitter_data['Date'].dt.date
                
                # Group by date
                daily_twitter = twitter_data.groupby('DateOnly')[twitter_sentiment_col].mean().reset_index()
                daily_twitter.columns = ['DateOnly', 'Twitter_Sentiment']
                
                # Merge with combined data
                combined_data = pd.merge(combined_data, daily_twitter, on='DateOnly', how='left')
                combined_data['Twitter_Sentiment'] = combined_data['Twitter_Sentiment'].fillna(0)
        
        # Add Reddit data if available
        if reddit_data is not None and not reddit_data.empty:
            reddit_data = reddit_data.copy()
            
            # Find Reddit sentiment column
            reddit_sentiment_col = None
            for col in reddit_data.columns:
                if 'sentiment' in col.lower() and 'score' in col.lower():
                    reddit_sentiment_col = col
                    break
            
            if reddit_sentiment_col:
                # Convert date and create DateOnly
                reddit_data['Date'] = pd.to_datetime(reddit_data['Date'])
                reddit_data['DateOnly'] = reddit_data['Date'].dt.date
                
                # Group by date
                daily_reddit = reddit_data.groupby('DateOnly')[reddit_sentiment_col].mean().reset_index()
                daily_reddit.columns = ['DateOnly', 'Reddit_Sentiment']
                
                # Merge with combined data
                combined_data = pd.merge(combined_data, daily_reddit, on='DateOnly', how='left')
                combined_data['Reddit_Sentiment'] = combined_data['Reddit_Sentiment'].fillna(0)
        
        # Sort by date
        combined_data = combined_data.sort_values('Date')
        
        # Forward fill any remaining NaN values
        combined_data = combined_data.ffill().bfill()
        
        st.success(f"Successfully prepared prediction data with {len(combined_data)} rows")
        return combined_data
        
    except Exception as e:
        st.error(f"Error preparing prediction data: {str(e)}")
        return None

def plot_predictions(historical_data, market_predictions, sentiment_predictions, actual_prices=None):
    """Create visualization comparing prediction types with enhanced visibility"""
    fig = go.Figure()

    # Plot historical data
    fig.add_trace(
        go.Scatter(
            x=historical_data.index[-30:],  # Last 30 days
            y=historical_data['Close'][-30:],
            name='Historical',
            line=dict(color='blue', width=2)
        )
    )

    # Calculate prediction dates
    last_date = historical_data.index[-1]
    prediction_dates = [last_date + timedelta(days=i+1) for i in range(len(market_predictions))]

    # Plot market-only predictions with enhanced visibility
    fig.add_trace(
        go.Scatter(
            x=prediction_dates,
            y=market_predictions,
            name='Market Prediction',
            mode='lines+markers',  # Added markers for better visibility
            line=dict(color='orange', width=3),  # Made line thicker
            marker=dict(size=8, symbol='circle')
        )
    )

    # Plot sentiment-enhanced predictions with enhanced visibility
    fig.add_trace(
        go.Scatter(
            x=prediction_dates,
            y=sentiment_predictions,
            name='Sentiment-Enhanced',
            mode='lines+markers',  # Added markers for better visibility
            line=dict(color='green', width=3),  # Made line thicker
            marker=dict(size=8, symbol='diamond')
        )
    )

    # Plot actual prices if available
    if actual_prices is not None:
        fig.add_trace(
            go.Scatter(
                x=prediction_dates,
                y=actual_prices,
                name='Actual Prices',
                line=dict(color='red', width=2)
            )
        )

    # Update layout with clearer formatting
    fig.update_layout(
        title="Price Predictions",
        xaxis_title="Date",
        yaxis_title="Price",
        height=600,
        showlegend=True,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig

def plot_multi_model_predictions(historical_data, model_predictions):
    """Create visualization comparing predictions from multiple models"""
    fig = go.Figure()

    # Plot historical data
    fig.add_trace(
        go.Scatter(
            x=historical_data.index[-30:],  # Last 30 days
            y=historical_data['Close'][-30:],
            name='Historical',
            line=dict(color='blue', width=2)
        )
    )

    # Calculate prediction dates
    last_date = historical_data.index[-1]
    sample_predictions = next(iter(model_predictions.values()))
    prediction_dates = [last_date + timedelta(days=i+1) for i in range(len(sample_predictions))]

    # Define model colors
    model_colors = {
        'market': '#666666',
        'ensemble': '#9b59b6',
        'textblob': '#3498db',
        'vader': '#2ecc71',
        'finbert': '#e74c3c'
    }

    # Plot each model's predictions
    for model_name, predictions in model_predictions.items():
        fig.add_trace(
            go.Scatter(
                x=prediction_dates,
                y=predictions,
                name=model_name.capitalize(),
                mode='lines+markers',
                line=dict(
                    color=model_colors.get(model_name, '#000'),
                    width=3 if model_name in ['market', 'ensemble'] else 2,
                    dash='dash' if model_name == 'market' else None
                ),
                marker=dict(
                    size=8 if model_name in ['market', 'ensemble'] else 6,
                    symbol='circle'
                )
            )
        )

    # Update layout
    fig.update_layout(
        title="Price Predictions by Model",
        xaxis_title="Date",
        yaxis_title="Price",
        height=600,
        showlegend=True,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig

def display_model_metrics(metrics):
    """Display comprehensive model performance metrics."""
    st.subheader("Model Performance Comparison")

    # Create three columns for metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Market-Only Model")
        st.metric(
            "Mean Absolute Error",
            f"${metrics['market_model']['mae']:.2f}",
            delta=None
        )
        st.metric(
            "MAPE",
            f"{metrics['market_model']['mape']:.1f}%",
            delta=None
        )
        st.metric(
            "Directional Accuracy",
            f"{metrics['market_model']['directional_accuracy']*100:.1f}%",
            delta=None
        )

    with col2:
        st.markdown("### Sentiment-Enhanced Model")
        st.metric(
            "Mean Absolute Error",
            f"${metrics['sentiment_model']['mae']:.2f}",
            delta=f"-{metrics['improvement']['mae_reduction']:.1f}%" 
            if metrics['improvement']['mae_reduction'] > 0 else 
            f"{-metrics['improvement']['mae_reduction']:.1f}%"
        )
        st.metric(
            "MAPE",
            f"{metrics['sentiment_model']['mape']:.1f}%",
            delta=f"-{metrics['improvement']['mape_reduction']:.1f}%" 
            if metrics['improvement']['mape_reduction'] > 0 else 
            f"{-metrics['improvement']['mape_reduction']:.1f}%"
        )
        st.metric(
            "Directional Accuracy",
            f"{metrics['sentiment_model']['directional_accuracy']*100:.1f}%",
            delta=f"{metrics['improvement']['direction_improvement']:.1f}%"
        )

    with col3:
        st.markdown("### Model Improvement")
        st.markdown(f"""
        ##### Error Reduction
        - MAE: {metrics['improvement']['mae_reduction']:.1f}%
        - MAPE: {metrics['improvement']['mape_reduction']:.1f}%
        
        ##### Accuracy Improvement
        - Direction: {metrics['improvement']['direction_improvement']:.1f}%
        """)

def display_multi_model_metrics(model_predictions, last_price):
    """Display metrics for predictions from multiple models"""
    st.subheader("Model Predictions Comparison")
    
    # Calculate model metrics
    metrics = []
    
    for model_name, predictions in model_predictions.items():
        # Calculate change from last price
        first_day_change = (predictions[0] - last_price) / last_price * 100
        period_change = (predictions[-1] - last_price) / last_price * 100
        
        # Calculate trend direction
        trend = "Up" if predictions[-1] > predictions[0] else "Down" if predictions[-1] < predictions[0] else "Flat"
        
        # Add to metrics
        metrics.append({
            'Model': model_name.capitalize(),
            'First Day': predictions[0],
            'Last Day': predictions[-1],
            'First Day Change': first_day_change,
            'Period Change': period_change,
            'Trend': trend
        })
    
    # Convert to DataFrame for display
    metrics_df = pd.DataFrame(metrics)
    
    # Display metrics table
    st.dataframe(
        metrics_df.style.format({
            'First Day': '${:,.2f}',
            'Last Day': '${:,.2f}',
            'First Day Change': '{:+.2f}%',
            'Period Change': '{:+.2f}%'
        }),
        use_container_width=True
    )
    
    # Find model with best and worst performance
    best_model = metrics_df.loc[metrics_df['Period Change'].idxmax()]
    worst_model = metrics_df.loc[metrics_df['Period Change'].idxmin()]
    
    # Display best and worst models
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**Most Optimistic: {best_model['Model']}**  \n"
                f"Predicts {best_model['Period Change']:+.2f}% change")
    
    with col2:
        st.info(f"**Most Pessimistic: {worst_model['Model']}**  \n"
                f"Predicts {worst_model['Period Change']:+.2f}% change")

def price_prediction_page():
    st.title("📈 Price Prediction Analysis")

    # Check for required data
    required_data = ['market_data', 'news_data']
    if not all(key in st.session_state for key in required_data):
        st.warning("⚠️ Please complete market and sentiment analysis first")
        return

    try:
        # Get available data sources
        data_sources = {
            'Market Data': st.session_state.market_data,
            'News Data': st.session_state.news_data,
            'Reddit Data': st.session_state.get('reddit_data'),
            'Twitter Data': st.session_state.get('twitter_data')
        }
        
        # Display data availability
        st.subheader("Available Data Sources")
        for source, data in data_sources.items():
            if data is not None and not data.empty:
                st.success(f"✅ {source} available")
                if 'Date' in data.columns:
                    date_range = f"({data['Date'].min().strftime('%Y-%m-%d')} to {data['Date'].max().strftime('%Y-%m-%d')})"
                    st.info(f"{source} date range: {date_range}")
            else:
                st.warning(f"⚠️ {source} not available")
        
        # Model training section
        st.header("Model Training")
        
        col1, col2 = st.columns(2)
        with col1:
            sequence_length = st.number_input("Sequence Length (Days)", min_value=2, max_value=10, value=3)
        
        with col2:
            prediction_days = st.number_input("Prediction Days", min_value=1, max_value=5, value=2)

        # Option to use multiple sentiment models
        multi_model = st.checkbox("Use multiple sentiment models for comparison", value=True)

        if st.button("Train Models"):
            with st.spinner("Preparing data for training..."):
                # Prepare combined data
                combined_data = prepare_data_for_prediction(
                    data_sources['Market Data'],
                    data_sources['News Data'],
                    data_sources['Reddit Data'],
                    data_sources['Twitter Data']
                )
                
                if combined_data is None:
                    st.error("Failed to prepare prediction data")
                    return
                
                # Store combined data in session state
                st.session_state.combined_data = combined_data
                
                if multi_model and 'available_models' in st.session_state:
                    # Train multi-model predictor
                    predictor = MultiModelPredictionAnalyser()
                    predictor.sequence_length = sequence_length
                    predictor.prediction_days = prediction_days
                    
                    # Train market model
                    with st.spinner("Training market-only model..."):
                        market_success = predictor.train_market_model(combined_data)
                        
                        if market_success:
                            st.success("✅ Market model trained successfully!")
                        else:
                            st.error("❌ Failed to train market model")
                            return
                    
                    # Train ensemble sentiment model
                    with st.spinner("Training ensemble sentiment model..."):
                        sentiment_success = predictor.train_sentiment_model(
                            combined_data,
                            data_sources['News Data']
                        )
                        
                        if sentiment_success:
                            st.success("✅ Ensemble sentiment model trained successfully!")
                        else:
                            st.error("❌ Failed to train ensemble sentiment model")
                            return
                    
                    # Train model-specific sentiment models
                    with st.spinner("Training model-specific sentiment models..."):
                        predictor.train_model_specific_models(
                            combined_data,
                            data_sources['News Data'],
                            st.session_state.available_models
                        )
                    
                    st.session_state.predictor = predictor
                    st.session_state.models_trained = True
                    st.session_state.multi_model = True
                    
                else:
                    # Train standard predictor
                    predictor = MarketPredictionAnalyser()
                    predictor.sequence_length = sequence_length
                    predictor.prediction_days = prediction_days
                    
                    # Train market model
                    with st.spinner("Training market-only model..."):
                        market_success = predictor.train_market_model(combined_data)
                        
                        if market_success:
                            st.success("✅ Market model trained successfully!")
                        else:
                            st.error("❌ Failed to train market model")
                            return
                    
                    # Train sentiment model
                    with st.spinner("Training sentiment model..."):
                        sentiment_success = predictor.train_sentiment_model(
                            combined_data,
                            data_sources['News Data']
                        )
                        
                        if sentiment_success:
                            st.success("✅ Sentiment model trained successfully!")
                            st.session_state.predictor = predictor
                            st.session_state.models_trained = True
                            st.session_state.multi_model = False
                        else:
                            st.error("❌ Failed to train sentiment model")
                            return

        # Predictions section
        if 'models_trained' in st.session_state and st.session_state.models_trained:
            st.header("Price Predictions")
            
            if 'combined_data' not in st.session_state:
                st.warning("Please train the models first to see predictions")
                return
                
            # Make predictions
            with st.spinner("Generating predictions..."):
                if st.session_state.multi_model:
                    # Get predictions from all models
                    model_predictions = st.session_state.predictor.predict_all_models(
                        st.session_state.combined_data,
                        data_sources['News Data'],
                        st.session_state.available_models
                    )
                    
                    if model_predictions:
                        # Plot all model predictions
                        fig = plot_multi_model_predictions(
                            st.session_state.combined_data.set_index('Date'),
                            model_predictions
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display metrics comparison
                        last_price = st.session_state.combined_data['Close'].iloc[-1]
                        display_multi_model_metrics(model_predictions, last_price)
                        
                else:
                    # Standard prediction with just market and ensemble
                    market_predictions = st.session_state.predictor.predict_market(st.session_state.combined_data)
                    sentiment_predictions = st.session_state.predictor.predict_with_sentiment(
                        st.session_state.combined_data,
                        data_sources['News Data']
                    )
                    
                    if market_predictions is not None and sentiment_predictions is not None:
                        # Plot predictions
                        fig = plot_predictions(
                            st.session_state.combined_data.set_index('Date'),
                            market_predictions,
                            sentiment_predictions
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show numerical predictions
                        st.subheader("Predicted Values")
                        last_price = st.session_state.combined_data['Close'].iloc[-1]
                        dates = [(datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d') 
                                for i in range(len(market_predictions))]
                        
                        df_predictions = pd.DataFrame({
                            'Date': dates,
                            'Market Model': market_predictions,
                            'Sentiment Model': sentiment_predictions,
                            'Market Change %': ((market_predictions - last_price) / last_price * 100),
                            'Sentiment Change %': ((sentiment_predictions - last_price) / last_price * 100)
                        })
                        
                        st.dataframe(
                            df_predictions.style.format({
                                'Market Model': '${:,.2f}',
                                'Sentiment Model': '${:,.2f}',
                                'Market Change %': '{:,.2f}%',
                                'Sentiment Change %': '{:,.2f}%'
                            }),
                            use_container_width=True
                        )
        else:
            st.info("👆 Please train the models first to see predictions")

    except Exception as e:
        st.error(f"Error in prediction analysis: {str(e)}")

if __name__ == "__main__":
    price_prediction_page()