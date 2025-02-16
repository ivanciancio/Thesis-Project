import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from analysers.prediction_analyser import MarketPredictionAnalyser

def plot_predictions(historical_data, market_predictions, sentiment_predictions, actual_prices=None):
    """Create visualisation comparing both prediction types"""
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
    prediction_dates = [last_date + timedelta(days=i+1) for i in range(5)]

    # Plot market-only predictions
    fig.add_trace(
        go.Scatter(
            x=prediction_dates,
            y=market_predictions,
            name='Market Prediction',
            line=dict(color='orange', width=2, dash='dash')
        )
    )

    # Plot sentiment-enhanced predictions
    fig.add_trace(
        go.Scatter(
            x=prediction_dates,
            y=sentiment_predictions,
            name='Sentiment-Enhanced Prediction',
            line=dict(color='green', width=2, dash='dash')
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

    fig.update_layout(
        title="5-Day Price Predictions",
        xaxis_title="Date",
        yaxis_title="Price",
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        plot_bgcolor='white',
        hovermode='x unified'
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

    # Add explanation of metrics
    st.markdown("""
    ### Understanding the Metrics
    
    1. **Mean Absolute Error (MAE)**
       - Average absolute difference between predicted and actual prices
       - Lower is better
       
    2. **Mean Absolute Percentage Error (MAPE)**
       - Average percentage difference from actual prices
       - Shows prediction accuracy as a percentage
       
    3. **Directional Accuracy**
       - How often the model correctly predicts price movement direction
       - Higher is better (50% would be random chance)
    """)

    # Add model comparison summary
    if metrics['improvement']['mae_reduction'] > 0:
        st.success(f"""
            📈 The sentiment-enhanced model shows improvement:
            - {metrics['improvement']['mae_reduction']:.1f}% reduction in prediction error
            - {metrics['improvement']['direction_improvement']:.1f}% improvement in directional accuracy
        """)
    else:
        st.warning(f"""
            ⚠️ The market-only model currently performs better:
            - Sentiment model has {-metrics['improvement']['mae_reduction']:.1f}% higher prediction error
            - {-metrics['improvement']['direction_improvement']:.1f}% lower directional accuracy
        """)

    return

def price_prediction_page():
    st.title("📈 Price Prediction Analysis")

    # Check if required data exists
    if ('market_data' not in st.session_state or 
        'news_data' not in st.session_state):
        st.warning("⚠️ Please complete market and sentiment analysis first")
        st.info("""
            To make predictions, we need:
            1. Market data (from the Market Data page)
            2. News sentiment data (from the Sentiment Analysis page)
            3. Reddit sentiment data (optional, but recommended)
        """)
        return

    try:
        # Create tabs for different sections
        tabs = st.tabs(["Model Training", "Predictions", "Performance Analysis"])

        with tabs[0]:
            st.header("Model Training")
            
            # Model parameters
            col1, col2 = st.columns(2)
            with col1:
                epochs = st.number_input("Training Epochs", min_value=10, max_value=200, value=50)
                sequence_length = st.number_input("Sequence Length (Days)", min_value=2, max_value=30, value=5)
            
            with col2:
                batch_size = st.number_input("Batch Size", min_value=8, max_value=64, value=32)
                validation_split = st.slider("Validation Split", min_value=0.1, max_value=0.3, value=0.2)

            if st.button("Train Models"):
                with st.spinner("Training prediction models..."):
                    # Display data diagnostics
                    st.info("Checking available data...")
                    
                    if st.session_state.market_data is None or len(st.session_state.market_data) == 0:
                        st.error("No market data available")
                        return
                        
                    if st.session_state.news_data is None or len(st.session_state.news_data) == 0:
                        st.error("No news sentiment data available")
                        return
                    
                    # Market Data Diagnostics
                    st.write("📊 Market Data:")
                    st.write(f"- Number of market data points: {len(st.session_state.market_data)}")
                    st.write(f"- Date range: {st.session_state.market_data['Date'].min()} to {st.session_state.market_data['Date'].max()}")
                    
                    # News Sentiment Diagnostics
                    st.write("📰 News Sentiment Data:")
                    st.write(f"- Number of news sentiment points: {len(st.session_state.news_data)}")
                    st.write(f"- Date range: {st.session_state.news_data['Date'].min()} to {st.session_state.news_data['Date'].max()}")
                    
                    # Reddit Data Diagnostics (if available)
                    if 'reddit_data' in st.session_state and st.session_state.reddit_data is not None:
                        st.write("🤖 Reddit Sentiment Data:")
                        st.write(f"- Number of Reddit sentiment points: {len(st.session_state.reddit_data)}")
                        st.write(f"- Date range: {st.session_state.reddit_data['Date'].min()} to {st.session_state.reddit_data['Date'].max()}")

                    # Initialize predictor
                    st.info("Initialising prediction models...")
                    predictor = MarketPredictionAnalyser()
                    
                    # Train market model
                    st.write("Training market-only model...")
                    market_history = predictor.train_market_model(
                        st.session_state.market_data,
                        epochs=epochs,
                        batch_size=batch_size
                    )

                    if market_history:
                        # Train sentiment model
                        st.write("Training sentiment-enhanced model...")
                        sentiment_history = predictor.train_sentiment_model(
                            st.session_state.market_data,
                            st.session_state.news_data,
                            st.session_state.get('reddit_data'),
                            epochs=epochs,
                            batch_size=batch_size
                        )

                        if sentiment_history:
                            st.session_state.predictor = predictor
                            st.success("✅ Models trained successfully!")
                        else:
                            st.error("❌ Failed to train sentiment-enhanced model")
                    else:
                        st.error("❌ Failed to train market-only model")

        with tabs[1]:
            st.header("5-Day Price Predictions")

            if 'predictor' not in st.session_state:
                st.warning("⚠️ Please train the models first")
                return

            try:
                # Make predictions
                market_predictions = st.session_state.predictor.predict_market(
                    st.session_state.market_data
                )
                
                if market_predictions is None:
                    st.error("Failed to generate market predictions")
                    return
                
                sentiment_predictions = st.session_state.predictor.predict_with_sentiment(
                    st.session_state.market_data,
                    st.session_state.news_data,
                    st.session_state.get('reddit_data')
                )
                
                if sentiment_predictions is None:
                    st.error("Failed to generate sentiment-enhanced predictions")
                    return

                # Display predictions
                fig = plot_predictions(
                    st.session_state.market_data.set_index('Date'),
                    market_predictions,
                    sentiment_predictions
                )
                st.plotly_chart(fig, use_container_width=True)

                # Show prediction values
                st.subheader("Predicted Prices")
                last_price = st.session_state.market_data['Close'].iloc[-1]
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

            except Exception as e:
                st.error(f"Error generating predictions: {str(e)}")

        with tabs[2]:
            st.header("Model Performance Analysis")

            if 'predictor' not in st.session_state:
                st.warning("⚠️ Please train the models first")
                return

            try:
                if market_predictions is not None and sentiment_predictions is not None:
                    # Get the latest available actual prices
                    actual_prices = st.session_state.market_data['Close'].iloc[-len(market_predictions):].values
                    metrics = st.session_state.predictor.evaluate_predictions(
                        actual_prices,
                        market_predictions,
                        sentiment_predictions
                    )

                    if metrics:
                        display_model_metrics(metrics)
            except Exception as e:
                st.error(f"Error in performance analysis: {str(e)}")

    except Exception as e:
        st.error(f"❌ Error in prediction analysis: {str(e)}")
        if st.button("Show detailed error information"):
            st.exception(e)

if __name__ == "__main__":
    price_prediction_page()