import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import logging
import streamlit as st

class MarketPredictionAnalyser:
    def __init__(self):
        """Initialise the prediction analyser"""
        self.market_scaler = MinMaxScaler()
        self.sentiment_scaler = MinMaxScaler()
        self.market_model = None
        self.sentiment_enhanced_model = None
        self.sequence_length = 5    # Increased from 2 to 5
        self.prediction_days = 2
        self.market_features = ['Close', 'Volume', 'Returns', 'Volatility', 
                            'MA5', 'price_momentum', 'volume_momentum', 
                            'recent_trend']  # Updated feature list
        self.used_features = None
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def prepare_market_data(self, market_data):
        """Prepare market data for Random Forest model."""
        try:
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Debug: Show initial data shape
            st.write(f"Initial market data shape: {market_data.shape}")
            
            data = market_data.copy()
            
            # Calculate base features first
            data['Returns'] = data['Close'].pct_change()
            data['Volatility'] = data['Returns'].rolling(window=3).std()
            data['MA5'] = data['Close'].rolling(window=3).mean()
            
            # Add momentum indicators
            data['price_momentum'] = data['Close'].pct_change(5)  # 5-day momentum
            data['volume_momentum'] = data['Volume'].pct_change(5)
            data['recent_trend'] = data['Close'].pct_change(3).rolling(window=3).mean()
            
            # Debug: Show data after calculations
            st.write("After technical indicators:")
            st.write(data.head())
            
            # Remove NaN values
            data = data.dropna()
            
            # Debug: Show data after removing NaN
            st.write(f"Data shape after removing NaN: {data.shape}")
            
            min_required_points = self.sequence_length + self.prediction_days
            if len(data) < min_required_points:
                st.error(f"Not enough data points. Have {len(data)}, need at least {min_required_points}")
                return None, None
            
            # Create sequences using all features
            X = []
            y = []
            features = ['Close', 'Volume', 'Returns', 'Volatility', 'MA5', 
                       'price_momentum', 'volume_momentum', 'recent_trend']
            
            # Debug: Show feature list
            st.write(f"Using features: {features}")
            
            for i in range(len(data) - self.sequence_length - self.prediction_days + 1):
                sequence = data.iloc[i:(i + self.sequence_length)][features].values
                target = data.iloc[i + self.sequence_length:i + self.sequence_length + self.prediction_days]['Close'].values
                
                if len(target) == self.prediction_days:
                    X.append(sequence.flatten())
                    y.append(target)
            
            # Debug: Show sequence info
            st.write(f"Number of sequences created: {len(X)}")
            if len(X) > 0:
                st.write(f"Shape of first sequence: {len(X[0])}")
            
            # Convert to numpy arrays
            X = np.array(X)
            y = np.array(y)
            
            if len(X) == 0 or len(y) == 0:
                st.error("No valid sequences could be created")
                return None, None
            
            progress_bar.progress(100)
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error preparing market data: {str(e)}")
            st.error(f"Error preparing market data: {str(e)}")
            return None, None
        finally:
            if 'progress_bar' in locals():
                progress_bar.empty()
            if 'status_text' in locals():
                status_text.empty()
        """Prepare market data for Random Forest model."""
        try:
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("Preparing market data...")
            
            # Ensure we have the required columns
            required_columns = ['Date', 'Close', 'Volume']
            if not all(col in market_data.columns for col in required_columns):
                raise ValueError(f"Missing required columns. Need: {required_columns}")
            
            data = market_data.copy()
            
            # Calculate technical indicators
            status_text.text("Calculating technical indicators...")
            progress_bar.progress(20)
            
            data['Returns'] = data['Close'].pct_change()
            data['Volatility'] = data['Returns'].rolling(window=5).std()
            data['MA5'] = data['Close'].rolling(window=5).mean()
            data['MA20'] = data['Close'].rolling(window=20).mean()
            
            # Remove NaN values
            data = data.dropna()
            
            if len(data) < self.sequence_length + self.prediction_days:
                raise ValueError("Not enough data points after preprocessing")
            
            progress_bar.progress(40)
            status_text.text("Creating feature sequences...")
            
            # Create sequences for features and targets
            X = []
            y = []
            features = ['Close', 'Volume', 'Returns', 'Volatility', 'MA5', 'MA20']
            
            for i in range(len(data) - self.sequence_length - self.prediction_days):
                sequence = data.iloc[i:(i + self.sequence_length)][features].values
                target = data.iloc[i + self.sequence_length:i + self.sequence_length + self.prediction_days]['Close'].values
                
                if len(target) == self.prediction_days:  # Ensure we have enough target values
                    X.append(sequence.flatten())  # Flatten the sequence
                    y.append(target)
            
            progress_bar.progress(80)
            status_text.text("Finalising data preparation...")
            
            # Convert to numpy arrays and reshape
            X = np.array(X)
            y = np.array(y)
            
            if len(X) == 0 or len(y) == 0:
                raise ValueError("No valid sequences could be created from the data")
            
            progress_bar.progress(100)
            status_text.text("Data preparation complete!")
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error preparing market data: {str(e)}")
            status_text.text(f"Error: {str(e)}")
            return None, None
        finally:
            progress_bar.empty()
            status_text.empty()

    def prepare_sentiment_enhanced_data(self, market_data, news_sentiment, social_sentiment=None):
        """Prepare combined market and sentiment data."""
        try:
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("Preparing combined market and sentiment data...")
            
            # Debug: Show initial shapes
            st.write(f"Initial market data shape: {market_data.shape}")
            st.write(f"Initial news sentiment shape: {news_sentiment.shape}")
            
            data = market_data.copy()
            
            # Calculate market features
            data['Returns'] = data['Close'].pct_change()
            data['Volatility'] = data['Returns'].rolling(window=3).std()
            data['MA5'] = data['Close'].rolling(window=3).mean()
            
            # Add momentum indicators
            data['price_momentum'] = data['Close'].pct_change(5)  # 5-day momentum
            data['volume_momentum'] = data['Volume'].pct_change(5)
            data['recent_trend'] = data['Close'].pct_change(3).rolling(window=3).mean()
            
            # Debug: Show after market features
            st.write("Market features calculated")
            
            # Process and align sentiment data
            status_text.text("Processing sentiment data...")
            progress_bar.progress(40)
            
            # Ensure dates are in the correct format
            data['Date'] = pd.to_datetime(data['Date']).dt.date
            news_sentiment['Date'] = pd.to_datetime(news_sentiment['Date']).dt.date
            
            # Calculate daily sentiment averages
            daily_sentiment = (news_sentiment.groupby('Date')['Sentiment Score']
                             .mean()
                             .reset_index())
            
            # Debug: Show daily sentiment
            st.write("Daily sentiment calculated:")
            st.write(daily_sentiment.head())
            
            # Merge market data with sentiment
            data = pd.merge(data, 
                          daily_sentiment,
                          on='Date',
                          how='left')
            
            # Debug: Show after merge
            st.write(f"Shape after merging sentiment: {data.shape}")
            
            # Add social sentiment if available
            if social_sentiment is not None:
                social_sentiment['Date'] = pd.to_datetime(social_sentiment['Date']).dt.date
                daily_social = (social_sentiment.groupby('Date')['Sentiment_Score']
                              .mean()
                              .reset_index())
                
                data = pd.merge(data, 
                              daily_social,
                              on='Date',
                              how='left')
            
            # Fill any missing sentiment values with 0
            data['Sentiment Score'] = data['Sentiment Score'].fillna(0)
            if 'Sentiment_Score' in data.columns:
                data['Sentiment_Score'] = data['Sentiment_Score'].fillna(0)
            
            # Remove any remaining NaN values
            data = data.dropna()
            
            # Debug: Show data after cleaning
            st.write(f"Final shape after cleaning: {data.shape}")
            
            # Create sequences
            X = []
            y = []
            features = ['Close', 'Volume', 'Returns', 'Volatility', 'MA5', 
                       'price_momentum', 'volume_momentum', 'recent_trend',
                       'Sentiment Score']
            if social_sentiment is not None:
                features.append('Sentiment_Score')
            
            # Store the features used for prediction later
            self.used_features = features.copy()
            
            # Debug: Show features
            st.write(f"Using features: {features}")
            
            for i in range(len(data) - self.sequence_length - self.prediction_days + 1):
                sequence = data.iloc[i:(i + self.sequence_length)][features].values
                target = data.iloc[i + self.sequence_length:i + self.sequence_length + self.prediction_days]['Close'].values
                
                if len(target) == self.prediction_days:
                    X.append(sequence.flatten())
                    y.append(target)
            
            # Debug: Show sequence info
            st.write(f"Number of sequences created: {len(X)}")
            
            if len(X) == 0:
                raise ValueError("No valid sequences could be created")
            
            # Convert to numpy arrays
            X = np.array(X)
            y = np.array(y)
            
            progress_bar.progress(100)
            status_text.text("Data preparation complete!")
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error preparing sentiment-enhanced data: {str(e)}")
            st.error(f"Error preparing sentiment-enhanced data: {str(e)}")
            return None, None
        finally:
            progress_bar.empty()
            status_text.empty()
        """Prepare combined market and sentiment data."""
        try:
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("Preparing combined market and sentiment data...")
            
            # Debug: Show initial shapes
            st.write(f"Initial market data shape: {market_data.shape}")
            st.write(f"Initial news sentiment shape: {news_sentiment.shape}")
            
            data = market_data.copy()
            
            # Calculate market features
            data['Returns'] = data['Close'].pct_change()
            data['Volatility'] = data['Returns'].rolling(window=3).std()
            data['MA5'] = data['Close'].rolling(window=3).mean()
            
            # Debug: Show after market features
            st.write("Market features calculated")
            
            # Process and align sentiment data
            status_text.text("Processing sentiment data...")
            progress_bar.progress(40)
            
            # Convert dates to datetime
            data['Date'] = pd.to_datetime(data['Date'])
            news_sentiment['Date'] = pd.to_datetime(news_sentiment['Date'])
            
            # Calculate daily sentiment averages
            daily_sentiment = news_sentiment.groupby(news_sentiment['Date'].dt.date)['Sentiment Score'].mean().reset_index()
            daily_sentiment['Date'] = pd.to_datetime(daily_sentiment['Date'])
            
            # Debug: Show daily sentiment
            st.write("Daily sentiment calculated:")
            st.write(daily_sentiment.head())
            
            # Merge market data with sentiment
            data = pd.merge(data, 
                          daily_sentiment, 
                          left_on=data['Date'].dt.date,
                          right_on='Date',
                          how='left',
                          suffixes=('', '_sentiment'))
            
            # Debug: Show after merge
            st.write(f"Shape after merging sentiment: {data.shape}")
            
            if social_sentiment is not None:
                social_sentiment['Date'] = pd.to_datetime(social_sentiment['Date'])
                daily_social = social_sentiment.groupby(social_sentiment['Date'].dt.date)['Sentiment_Score'].mean().reset_index()
                daily_social['Date'] = pd.to_datetime(daily_social['Date'])
                
                data = pd.merge(data, 
                              daily_social,
                              left_on=data['Date'].dt.date,
                              right_on='Date',
                              how='left',
                              suffixes=('', '_social'))
            
            # Remove NaN values
            data = data.dropna()
            
            # Debug: Show data after cleaning
            st.write(f"Final shape after cleaning: {data.shape}")
            
            if len(data) < self.sequence_length + self.prediction_days:
                raise ValueError(f"Not enough data points. Have {len(data)}, need {self.sequence_length + self.prediction_days}")
            
            # Create sequences
            X = []
            y = []
            features = ['Close', 'Volume', 'Returns', 'Volatility', 'MA5', 'Sentiment Score']
            if social_sentiment is not None:
                features.append('Sentiment_Score')
            
            # Debug: Show features
            st.write(f"Using features: {features}")
            
            for i in range(len(data) - self.sequence_length - self.prediction_days + 1):
                sequence = data.iloc[i:(i + self.sequence_length)][features].values
                target = data.iloc[i + self.sequence_length:i + self.sequence_length + self.prediction_days]['Close'].values
                
                if len(target) == self.prediction_days:
                    X.append(sequence.flatten())
                    y.append(target)
            
            # Debug: Show sequence info
            st.write(f"Number of sequences created: {len(X)}")
            
            if len(X) == 0:
                raise ValueError("No valid sequences could be created")
            
            # Convert to numpy arrays
            X = np.array(X)
            y = np.array(y)
            
            progress_bar.progress(100)
            status_text.text("Data preparation complete!")
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error preparing sentiment-enhanced data: {str(e)}")
            st.error(f"Error preparing sentiment-enhanced data: {str(e)}")
            return None, None
        finally:
            progress_bar.empty()
            status_text.empty()

    def build_market_model(self):
        """Build Random Forest model for pure market data predictions."""
        self.market_model = RandomForestRegressor(
            n_estimators=50,
            max_depth=5,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )

    def build_sentiment_model(self):
        """Build Random Forest model for sentiment-enhanced predictions."""
        self.sentiment_enhanced_model = RandomForestRegressor(
            n_estimators=50,
            max_depth=5,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )

    def train_market_model(self, market_data, **kwargs):
        """Train the pure market data prediction model."""
        try:
            X, y = self.prepare_market_data(market_data)
            if X is None or y is None:
                return None
            
            st.text("Training market model...")
            progress_bar = st.progress(0)
            
            self.build_market_model()
            self.market_model.fit(X, y)
            
            progress_bar.progress(100)
            return {'status': 'success'}
            
        except Exception as e:
            self.logger.error(f"Error training market model: {str(e)}")
            st.error(f"Error training market model: {str(e)}")
            return None
        finally:
            if 'progress_bar' in locals():
                progress_bar.empty()

    def train_sentiment_model(self, market_data, news_sentiment, social_sentiment=None, **kwargs):
        """Train the sentiment-enhanced prediction model."""
        try:
            X, y = self.prepare_sentiment_enhanced_data(market_data, news_sentiment, social_sentiment)
            if X is None or y is None:
                return None
            
            st.text("Training sentiment-enhanced model...")
            progress_bar = st.progress(0)
            
            self.build_sentiment_model()
            self.sentiment_enhanced_model.fit(X, y)
            
            progress_bar.progress(100)
            return {'status': 'success'}
            
        except Exception as e:
            self.logger.error(f"Error training sentiment model: {str(e)}")
            st.error(f"Error training sentiment model: {str(e)}")
            return None
        finally:
            if 'progress_bar' in locals():
                progress_bar.empty()

    def predict_market(self, market_data):
        """Make predictions using pure market data."""
        try:
            if self.market_model is None:
                raise ValueError("Market model has not been trained")
            
            # Prepare recent data
            data = market_data.copy()
            
            # Calculate all technical indicators
            data['Returns'] = data['Close'].pct_change()
            data['Volatility'] = data['Returns'].rolling(window=3).std()
            data['MA5'] = data['Close'].rolling(window=3).mean()
            data['price_momentum'] = data['Close'].pct_change(5)
            data['volume_momentum'] = data['Volume'].pct_change(5)
            data['recent_trend'] = data['Close'].pct_change(3).rolling(window=3).mean()
            
            # Remove NaN values
            data = data.dropna()
            
            # Get most recent sequence
            features = ['Close', 'Volume', 'Returns', 'Volatility', 'MA5', 
                    'price_momentum', 'volume_momentum', 'recent_trend']
            
            recent_data = data.iloc[-self.sequence_length:][features]
            features = recent_data.values.flatten().reshape(1, -1)
            
            return self.market_model.predict(features)[0]
            
        except Exception as e:
            self.logger.error(f"Error making market prediction: {str(e)}")
            st.error(f"Error making market prediction: {str(e)}")
            return None

    def predict_with_sentiment(self, market_data, news_sentiment, social_sentiment=None):
        """Make predictions using market data and sentiment."""
        try:
            if self.sentiment_enhanced_model is None:
                raise ValueError("Sentiment-enhanced model has not been trained")
            
            # Prepare recent data
            data = market_data.copy()
            
            # Calculate market features
            data['Returns'] = data['Close'].pct_change()
            data['Volatility'] = data['Returns'].rolling(window=3).std()
            data['MA5'] = data['Close'].rolling(window=3).mean()
            data['price_momentum'] = data['Close'].pct_change(5)
            data['volume_momentum'] = data['Volume'].pct_change(5)
            data['recent_trend'] = data['Close'].pct_change(3).rolling(window=3).mean()
            
            # Prepare sentiment data
            data['Date'] = pd.to_datetime(data['Date']).dt.date
            news_sentiment['Date'] = pd.to_datetime(news_sentiment['Date']).dt.date
            
            # Calculate daily sentiment
            daily_sentiment = (news_sentiment.groupby('Date')['Sentiment Score']
                            .mean()
                            .reset_index())
            
            # Merge sentiment data
            data = pd.merge(data, daily_sentiment, on='Date', how='left')
            data['Sentiment Score'] = data['Sentiment Score'].fillna(0)
            
            # Add social sentiment if available
            if social_sentiment is not None:
                social_sentiment['Date'] = pd.to_datetime(social_sentiment['Date']).dt.date
                daily_social = (social_sentiment.groupby('Date')['Sentiment_Score']
                            .mean()
                            .reset_index())
                data = pd.merge(data, daily_social, on='Date', how='left')
                data['Sentiment_Score'] = data['Sentiment_Score'].fillna(0)
            
            # Remove NaN values
            data = data.dropna()
            
            if self.used_features is None:
                features = ['Close', 'Volume', 'Returns', 'Volatility', 'MA5', 
                        'price_momentum', 'volume_momentum', 'recent_trend',
                        'Sentiment Score']
                if social_sentiment is not None:
                    features.append('Sentiment_Score')
            else:
                features = self.used_features
            
            # Get the most recent sequence
            recent_data = data.iloc[-self.sequence_length:][features]
            features_array = recent_data.values.flatten().reshape(1, -1)
            
            return self.sentiment_enhanced_model.predict(features_array)[0]
            
        except Exception as e:
            self.logger.error(f"Error making sentiment-enhanced prediction: {str(e)}")
            st.error(f"Error making sentiment-enhanced prediction: {str(e)}")
            return None
    
    def evaluate_predictions(self, actual_prices, market_predictions, sentiment_predictions):
        """Evaluate and compare both types of predictions."""
        try:
            # Calculate price-based metrics
            market_mse = np.mean((actual_prices - market_predictions) ** 2)
            sentiment_mse = np.mean((actual_prices - sentiment_predictions) ** 2)
            
            market_mae = np.mean(np.abs(actual_prices - market_predictions))
            sentiment_mae = np.mean(np.abs(actual_prices - sentiment_predictions))
            
            # Calculate directional accuracy
            market_direction = np.sign(np.diff(market_predictions))
            sentiment_direction = np.sign(np.diff(sentiment_predictions))
            actual_direction = np.sign(np.diff(actual_prices))
            
            market_dir_acc = np.mean(market_direction == actual_direction)
            sentiment_dir_acc = np.mean(sentiment_direction == actual_direction)
            
            # Calculate percentage errors
            market_mape = np.mean(np.abs((actual_prices - market_predictions) / actual_prices)) * 100
            sentiment_mape = np.mean(np.abs((actual_prices - sentiment_predictions) / actual_prices)) * 100
            
            return {
                'market_model': {
                    'mse': market_mse,
                    'mae': market_mae,
                    'mape': market_mape,
                    'directional_accuracy': market_dir_acc
                },
                'sentiment_model': {
                    'mse': sentiment_mse,
                    'mae': sentiment_mae,
                    'mape': sentiment_mape,
                    'directional_accuracy': sentiment_dir_acc
                },
                'improvement': {
                    'mse_reduction': (market_mse - sentiment_mse) / market_mse * 100,
                    'mae_reduction': (market_mae - sentiment_mae) / market_mae * 100,
                    'mape_reduction': (market_mape - sentiment_mape) / market_mape * 100,
                    'direction_improvement': (sentiment_dir_acc - market_dir_acc) * 100
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating predictions: {str(e)}")
            st.error(f"Error evaluating predictions: {str(e)}")
            return None