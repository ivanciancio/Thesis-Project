import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import logging
import streamlit as st
from utils.date_helpers import prepare_dates_for_merge

class MarketPredictionAnalyser:
    def __init__(self):
        self.sequence_length = 3
        self.prediction_days = 2
        self.market_features = [
            'Open',
            'High',
            'Low',
            'Close',
            'Volume',
            'adjusted_close'
        ]
        self.market_model = None
        self.sentiment_enhanced_model = None
        self.used_features = None
        self.scaler = MinMaxScaler()

    def prepare_market_data(self, market_data):
        try:
            data = market_data.copy()
            
            # Check data length
            if len(data) < self.sequence_length + self.prediction_days:
                st.error(f"Not enough data points. Need at least {self.sequence_length + self.prediction_days}")
                return None, None

            # Verify features exist
            missing_features = [f for f in self.market_features if f not in data.columns]
            if missing_features:
                st.error(f"Missing features: {missing_features}")
                return None, None

            X = []
            y = []
            
            # Use up to second-to-last entry to ensure we have room for prediction
            valid_range = len(data) - self.prediction_days
            
            for i in range(valid_range - self.sequence_length + 1):
                sequence = []
                for j in range(self.sequence_length):
                    row = data.iloc[i + j][self.market_features].values
                    sequence.extend(row)
                
                target = data.iloc[i + self.sequence_length:i + self.sequence_length + self.prediction_days]['Close'].values
                
                if len(target) == self.prediction_days:
                    X.append(sequence)
                    y.append(target)

            if not X:
                st.error("No valid sequences could be created")
                return None, None

            X = np.array(X)
            y = np.array(y)
            X = self.scaler.fit_transform(X)
            
            return X, y
            
        except Exception as e:
            st.error(f"Error in prepare_market_data: {str(e)}")
            return None, None

    def prepare_sentiment_enhanced_data(self, market_data, news_data):
        """
        Prepare market and sentiment data for training with error handling
        """
        try:
            # Input validation
            if market_data is None or news_data is None:
                st.error("Missing input data")
                return None, None
                    
            if len(market_data) < self.sequence_length + self.prediction_days:
                st.error(f"Insufficient market data. Need at least {self.sequence_length + self.prediction_days} points")
                return None, None
                    
            # Create copies of input data
            market = market_data.copy()
            news = news_data.copy()
            
            # Ensure datetime format
            market['Date'] = pd.to_datetime(market['Date'])
            news['Date'] = pd.to_datetime(news['Date'])
            
            # Create date-only columns for merging
            market['DateOnly'] = market['Date'].dt.date
            news['DateOnly'] = news['Date'].dt.date
            
            # Find sentiment column in news data
            sentiment_col = None
            for col in news.columns:
                if 'sentiment' in col.lower() and 'score' in col.lower():
                    sentiment_col = col
                    break
                    
            if sentiment_col is None:
                st.error("Could not find sentiment score column in news data")
                return None, None
            
            # Calculate daily sentiment and explicitly name it 'News_Sentiment'
            daily_sentiment = news.groupby('DateOnly')[sentiment_col].mean().reset_index()
            daily_sentiment.columns = ['DateOnly', 'News_Sentiment']
            
            # Merge data
            data = pd.merge(market, daily_sentiment, on='DateOnly', how='left')
            
            # Verify News_Sentiment exists
            if 'News_Sentiment' not in data.columns:
                st.error("News_Sentiment column not found after merge")
                # Fix it by creating it if needed
                data['News_Sentiment'] = 0
            
            # Fill missing sentiment values
            data['News_Sentiment'] = data['News_Sentiment'].fillna(0)
            
            # Verify minimum required data points
            if len(data) < self.sequence_length + self.prediction_days:
                st.error(f"Insufficient data after merge. Need at least {self.sequence_length + self.prediction_days} points")
                return None, None
            
            # Prepare features - ensure they exist
            market_features = []
            
            # Basic required features
            required_features = ['Open', 'High', 'Low', 'Close', 'Volume']
            for feature in required_features:
                if feature in data.columns:
                    market_features.append(feature)
                else:
                    st.error(f"Required feature {feature} not found in data")
                    return None, None
                    
            # Add News_Sentiment
            features = market_features + ['News_Sentiment']
            self.used_features = features
            
            # Create sequences
            X = []
            y = []
            
            valid_range = len(data) - self.prediction_days
            
            for i in range(valid_range - self.sequence_length + 1):
                try:
                    sequence = []
                    for j in range(self.sequence_length):
                        # Extract values from the row for each feature
                        feature_values = []
                        for feature in features:
                            feature_values.append(data.iloc[i + j][feature])
                        sequence.extend(feature_values)
                    
                    target = data.iloc[i + self.sequence_length:i + self.sequence_length + self.prediction_days]['Close'].values
                    
                    if len(target) == self.prediction_days:
                        X.append(sequence)
                        y.append(target)
                except Exception:
                    continue
            
            if not X:
                st.error("No valid sequences could be created")
                return None, None
            
            X = np.array(X)
            y = np.array(y)
            
            # Normalize features
            X = self.scaler.fit_transform(X)
            
            st.success(f"Successfully prepared {len(X)} training samples")
            return X, y
            
        except Exception as e:
            st.error(f"Error in prepare_sentiment_enhanced_data: {str(e)}")
            return None, None

    def train_market_model(self, market_data):
        try:
            X, y = self.prepare_market_data(market_data)
            
            if X is None or y is None:
                return False
            
            self.market_model = RandomForestRegressor(
                n_estimators=50,
                max_depth=3,
                min_samples_split=2,
                random_state=42
            )
            
            self.market_model.fit(X, y)
            st.success("Market model trained successfully!")
            return True
            
        except Exception as e:
            st.error(f"Error training market model: {str(e)}")
            return False

    def train_sentiment_model(self, market_data, news_data):
        """Train sentiment-enhanced model with a direct approach to column handling"""
        try:
            # Use date helper to prepare dates
            market_copy, news_copy = prepare_dates_for_merge([market_data, news_data])
            
            # Find sentiment column in news data
            sentiment_col = None
            for col in news_copy.columns:
                if 'sentiment' in col.lower() and 'score' in col.lower():
                    sentiment_col = col
                    break
            
            # Handle case where sentiment column is not found
            if sentiment_col is None:
                sentiment_col = 'Sentiment Score'
                news_copy[sentiment_col] = 0
                        
            # Ensure datetime format
            market_copy['Date'] = pd.to_datetime(market_copy['Date'])
            news_copy['Date'] = pd.to_datetime(news_copy['Date'])
            
            # Create date-only columns for merging
            market_copy['DateOnly'] = market_copy['Date'].dt.date
            news_copy['DateOnly'] = news_copy['Date'].dt.date
            
            # DIRECT APPROACH: Skip the merge entirely
            # First, create a complete set of dates from the market data
            all_dates = pd.DataFrame({'DateOnly': market_copy['DateOnly'].unique()})
            
            # Second, calculate daily sentiment and join with all dates
            if len(news_copy) > 0:
                daily_news = news_copy.groupby('DateOnly')[sentiment_col].mean().reset_index()
                daily_news.columns = ['DateOnly', 'News_Sentiment']
                
                # Join with complete date set to fill in any missing dates
                sentiment_for_all_dates = pd.merge(
                    all_dates, 
                    daily_news, 
                    on='DateOnly', 
                    how='left'
                )
            else:
                # If no news data, create empty sentiment column
                sentiment_for_all_dates = all_dates.copy()
                sentiment_for_all_dates['News_Sentiment'] = 0
                
            # Fill missing sentiment values with 0
            sentiment_for_all_dates['News_Sentiment'] = sentiment_for_all_dates['News_Sentiment'].fillna(0)
                
            # Now join sentiment data to the market data
            # This guarantees the News_Sentiment column will exist
            combined = pd.merge(
                market_copy,
                sentiment_for_all_dates,
                on='DateOnly',
                how='left'
            )
            
            # Final safety check
            if 'News_Sentiment' not in combined.columns:
                # This should never happen with our approach, but just in case
                combined['News_Sentiment'] = 0
            
            # Prepare features - use core features and News_Sentiment
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 'News_Sentiment']
            self.used_features = features
            
            # Create sequences
            X = []
            y = []
            
            # Set sequence length and prediction days
            if not hasattr(self, 'sequence_length') or self.sequence_length < 1:
                self.sequence_length = 3
            if not hasattr(self, 'prediction_days') or self.prediction_days < 1:
                self.prediction_days = 2
                
            valid_range = len(combined) - self.prediction_days
            
            for i in range(valid_range - self.sequence_length + 1):
                try:
                    sequence = []
                    for j in range(self.sequence_length):
                        row_features = []
                        for feature in features:
                            if feature in combined.columns:
                                row_features.append(combined.iloc[i + j][feature])
                            else:
                                # This should never happen now, but just in case
                                row_features.append(0)
                        sequence.extend(row_features)
                    
                    target = combined.iloc[i + self.sequence_length:i + self.sequence_length + self.prediction_days]['Close'].values
                    
                    if len(target) == self.prediction_days:
                        X.append(sequence)
                        y.append(target)
                except Exception:
                    continue
            
            if not X:
                st.error("No valid sequences could be created")
                return False
            
            X = np.array(X)
            y = np.array(y)
            
            # Normalize features
            X = self.scaler.fit_transform(X)
            
            # Train model
            self.sentiment_enhanced_model = RandomForestRegressor(
                n_estimators=50,
                max_depth=5,
                min_samples_split=2,
                random_state=42
            )
            
            self.sentiment_enhanced_model.fit(X, y)
            st.success("Sentiment model trained successfully!")
            return True
            
        except Exception as e:
            st.error(f"Error training sentiment model: {str(e)}")
            return False

    def predict_market(self, market_data):
        try:
            if self.market_model is None:
                raise ValueError("Market model has not been trained")
            
            data = market_data.copy()
            
            # Check data length
            if len(data) < self.sequence_length:
                raise ValueError(f"Not enough data points. Need at least {self.sequence_length}")
            
            # Verify features
            missing_features = [f for f in self.market_features if f not in data.columns]
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            # Create sequence from last available data points
            sequence = []
            start_idx = len(data) - self.sequence_length
            
            for i in range(self.sequence_length):
                row = data.iloc[start_idx + i][self.market_features].values
                sequence.extend(row)
            
            features = np.array(sequence).reshape(1, -1)
            features = self.scaler.transform(features)
            
            return self.market_model.predict(features)[0]
            
        except Exception as e:
            st.error(f"Error in predict_market: {str(e)}")
            return None

    def predict_with_sentiment(self, market_data, news_data):
        """Predict with sentiment model"""
        try:
            if self.sentiment_enhanced_model is None:
                raise ValueError("Sentiment-enhanced model has not been trained")
            
            # Create copies to avoid modifying originals
            market_copy = market_data.copy()
            news_copy = news_data.copy()
            
            # Find sentiment column in news data
            sentiment_col = None
            for col in news_copy.columns:
                if 'sentiment' in col.lower() and 'score' in col.lower():
                    sentiment_col = col
                    break
            
            # Handle case where sentiment column is not found
            if sentiment_col is None:
                sentiment_col = 'Sentiment Score'
                news_copy[sentiment_col] = 0
            
            # Ensure datetime format
            market_copy['Date'] = pd.to_datetime(market_copy['Date'])
            news_copy['Date'] = pd.to_datetime(news_copy['Date'])
            
            # Create date-only columns for merging
            market_copy['DateOnly'] = market_copy['Date'].dt.date
            news_copy['DateOnly'] = news_copy['Date'].dt.date
            
            # Calculate daily sentiment with explicit naming
            daily_sentiment = news_copy.groupby('DateOnly')[sentiment_col].mean().reset_index()
            daily_sentiment.rename(columns={sentiment_col: 'News_Sentiment'}, inplace=True)
            
            # Merge data
            combined = pd.merge(market_copy, daily_sentiment, on='DateOnly', how='left')
            
            # Check if News_Sentiment exists after merge
            if 'News_Sentiment' not in combined.columns:
                combined['News_Sentiment'] = 0
            else:
                # Fill missing values
                combined['News_Sentiment'] = combined['News_Sentiment'].fillna(0)
            
            # Get features - use the same as in training
            features = self.used_features
            if not features:
                features = ['Open', 'High', 'Low', 'Close', 'Volume', 'News_Sentiment']
            
            # Create sequence
            sequence = []
            start_idx = len(combined) - self.sequence_length
            
            for i in range(self.sequence_length):
                row_features = []
                for feature in features:
                    if feature in combined.columns:
                        row_features.append(combined.iloc[start_idx + i][feature])
                    else:
                        # Use 0 as default if feature is missing
                        row_features.append(0)
                sequence.extend(row_features)
            
            features = np.array(sequence).reshape(1, -1)
            features = self.scaler.transform(features)
            
            return self.sentiment_enhanced_model.predict(features)[0]
            
        except Exception as e:
            st.error(f"Error in predict_with_sentiment: {str(e)}")
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
            
            # Create results dictionary
            results = {
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
            
            return results
            
        except Exception as e:
            st.error(f"Error evaluating predictions: {str(e)}")
            return None
    
class MultiModelPredictionAnalyser(MarketPredictionAnalyser):
    """Extension of MarketPredictionAnalyser that supports multiple sentiment models"""
    
    def __init__(self):
        super().__init__()
        self.model_sentiment_models = {}  # Dictionary to store different models
        self.model_predictions = {}  # Dictionary to store predictions from each model
    
    def prepare_model_specific_data(self, market_data, news_data, model_name):
        """Prepare data for a specific sentiment model"""
        try:
            # Create copies to avoid modifying originals
            market_copy = market_data.copy()
            news_copy = news_data.copy()
            
            # Find model-specific sentiment column
            sentiment_col = f'{model_name}_score'
            if sentiment_col not in news_copy.columns:
                st.error(f"Could not find {sentiment_col} column in news data")
                return None
            
            # Ensure datetime format
            market_copy['Date'] = pd.to_datetime(market_copy['Date'])
            news_copy['Date'] = pd.to_datetime(news_copy['Date'])
            
            # Create date-only columns for merging
            market_copy['DateOnly'] = market_copy['Date'].dt.date
            news_copy['DateOnly'] = news_copy['Date'].dt.date
            
            # Calculate daily sentiment
            daily_sentiment = news_copy.groupby('DateOnly')[sentiment_col].mean().reset_index()
            daily_sentiment.columns = ['DateOnly', f'{model_name.capitalize()}_Sentiment']
            
            # Merge data
            combined = pd.merge(market_copy, daily_sentiment, on='DateOnly', how='left')
            
            # Fill missing sentiment values
            combined[f'{model_name.capitalize()}_Sentiment'] = combined[f'{model_name.capitalize()}_Sentiment'].fillna(0)
            
            return combined
            
        except Exception as e:
            st.error(f"Error preparing model-specific data for {model_name}: {str(e)}")
            return None
    
    def train_model_specific_models(self, market_data, news_data, available_models):
        """Train a separate model for each sentiment model"""
        for model_name in available_models:
            try:
                st.write(f"Training model for {model_name.capitalize()} sentiment...")
                
                # Prepare data for this model
                combined_data = self.prepare_model_specific_data(market_data, news_data, model_name)
                
                if combined_data is None:
                    continue
                
                # Create a new prediction model for this sentiment model
                self.model_sentiment_models[model_name] = RandomForestRegressor(
                    n_estimators=50,
                    max_depth=5,
                    min_samples_split=2,
                    random_state=42
                )
                
                # Define features
                features = ['Open', 'High', 'Low', 'Close', 'Volume', f'{model_name.capitalize()}_Sentiment']
                
                # Create sequences
                X = []
                y = []
                
                valid_range = len(combined_data) - self.prediction_days
                
                for i in range(valid_range - self.sequence_length + 1):
                    # Check bounds safely without using try/except
                    if i + self.sequence_length + self.prediction_days > len(combined_data):
                        continue
                        
                    sequence = []
                    for j in range(self.sequence_length):
                        row_features = []
                        for feature in features:
                            if feature in combined_data.columns and i+j < len(combined_data):
                                row_features.append(combined_data.iloc[i + j][feature])
                            else:
                                row_features.append(0)
                        sequence.extend(row_features)
                    
                    target = combined_data.iloc[i + self.sequence_length:i + self.sequence_length + self.prediction_days]['Close'].values
                    
                    if len(target) == self.prediction_days:
                        X.append(sequence)
                        y.append(target)
                
                if not X:
                    st.warning(f"No valid sequences for {model_name}")
                    continue
                
                X = np.array(X)
                y = np.array(y)
                
                # Normalize features
                X = self.scaler.fit_transform(X)
                
                # Train model
                self.model_sentiment_models[model_name].fit(X, y)
                st.success(f"✅ {model_name.capitalize()} model trained successfully!")
                
            except Exception as e:
                st.error(f"Error training {model_name} model: {str(e)}")
    
    def predict_with_model_specific_sentiment(self, market_data, news_data, model_name):
        """Predict using a specific sentiment model"""
        try:
            if model_name not in self.model_sentiment_models:
                raise ValueError(f"{model_name.capitalize()} model has not been trained")
            
            # Prepare data for this model
            combined_data = self.prepare_model_specific_data(market_data, news_data, model_name)
            
            if combined_data is None:
                raise ValueError(f"Could not prepare data for {model_name}")
            
            # Define features
            features = ['Open', 'High', 'Low', 'Close', 'Volume', f'{model_name.capitalize()}_Sentiment']
            
            # Create sequence
            sequence = []
            start_idx = len(combined_data) - self.sequence_length
            
            for i in range(self.sequence_length):
                row_features = []
                for feature in features:
                    if feature in combined_data.columns:
                        row_features.append(combined_data.iloc[start_idx + i][feature])
                    else:
                        row_features.append(0)
                sequence.extend(row_features)
            
            features = np.array(sequence).reshape(1, -1)
            features = self.scaler.transform(features)
            
            return self.model_sentiment_models[model_name].predict(features)[0]
            
        except Exception as e:
            st.error(f"Error in prediction with {model_name} model: {str(e)}")
            return None
    
    def predict_all_models(self, market_data, news_data, available_models):
        """Generate predictions using all available models"""
        self.model_predictions = {}
        
        # First, get the market-only prediction
        market_predictions = self.predict_market(market_data)
        if market_predictions is not None:
            self.model_predictions['market'] = market_predictions
        
        # Next, get the ensemble sentiment prediction
        ensemble_predictions = self.predict_with_sentiment(market_data, news_data)
        if ensemble_predictions is not None:
            self.model_predictions['ensemble'] = ensemble_predictions
        
        # Finally, get predictions for each individual model
        for model_name in available_models:
            predictions = self.predict_with_model_specific_sentiment(market_data, news_data, model_name)
            if predictions is not None:
                self.model_predictions[model_name] = predictions
        
        return self.model_predictions    