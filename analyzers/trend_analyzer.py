import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import pandas as pd
import streamlit as st

class SentimentTrendAnalyzer:
    def analyze_trends(self, data_series):
        """
        Analyze sentiment trends over time
        
        Args:
            data_series (pd.Series): Time series data to analyze
        """
        try:
            if data_series is None or len(data_series) < 2:
                return None

            # Convert to numpy array for analysis
            values = np.array(data_series)
            dates_numeric = np.array(range(len(values))).reshape(-1, 1)
            
            # Calculate basic statistics
            basic_stats = self.calculate_basic_stats(values)
            
            # Calculate trend metrics
            trend_metrics = self.calculate_trend_metrics(dates_numeric, values)
            
            # Calculate volatility metrics
            volatility_metrics = self.calculate_volatility_metrics(values)
            
            # Combine all metrics
            return {
                **basic_stats,
                **trend_metrics,
                **volatility_metrics
            }
            
        except Exception as e:
            st.error(f"Error in trend analysis: {e}")
            return None

    def calculate_basic_stats(self, values):
        """Calculate basic statistical measures"""
        return {
            'mean': np.mean(values),
            'median': np.median(values),
            'min': np.min(values),
            'max': np.max(values),
            'range': np.ptp(values),
            'std': np.std(values)
        }

    def calculate_trend_metrics(self, dates_numeric, values):
        """Calculate trend-related metrics"""
        try:
            # Fit linear regression
            model = LinearRegression()
            model.fit(dates_numeric, values)
            
            # Calculate slope and related metrics
            slope = model.coef_[0]
            trend_direction = 'Improving' if slope > 0 else 'Declining' if slope < 0 else 'Stable'
            
            # Calculate R-squared
            r_squared = model.score(dates_numeric, values)
            
            # Calculate momentum (rate of change)
            momentum = np.diff(values)
            avg_momentum = np.mean(momentum) if len(momentum) > 0 else 0
            
            return {
                'trend_direction': trend_direction,
                'trend_strength': abs(slope),
                'trend_confidence': r_squared * 100,  # Convert to percentage
                'momentum': avg_momentum
            }
        except Exception as e:
            st.error(f"Error calculating trend metrics: {e}")
            return {
                'trend_direction': 'Unknown',
                'trend_strength': 0,
                'trend_confidence': 0,
                'momentum': 0
            }

    def calculate_volatility_metrics(self, values):
        """Calculate volatility-related metrics"""
        try:
            # Calculate standard deviation
            volatility = np.std(values)
            
            # Calculate z-scores for outlier detection
            z_scores = stats.zscore(values)
            outliers = np.abs(z_scores) > 2
            
            # Calculate moving averages
            series = pd.Series(values)
            ma5 = series.rolling(window=min(5, len(values))).mean().iloc[-1]
            ma20 = series.rolling(window=min(20, len(values))).mean().iloc[-1]
            
            return {
                'volatility': volatility,
                'outlier_count': sum(outliers),
                'has_outliers': any(outliers),
                'moving_average_5': ma5,
                'moving_average_20': ma20
            }
        except Exception as e:
            st.error(f"Error calculating volatility metrics: {e}")
            return {
                'volatility': 0,
                'outlier_count': 0,
                'has_outliers': False,
                'moving_average_5': None,
                'moving_average_20': None
            }

    def get_trend_summary(self, trend_analysis):
        """Generate a human-readable summary of the trend analysis"""
        if not trend_analysis:
            return "Insufficient data for trend analysis"

        summary_parts = []
        
        # Overall trend
        summary_parts.append(
            f"The trend is {trend_analysis['trend_direction'].lower()} "
            f"with {trend_analysis['trend_confidence']:.1f}% confidence"
        )
        
        # Volatility
        if trend_analysis['volatility'] > 0.5:
            volatility_desc = "highly volatile"
        elif trend_analysis['volatility'] > 0.2:
            volatility_desc = "moderately volatile"
        else:
            volatility_desc = "stable"
            
        summary_parts.append(f"The data is {volatility_desc}")
        
        # Momentum
        if abs(trend_analysis['momentum']) > 0.1:
            momentum_desc = "rapidly" if abs(trend_analysis['momentum']) > 0.2 else "gradually"
            direction = "improving" if trend_analysis['momentum'] > 0 else "declining"
            summary_parts.append(f"The trend is {momentum_desc} {direction}")
        
        # Moving averages comparison
        if (trend_analysis['moving_average_5'] is not None and 
            trend_analysis['moving_average_20'] is not None):
            ma_diff = trend_analysis['moving_average_5'] - trend_analysis['moving_average_20']
            if abs(ma_diff) > 0.1:
                ma_desc = "diverging from" if ma_diff > 0 else "converging with"
                summary_parts.append(f"Short-term trend is {ma_desc} long-term trend")
        
        # Outliers
        if trend_analysis['has_outliers']:
            summary_parts.append(f"Found {trend_analysis['outlier_count']} significant outliers")
        
        return " | ".join(summary_parts)

    def get_trend_recommendations(self, market_trend, sentiment_trend):
        """Generate recommendations based on trend analysis"""
        recommendations = []
        
        if not market_trend or not sentiment_trend:
            return ["Insufficient data for trend recommendations"]

        # Market trend recommendations
        if market_trend['trend_confidence'] > 70:
            if market_trend['trend_direction'] == 'Improving':
                recommendations.append("Strong positive market trend detected")
            elif market_trend['trend_direction'] == 'Declining':
                recommendations.append("Strong negative market trend detected")

        # Sentiment trend recommendations
        if sentiment_trend['trend_confidence'] > 70:
            if sentiment_trend['trend_direction'] == 'Improving':
                recommendations.append("Sentiment is showing strong improvement")
            elif sentiment_trend['trend_direction'] == 'Declining':
                recommendations.append("Sentiment is showing significant decline")

        # Trend alignment
        if market_trend['trend_direction'] == sentiment_trend['trend_direction']:
            recommendations.append("Market and sentiment trends are aligned")
        else:
            recommendations.append("Market and sentiment trends show divergence")

        # Volatility recommendations
        if market_trend['volatility'] > 0.5:
            recommendations.append("High market volatility detected")
        if sentiment_trend['volatility'] > 0.5:
            recommendations.append("High sentiment volatility detected")

        return recommendations