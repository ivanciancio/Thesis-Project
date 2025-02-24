import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import pandas as pd
import streamlit as st

class SentimentTrendAnalyser:
    def analyse_trends(self, data_series):
        """Analyse sentiment trends over time with improved confidence calculation"""
        try:
            if data_series is None or len(data_series) < 2:
                return None

            # Convert to numpy array and remove any NaN values
            values = pd.Series(data_series).dropna().values
            dates_numeric = np.array(range(len(values))).reshape(-1, 1)
            
            # Fit linear regression for trend
            model = LinearRegression()
            model.fit(dates_numeric, values)
            
            # Calculate trend metrics
            slope = model.coef_[0]
            trend_direction = 'Improving' if slope > 0 else 'Declining' if slope < 0 else 'Stable'
            
            # Calculate R-squared (trend confidence)
            r_squared = model.score(dates_numeric, values)
            trend_confidence = r_squared * 100  # Convert to percentage
            
            # Calculate volatility (standard deviation)
            volatility = float(np.std(values))
            
            # Calculate moving averages
            series = pd.Series(values)
            ma5 = float(series.rolling(window=min(5, len(values))).mean().iloc[-1])
            ma20 = float(series.rolling(window=min(20, len(values))).mean().iloc[-1])
            
            # Calculate momentum
            momentum = float(np.mean(np.diff(values))) if len(values) > 1 else 0.0
            
            # Detect outliers using z-score
            z_scores = np.abs((values - np.mean(values)) / np.std(values))
            outliers = sum(z_scores > 2)
            
            # Create trend result dictionary
            trend_analysis = {
                'trend_direction': trend_direction,
                'trend_strength': float(abs(slope)),
                'trend_confidence': float(f"{trend_confidence:.1f}"),
                'volatility': float(f"{volatility:.3f}"),
                'momentum': float(f"{momentum:.3f}"),
                'moving_average_5': float(f"{ma5:.2f}"),
                'moving_average_20': float(f"{ma20:.2f}"),
                'has_outliers': outliers > 0,
                'outlier_count': int(outliers)
            }
            
            return trend_analysis
            
        except Exception as e:
            st.error(f"Error in trend analysis: {str(e)}")
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
            window5 = min(5, len(values))
            window20 = min(20, len(values))
            ma5 = series.rolling(window=window5).mean().iloc[-1]
            ma20 = series.rolling(window=window20).mean().iloc[-1]
            
            return {
                'volatility': volatility,
                'outlier_count': sum(outliers),
                'has_outliers': any(outliers),
                'moving_average_5': ma5,
                'moving_average_20': ma20
            }
        except Exception as e:
            st.error(f"Error calculating volatility metrics: {str(e)}")
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
        
        # Moving averages comparison
        if trend_analysis['moving_average_5'] is not None and trend_analysis['moving_average_20'] is not None:
            ma_diff = trend_analysis['moving_average_5'] - trend_analysis['moving_average_20']
            if abs(ma_diff) > 0.1:
                ma_desc = "diverging from" if ma_diff > 0 else "converging with"
                summary_parts.append(f"Short-term trend is {ma_desc} long-term trend")
        
        # Outliers
        if trend_analysis['has_outliers']:
            summary_parts.append(f"Found {trend_analysis['outlier_count']} significant outliers")
        
        return " | ".join(summary_parts)

    def get_trend_recommendations(self, market_trend, sentiment_trends):
        """Generate recommendations based on trend analysis"""
        try:
            if market_trend is None:
                return ["No market trend data available"]

            recommendations = []

            # Market trend recommendations
            if market_trend.get('trend_confidence', 0) > 40:
                if market_trend['trend_direction'] == 'Improving':
                    recommendations.append("Strong positive market trend detected")
                elif market_trend['trend_direction'] == 'Declining':
                    recommendations.append("Strong negative market trend detected")

            # Process each sentiment source
            for source, trend in sentiment_trends.items():
                if trend is None:
                    continue
                    
                # Trend strength recommendations
                if trend.get('trend_confidence', 0) > 40:
                    if trend['trend_direction'] == 'Improving':
                        recommendations.append(f"{source.title()} sentiment shows strong improvement")
                    elif trend['trend_direction'] == 'Declining':
                        recommendations.append(f"{source.title()} sentiment shows significant decline")

                # Trend alignment
                if trend['trend_direction'] == market_trend['trend_direction']:
                    recommendations.append(f"{source.title()} sentiment aligns with market trend")
                else:
                    recommendations.append(f"{source.title()} sentiment diverges from market trend")

                # Volatility checks
                if trend.get('volatility', 0) > 0.5:
                    recommendations.append(f"High {source.lower()} sentiment volatility detected")

            if not recommendations:
                recommendations.append("No significant trends detected")

            return recommendations

        except Exception as e:
            st.error(f"Error generating trend recommendations: {str(e)}")
            return ["Unable to generate recommendations due to an error"]
        
def display_trend_analysis(trend_data):
    """Display trend analysis results"""
    if trend_data is None:
        st.warning("No trend analysis results available")
        return

    try:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Trend Direction",
                trend_data['trend_direction'],
                delta=f"{trend_data['trend_strength']:.3f}"
            )
        
        with col2:
            st.metric(
                "Trend Confidence",
                f"{trend_data['trend_confidence']:.1f}%"
            )
        
        with col3:
            st.metric(
                "Volatility",
                f"{trend_data['volatility']:.3f}"
            )
        
        # Moving Averages
        st.subheader("Moving Averages")
        st.write(f"Short-term MA (5): {trend_data['moving_average_5']:.2f}")
        st.write(f"Long-term MA (20): {trend_data['moving_average_20']:.2f}")
        
        # Outlier warning if needed
        if trend_data['has_outliers']:
            st.warning(f"Found {trend_data['outlier_count']} significant outliers in the data")
            
    except Exception as e:
        st.error(f"Error displaying trend analysis: {str(e)}")