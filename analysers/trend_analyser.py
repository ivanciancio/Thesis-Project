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
