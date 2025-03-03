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
            
            # Convert slope to percentage and apply threshold for stable trend
            slope_percentage = float(slope * 100)  # Convert to percentage
            
            # Use a threshold to determine if trend is significant enough
            TREND_THRESHOLD = 0.1  # Minimum slope to consider a trend significant (0.1%)
            
            if abs(slope_percentage) < TREND_THRESHOLD:
                trend_direction = 'Stable'
            else:
                trend_direction = 'Improving' if slope > 0 else 'Declining'
            
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
            
            # Calculate returns from values (needed for proper financial metrics)
            # For sentiment data or non-price data, we use differences
            value_returns = np.diff(values)
            if len(value_returns) > 0:
                # Add a leading zero to maintain array length
                value_returns = np.insert(value_returns, 0, 0)
            else:
                value_returns = np.zeros_like(values)

            # Calculate Sharpe Ratio properly
            risk_free_rate = 0
            avg_return = np.mean(value_returns)
            return_volatility = np.std(value_returns)
            if return_volatility > 0 and not np.isnan(return_volatility):
                sharpe_ratio = float((avg_return - risk_free_rate) / return_volatility)
                # Scale for more intuitive values
                sharpe_ratio = float(f"{sharpe_ratio:.3f}")
            else:
                # Handle edge case with safe default
                sharpe_ratio = float(0.001 if avg_return > 0 else -0.001) if avg_return != 0 else 0.0

            # Determine if this is price data or sentiment data based on value characteristics
            is_price_data = np.mean(values) > 1.0  # Price data typically has values > 1
            
            # Different drawdown calculation approaches based on data type
            if is_price_data:  # Likely price data
                # Calculate Maximum Drawdown with cumulative approach for price data
                cumulative_values = np.cumsum(value_returns) + values[0]
                max_drawdown = 0.0

                # Track running peak
                peak = cumulative_values[0]
                for value in cumulative_values[1:]:
                    # Update peak if new high
                    if value > peak:
                        peak = value
                    # Calculate drawdown if below peak
                    elif peak > 0:  # Avoid division by zero
                        drawdown = (peak - value) / peak
                        max_drawdown = max(max_drawdown, drawdown)
                        
                # Convert to percentage
                max_drawdown = float(f"{max_drawdown * 100:.2f}")
            else:
                # IMPROVED SENTIMENT DRAWDOWN CALCULATION
                # Use normalized z-score based approach for sentiment data
                if len(values) > 1:
                    # Find the high point (peak) and low point (trough)
                    peak_value = np.max(values)
                    trough_value = np.min(values)
                    
                    # Calculate sentiment range
                    sentiment_range = peak_value - trough_value
                    
                    if sentiment_range > 0:
                        # For sentiment data, measure max reversal from positive to negative
                        # Then scale to a reasonable percentage range (0-40%)
                        pos_to_neg_drops = []
                        for i in range(1, len(values)):
                            if values[i-1] > 0 and values[i] < values[i-1]:
                                # Calculate drop as percentage of the total range
                                drop = (values[i-1] - values[i]) / sentiment_range
                                pos_to_neg_drops.append(drop)
                        
                        # If we found drops, use the maximum one
                        if pos_to_neg_drops:
                            max_drop = max(pos_to_neg_drops)
                            # Scale to reasonable range (0-30%)
                            max_drawdown = min(max_drop * 30, 30.0)
                        else:
                            # If no significant drops, use a smaller value
                            max_drawdown = sentiment_range * 5.0
                            
                        # Ensure reasonable bounds (5-30%)
                        max_drawdown = max(min(max_drawdown, 30.0), 5.0)
                    else:
                        # Default for little variation
                        max_drawdown = 5.0
                else:
                    max_drawdown = 0.0
                    
                # Format to 2 decimal places
                max_drawdown = float(f"{max_drawdown:.2f}")
            
            # Create trend result dictionary
            trend_analysis = {
                'trend_direction': trend_direction,
                'trend_strength': float(abs(slope_percentage)),  # Store as absolute percentage
                'trend_confidence': float(f"{trend_confidence:.1f}"),
                'volatility': float(f"{volatility:.3f}"),
                'momentum': float(f"{momentum:.3f}"),
                'moving_average_5': float(f"{ma5:.2f}"),
                'moving_average_20': float(f"{ma20:.2f}"),
                'has_outliers': outliers > 0,
                'outlier_count': int(outliers),
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'is_price_data': is_price_data  # Include this flag for reference
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
