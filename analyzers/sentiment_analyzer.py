import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob

class FinancialSentimentAnalyzer:
    def __init__(self):
        """Initialize the sentiment analyzer"""
        pass

    def analyze_sentiment(self, text):
        """
        Analyze sentiment using TextBlob as a simpler alternative to FinBERT
        """
        try:
            if not text or not isinstance(text, str):
                return {
                    'sentiment': 'Neutral',
                    'score': 0,
                    'confidence': 0.5
                }

            # Use TextBlob for sentiment analysis
            analysis = TextBlob(text)
            
            # Get polarity score (-1 to 1)
            polarity = analysis.sentiment.polarity
            
            # Get subjectivity score (0 to 1) as confidence
            confidence = analysis.sentiment.subjectivity
            
            # Map polarity to sentiment labels
            if polarity > 0.1:
                sentiment = 'Positive'
            elif polarity < -0.1:
                sentiment = 'Negative'
            else:
                sentiment = 'Neutral'

            return {
                'sentiment': sentiment,
                'score': polarity,
                'confidence': confidence
            }

        except Exception as e:
            st.error(f"Error in sentiment analysis: {str(e)}")
            return {
                'sentiment': 'Neutral',
                'score': 0,
                'confidence': 0.5
            }

    def analyze_text_batch(self, texts):
        """
        Analyze a batch of texts
        """
        results = []
        for text in texts:
            results.append(self.analyze_sentiment(text))
        return results

    def get_sentiment_summary(self, results):
        """
        Generate summary statistics from multiple sentiment analyses
        """
        if not results:
            return None

        scores = [r['score'] for r in results]
        sentiments = [r['sentiment'] for r in results]

        return {
            'average_score': np.mean(scores),
            'sentiment_distribution': {
                'Positive': sentiments.count('Positive'),
                'Negative': sentiments.count('Negative'),
                'Neutral': sentiments.count('Neutral')
            },
            'strongest_positive': max(scores),
            'strongest_negative': min(scores)
        }