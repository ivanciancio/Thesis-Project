import warnings
import os
import sys
from pathlib import Path

# Add the parent directory to Python path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))
    
# Add these lines at the very top of your file, before any other imports
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*torch.classes.*')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
from typing import Dict, List, Optional, Union
import logging

try:
    # Try different import paths
    try:
        from .finbert_analyser import FinBERTAnalyser  # Same directory
    except ImportError:
        try:
            from analysers.finbert_analyser import FinBERTAnalyser  # From package
        except ImportError:
            from finbert_analyser import FinBERTAnalyser  # Direct import
except ImportError as e:
    logging.error(f"Could not import FinBERTAnalyser: {str(e)}")
    FinBERTAnalyser = None

class FinancialSentimentAnalyser:
    def __init__(self):
        """Initialise the sentiment analyser with multiple models"""
        self.models = {'textblob': True}  # TextBlob is always available
        self.model_weights = {
            'textblob': 1.0,  # Default weight when only TextBlob is available
            'vader': 0.0,
            'finbert': 0.0
        }
        
        # Initialise VADER quietly
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.vader = SentimentIntensityAnalyzer()
            self.models['vader'] = True
            self.model_weights.update({
                'textblob': 0.3,
                'vader': 0.7,
                'finbert': 0.0
            })
        except ImportError:
            self.vader = None
            self.models['vader'] = False
        
        # Initialise FinBERT
        try:
            if FinBERTAnalyser is not None:
                self.finbert = FinBERTAnalyser()
                self.models['finbert'] = True
            else:
                self.finbert = None
                self.models['finbert'] = False
                logging.warning("FinBERT is not available - continuing without it")
            # Update weights if FinBERT is available
            if self.vader:
                self.model_weights.update({
                    'textblob': 0.2,
                    'vader': 0.3,
                    'finbert': 0.5
                })
            else:
                self.model_weights.update({
                    'textblob': 0.3,
                    'vader': 0.0,
                    'finbert': 0.7
                })
        except Exception as e:
            self.finbert = None
            self.models['finbert'] = False
            logging.error(f"Error initialising FinBERT: {str(e)}")
        
        # Normalise weights based on available models
        self._normalise_weights()
        
        # Setup logging with less verbosity
        logging.basicConfig(level=logging.ERROR)
        self.logger = logging.getLogger(__name__)

    def _normalise_weights(self):
        """Normalise model weights based on available models"""
        available_models = [model for model, available in self.models.items() if available]
        total_weight = sum(self.model_weights[model] for model in available_models)
        
        if total_weight == 0:
            # If no weights or models available, set equal weights
            weight = 1.0 / len(available_models)
            for model in available_models:
                self.model_weights[model] = weight
        else:
            # Normalise existing weights
            for model in self.model_weights:
                if not self.models.get(model, False):
                    self.model_weights[model] = 0
                else:
                    self.model_weights[model] /= total_weight

    def analyse_with_textblob(self, text: str) -> Dict:
        """
        Analyse sentiment using TextBlob
        
        Args:
            text (str): Text to analyse
            
        Returns:
            Dict: Sentiment analysis results
        """
        try:
            analysis = TextBlob(text)
            polarity = analysis.sentiment.polarity
            subjectivity = analysis.sentiment.subjectivity
            
            return {
                'score': polarity,
                'confidence': subjectivity,
                'model': 'textblob'
            }
        except Exception as e:
            self.logger.error(f"TextBlob analysis error: {str(e)}")
            return {
                'score': 0,
                'confidence': 0,
                'model': 'textblob'
            }

    def analyse_with_vader(self, text: str) -> Optional[Dict]:
        """
        Analyse sentiment using VADER if available
        
        Args:
            text (str): Text to analyse
            
        Returns:
            Optional[Dict]: Sentiment analysis results or None if VADER unavailable
        """
        if not self.vader:
            return None
            
        try:
            scores = self.vader.polarity_scores(text)
            return {
                'score': scores['compound'],
                'confidence': max(scores['pos'], scores['neg'], scores['neu']),
                'model': 'vader',
                'detail_scores': {
                    'positive': scores['pos'],
                    'negative': scores['neg'],
                    'neutral': scores['neu']
                }
            }
        except Exception as e:
            self.logger.error(f"VADER analysis error: {str(e)}")
            return None

    def analyse_with_finbert(self, text: str) -> Optional[Dict]:
        """
        Analyse sentiment using FinBERT if available
        
        Args:
            text (str): Text to analyse
            
        Returns:
            Optional[Dict]: Sentiment analysis results or None if FinBERT unavailable
        """
        if not self.finbert:
            return None
            
        try:
            result = self.finbert.analyse_text(text)
            
            return {
                'score': result['score'],
                'confidence': result['confidence'],
                'model': 'finbert',
                'detail_scores': result['probabilities']
            }
        except Exception as e:
            self.logger.error(f"FinBERT analysis error: {str(e)}")
            return None

    def analyse_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment using ensemble of available models
        
        Args:
            text (str): Text to analyse
            
        Returns:
            Dict: Combined sentiment analysis results
        """
        try:
            if not text or not isinstance(text, str):
                return self._get_neutral_result()

            # Get results from all available models
            model_results = {}
            
            # TextBlob analysis
            textblob_result = self.analyse_with_textblob(text)
            if textblob_result:
                model_results['textblob'] = textblob_result
            
            # VADER analysis
            if self.models.get('vader'):
                vader_result = self.analyse_with_vader(text)
                if vader_result:
                    model_results['vader'] = vader_result
            
            # FinBERT analysis
            if self.models.get('finbert'):
                finbert_result = self.analyse_with_finbert(text)
                if finbert_result:
                    model_results['finbert'] = finbert_result
            
            # Calculate weighted ensemble score
            ensemble_score = 0
            ensemble_confidence = 0
            total_weight = 0
            
            for model, result in model_results.items():
                weight = self.model_weights[model]
                ensemble_score += result['score'] * weight
                ensemble_confidence += result['confidence'] * weight
                total_weight += weight
            
            if total_weight > 0:
                ensemble_score /= total_weight
                ensemble_confidence /= total_weight
            
            # Calculate polarisation metrics
            polarisation_metrics = self._calculate_polarisation(model_results)
            
            # Determine sentiment label
            sentiment = self._get_sentiment_label(ensemble_score)
            
            return {
                'sentiment': sentiment,
                'score': ensemble_score,
                'confidence': ensemble_confidence,
                'model_scores': {
                    model: result['score'] 
                    for model, result in model_results.items()
                },
                'model_confidences': {
                    model: result['confidence'] 
                    for model, result in model_results.items()
                },
                'polarisation_metrics': polarisation_metrics
            }

        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {str(e)}")
            return self._get_neutral_result()

    def _calculate_polarisation(self, model_results: Dict) -> Dict:
        """
        Calculate sentiment polarisation metrics across models
        
        Args:
            model_results (Dict): Results from different models
            
        Returns:
            Dict: Polarisation metrics
        """
        if not model_results:
            return {
                'std': 0,
                'range': 0,
                'disagreement': 0
            }
        
        scores = [result['score'] for result in model_results.values()]
        
        # Calculate polarisation metrics
        score_std = np.std(scores) if len(scores) > 1 else 0
        score_range = max(scores) - min(scores) if scores else 0
        
        # Calculate model disagreement (how often models disagree on sentiment direction)
        signs = [1 if s > 0 else -1 if s < 0 else 0 for s in scores]
        disagreement = len(set(signs)) / len(signs) if scores else 0
        
        return {
            'std': score_std,
            'range': score_range,
            'disagreement': disagreement
        }

    def _get_sentiment_label(self, score: float) -> str:
        """
        Map sentiment score to label with more granular categories
        
        Args:
            score (float): Sentiment score
            
        Returns:
            str: Sentiment label
        """
        if score > 0.5:
            return 'Very Positive'
        elif score > 0.1:
            return 'Positive'
        elif score < -0.5:
            return 'Very Negative'
        elif score < -0.1:
            return 'Negative'
        else:
            return 'Neutral'

    def _get_neutral_result(self) -> Dict:
        """
        Get neutral result for invalid inputs
        
        Returns:
            Dict: Neutral sentiment result
        """
        return {
            'sentiment': 'Neutral',
            'score': 0,
            'confidence': 0.5,
            'model_scores': {},
            'model_confidences': {},
            'polarisation_metrics': {
                'std': 0,
                'range': 0,
                'disagreement': 0
            }
        }

    def analyse_text_batch(self, texts: List[str]) -> List[Dict]:
        """
        Analyse sentiment for a batch of texts
        
        Args:
            texts (List[str]): List of texts to analyse
            
        Returns:
            List[Dict]: List of sentiment analysis results
        """
        return [self.analyse_sentiment(text) for text in texts]

    def get_sentiment_summary(self, results: List[Dict]) -> Optional[Dict]:
        """
        Generate summary statistics from multiple sentiment analyses
        
        Args:
            results (List[Dict]): List of sentiment analysis results
            
        Returns:
            Optional[Dict]: Summary statistics or None if no results
        """
        if not results:
            return None

        scores = [r['score'] for r in results]
        sentiments = [r['sentiment'] for r in results]
        confidences = [r['confidence'] for r in results]
        polarisations = [r['polarisation_metrics']['std'] for r in results]

        return {
            'average_score': np.mean(scores),
            'score_std': np.std(scores),
            'average_confidence': np.mean(confidences),
            'sentiment_distribution': {
                label: sentiments.count(label)
                for label in set(sentiments)
            },
            'strongest_positive': max(scores),
            'strongest_negative': min(scores),
            'average_polarisation': np.mean(polarisations),
            'max_polarisation': max(polarisations)
        }

    def calculate_temporal_sentiment(self, texts: List[str], dates: List[Union[str, pd.Timestamp]]) -> pd.DataFrame:
        """
        Calculate sentiment over time for a series of texts
        
        Args:
            texts (List[str]): List of texts
            dates (List[Union[str, pd.Timestamp]]): Corresponding dates
            
        Returns:
            pd.DataFrame: Temporal sentiment analysis
        """
        results = self.analyse_text_batch(texts)
        
        # Create DataFrame with dates and sentiment scores
        df = pd.DataFrame({
            'date': pd.to_datetime(dates),
            'sentiment_score': [r['score'] for r in results],
            'confidence': [r['confidence'] for r in results],
            'polarisation': [r['polarisation_metrics']['std'] for r in results]
        })
        
        # Calculate daily aggregates
        daily_sentiment = df.groupby(df['date'].dt.date).agg({
            'sentiment_score': ['mean', 'std'],
            'confidence': 'mean',
            'polarisation': 'mean'
        }).reset_index()
        
        return daily_sentiment