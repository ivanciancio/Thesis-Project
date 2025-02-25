from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from typing import Dict, List, Union
import logging

# Configure logging
logging.basicConfig(level=logging.ERROR)

class FinBERTAnalyser:
    """
    FinBERT sentiment analyser for financial text
    """
    def __init__(self, model_name: str = 'ProsusAI/finbert'):
        """
        Initialise FinBERT analyser
        
        Args:
            model_name (str): HuggingFace model name/path
        """
        try:
            self.tokeniser = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            # Label mapping
            self.labels = ['positive', 'neutral', 'negative']
        except Exception as e:
            logging.error(f"Error initialising FinBERT: {str(e)}")
            raise

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for FinBERT analysis
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Truncate if too long (FinBERT maximum length is 512 tokens)
        max_chars = 1024  # Approximate character limit
        if len(text) > max_chars:
            text = text[:max_chars]
        
        return text

    def analyse_text(self, text: str) -> Dict:
        """
        Analyse sentiment of a single text
        
        Args:
            text (str): Text to analyse
            
        Returns:
            Dict: Sentiment analysis results
        """
        try:
            # Preprocess text
            text = self.preprocess_text(text)
            
            # Tokenise
            inputs = self.tokeniser(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Get model outputs
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=1)
            
            # Convert to numpy for easier handling
            scores = predictions[0].cpu().numpy()
            
            # Calculate sentiment score (-1 to 1)
            # Positive - Negative, weighted by probabilities
            sentiment_score = float(scores[0] - scores[2])
            
            # Get highest probability label
            label_idx = np.argmax(scores)
            sentiment_label = self.labels[label_idx]
            
            # Calculate confidence
            confidence = float(scores[label_idx])
            
            return {
                'sentiment': sentiment_label,
                'score': sentiment_score,
                'confidence': confidence,
                'probabilities': {
                    'positive': float(scores[0]),
                    'neutral': float(scores[1]),
                    'negative': float(scores[2])
                }
            }
            
        except Exception as e:
            logging.error(f"Error in FinBERT analysis: {str(e)}")
            return {
                'sentiment': 'neutral',
                'score': 0.0,
                'confidence': 0.0,
                'probabilities': {
                    'positive': 0.0,
                    'neutral': 1.0,
                    'negative': 0.0
                }
            }

    def analyse_batch(self, texts: List[str], batch_size: int = 8) -> List[Dict]:
        """
        Analyse sentiment for a batch of texts
        
        Args:
            texts (List[str]): List of texts to analyse
            batch_size (int): Batch size for processing
            
        Returns:
            List[Dict]: List of sentiment analysis results
        """
        results = []
        
        try:
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Preprocess batch
                processed_texts = [self.preprocess_text(text) for text in batch_texts]
                
                # Tokenise
                inputs = self.tokeniser(
                    processed_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                # Get model outputs
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=1)
                
                # Process each prediction in the batch
                batch_scores = predictions.cpu().numpy()
                for scores in batch_scores:
                    sentiment_score = float(scores[0] - scores[2])
                    label_idx = np.argmax(scores)
                    sentiment_label = self.labels[label_idx]
                    confidence = float(scores[label_idx])
                    
                    results.append({
                        'sentiment': sentiment_label,
                        'score': sentiment_score,
                        'confidence': confidence,
                        'probabilities': {
                            'positive': float(scores[0]),
                            'neutral': float(scores[1]),
                            'negative': float(scores[2])
                        }
                    })
            
            return results
            
        except Exception as e:
            logging.error(f"Error in batch analysis: {str(e)}")
            return [self.analyse_text(text) for text in texts]  # Fallback to single processing

    def get_model_info(self) -> Dict:
        """
        Get information about the model configuration
        
        Returns:
            Dict: Model information
        """
        return {
            'model_name': self.model.config.name_or_path,
            'device': str(self.device),
            'vocab_size': self.model.config.vocab_size,
            'hidden_size': self.model.config.hidden_size,
            'num_labels': self.model.config.num_labels
        }