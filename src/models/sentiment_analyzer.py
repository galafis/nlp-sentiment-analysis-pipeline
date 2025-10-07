"""
Sentiment Analysis Module

This module provides a comprehensive sentiment analysis interface using
transformer-based models (BERT, RoBERTa, DistilBERT) for text classification.

Author: Gabriel Demetrios Lafis
"""

import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
from typing import Dict, List, Union, Optional
import numpy as np
from loguru import logger
import time


class SentimentAnalyzer:
    """
    A comprehensive sentiment analyzer using transformer models.
    
    Supports multiple pre-trained and fine-tuned models for sentiment analysis
    with configurable parameters and batch processing capabilities.
    
    Attributes:
        model_name (str): Name of the transformer model to use
        device (str): Device to run inference on ('cuda' or 'cpu')
        max_length (int): Maximum sequence length for tokenization
        num_labels (int): Number of sentiment classes
    """
    
    SENTIMENT_LABELS = {
        0: "negative",
        1: "neutral",
        2: "positive"
    }
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        device: Optional[str] = None,
        max_length: int = 512,
        num_labels: int = 3
    ):
        """
        Initialize the sentiment analyzer.
        
        Args:
            model_name: HuggingFace model identifier or path to local model
            device: Device to use for inference ('cuda', 'cpu', or None for auto)
            max_length: Maximum sequence length for tokenization
            num_labels: Number of sentiment classes (2 for binary, 3 for pos/neg/neu)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.num_labels = num_labels
        
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing SentimentAnalyzer with model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels
            )
            self.model.to(self.device)
            self.model.eval()
            
            logger.success("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess input text before tokenization.
        
        Args:
            text: Raw input text
            
        Returns:
            Preprocessed text
        """
        # Basic preprocessing
        text = text.strip()
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        return text
    
    def predict(
        self,
        text: Union[str, List[str]],
        return_all_scores: bool = False,
        return_attention: bool = False
    ) -> Union[Dict, List[Dict]]:
        """
        Predict sentiment for single text or batch of texts.
        
        Args:
            text: Input text or list of texts
            return_all_scores: Whether to return scores for all classes
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary or list of dictionaries containing:
                - sentiment: Predicted sentiment label
                - confidence: Confidence score for predicted class
                - scores: Scores for all classes (if return_all_scores=True)
                - attention: Attention weights (if return_attention=True)
                - processing_time_ms: Inference time in milliseconds
        """
        start_time = time.time()
        
        # Handle single text vs batch
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        # Preprocess texts
        texts = [self.preprocess_text(t) for t in texts]
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=return_attention)
            logits = outputs.logits
            
            # Get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Get predictions
            predictions = torch.argmax(probs, dim=-1)
            confidences = torch.max(probs, dim=-1).values
        
        # Prepare results
        results = []
        for i, (pred, conf, prob) in enumerate(
            zip(predictions, confidences, probs)
        ):
            result = {
                "sentiment": self.SENTIMENT_LABELS.get(
                    pred.item(),
                    f"class_{pred.item()}"
                ),
                "confidence": conf.item(),
            }
            
            if return_all_scores:
                result["scores"] = {
                    self.SENTIMENT_LABELS.get(j, f"class_{j}"): prob[j].item()
                    for j in range(len(prob))
                }
            
            if return_attention and outputs.attentions is not None:
                # Average attention across all layers and heads
                attention = torch.stack(outputs.attentions).mean(dim=(0, 1))
                result["attention"] = attention[i].cpu().numpy().tolist()
            
            results.append(result)
        
        # Add processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        for result in results:
            result["processing_time_ms"] = processing_time / len(results)
        
        # Return single result or batch
        return results[0] if is_single else results
    
    def predict_proba(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Get probability distributions for sentiment classes.
        
        Args:
            text: Input text or list of texts
            
        Returns:
            Numpy array of shape (n_samples, n_classes) with probabilities
        """
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        # Preprocess and tokenize
        texts = [self.preprocess_text(t) for t in texts]
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
        
        probs_array = probs.cpu().numpy()
        return probs_array[0] if is_single else probs_array
    
    def explain_prediction(self, text: str, top_k: int = 10) -> Dict:
        """
        Explain prediction using attention weights.
        
        Args:
            text: Input text to explain
            top_k: Number of top tokens to return
            
        Returns:
            Dictionary with explanation details
        """
        # Get prediction with attention
        result = self.predict(text, return_attention=True, return_all_scores=True)
        
        # Tokenize to get tokens
        tokens = self.tokenizer.tokenize(self.preprocess_text(text))
        
        if "attention" in result and len(tokens) > 0:
            # Get average attention for each token
            attention = np.array(result["attention"])
            # Average over sequence dimension (excluding special tokens)
            token_importance = attention.mean(axis=0)[1:len(tokens)+1]
            
            # Get top-k tokens
            top_indices = np.argsort(token_importance)[-top_k:][::-1]
            top_tokens = [
                {
                    "token": tokens[i],
                    "importance": float(token_importance[i])
                }
                for i in top_indices if i < len(tokens)
            ]
            
            result["top_tokens"] = top_tokens
        
        return result
    
    def batch_predict(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Predict sentiment for large batch of texts with batching.
        
        Args:
            texts: List of input texts
            batch_size: Size of processing batches
            show_progress: Whether to show progress bar
            
        Returns:
            List of prediction dictionaries
        """
        from tqdm import tqdm
        
        results = []
        
        # Process in batches
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Processing batches")
        
        for i in iterator:
            batch = texts[i:i + batch_size]
            batch_results = self.predict(batch, return_all_scores=True)
            results.extend(batch_results)
        
        return results
    
    def __repr__(self) -> str:
        return (
            f"SentimentAnalyzer(model={self.model_name}, "
            f"device={self.device}, "
            f"num_labels={self.num_labels})"
        )


class BaselineSentimentAnalyzer:
    """
    Baseline sentiment analyzer using VADER for comparison.
    """
    
    def __init__(self):
        """Initialize VADER sentiment analyzer."""
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self.analyzer = SentimentIntensityAnalyzer()
        logger.info("Initialized VADER baseline analyzer")
    
    def predict(self, text: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """
        Predict sentiment using VADER.
        
        Args:
            text: Input text or list of texts
            
        Returns:
            Dictionary or list of dictionaries with sentiment predictions
        """
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        results = []
        for t in texts:
            scores = self.analyzer.polarity_scores(t)
            
            # Determine sentiment
            compound = scores['compound']
            if compound >= 0.05:
                sentiment = "positive"
            elif compound <= -0.05:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            result = {
                "sentiment": sentiment,
                "confidence": abs(compound),
                "scores": {
                    "positive": scores['pos'],
                    "negative": scores['neg'],
                    "neutral": scores['neu'],
                    "compound": compound
                }
            }
            results.append(result)
        
        return results[0] if is_single else results


if __name__ == "__main__":
    # Example usage
    analyzer = SentimentAnalyzer()
    
    # Test texts
    texts = [
        "This product is absolutely amazing! I love it!",
        "Terrible experience. Would not recommend.",
        "It's okay, nothing special.",
    ]
    
    # Single prediction
    result = analyzer.predict(texts[0], return_all_scores=True)
    print(f"\nSingle prediction:")
    print(f"Text: {texts[0]}")
    print(f"Result: {result}")
    
    # Batch prediction
    results = analyzer.predict(texts, return_all_scores=True)
    print(f"\nBatch predictions:")
    for text, result in zip(texts, results):
        print(f"Text: {text}")
        print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.3f})")
    
    # Explanation
    explanation = analyzer.explain_prediction(texts[0])
    print(f"\nExplanation for: {texts[0]}")
    print(f"Top tokens: {explanation.get('top_tokens', [])[:5]}")
    
    # Baseline comparison
    baseline = BaselineSentimentAnalyzer()
    baseline_result = baseline.predict(texts[0])
    print(f"\nVADER baseline:")
    print(f"Result: {baseline_result}")
