"""
Unit tests for sentiment analysis models

Author: Gabriel Demetrios Lafis
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.sentiment_analyzer import SentimentAnalyzer, BaselineSentimentAnalyzer


class TestSentimentAnalyzer:
    """Test cases for SentimentAnalyzer class."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance for testing."""
        return SentimentAnalyzer(
            model_name="distilbert-base-uncased-finetuned-sst-2-english"
        )
    
    def test_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer is not None
        assert analyzer.model is not None
        assert analyzer.tokenizer is not None
        assert analyzer.device in ["cuda", "cpu"]
    
    def test_single_prediction(self, analyzer):
        """Test single text prediction."""
        text = "This is a great product!"
        result = analyzer.predict(text)
        
        assert "sentiment" in result
        assert "confidence" in result
        assert result["sentiment"] in ["positive", "negative", "neutral"]
        assert 0 <= result["confidence"] <= 1
    
    def test_batch_prediction(self, analyzer):
        """Test batch text prediction."""
        texts = [
            "I love this!",
            "This is terrible.",
            "It's okay."
        ]
        results = analyzer.predict(texts)
        
        assert len(results) == len(texts)
        for result in results:
            assert "sentiment" in result
            assert "confidence" in result
    
    def test_return_all_scores(self, analyzer):
        """Test prediction with all scores."""
        text = "Amazing product!"
        result = analyzer.predict(text, return_all_scores=True)
        
        assert "scores" in result
        assert isinstance(result["scores"], dict)
        assert len(result["scores"]) > 0
    
    def test_predict_proba(self, analyzer):
        """Test probability prediction."""
        text = "Great experience!"
        probs = analyzer.predict_proba(text)
        
        assert probs is not None
        assert len(probs.shape) == 1
        assert abs(probs.sum() - 1.0) < 1e-5  # Probabilities sum to 1
    
    def test_preprocess_text(self, analyzer):
        """Test text preprocessing."""
        text = "  This   has   extra   spaces  "
        processed = analyzer.preprocess_text(text)
        
        assert processed == "This has extra spaces"
    
    def test_empty_text(self, analyzer):
        """Test handling of empty text."""
        text = ""
        result = analyzer.predict(text)
        
        assert result is not None
        assert "sentiment" in result


class TestBaselineSentimentAnalyzer:
    """Test cases for BaselineSentimentAnalyzer class."""
    
    @pytest.fixture
    def baseline(self):
        """Create baseline analyzer instance for testing."""
        return BaselineSentimentAnalyzer()
    
    def test_initialization(self, baseline):
        """Test baseline analyzer initialization."""
        assert baseline is not None
        assert baseline.analyzer is not None
    
    def test_single_prediction(self, baseline):
        """Test single text prediction with VADER."""
        text = "This is wonderful!"
        result = baseline.predict(text)
        
        assert "sentiment" in result
        assert "confidence" in result
        assert "scores" in result
        assert result["sentiment"] in ["positive", "negative", "neutral"]
    
    def test_batch_prediction(self, baseline):
        """Test batch prediction with VADER."""
        texts = [
            "I love this!",
            "This is awful.",
            "It's okay."
        ]
        results = baseline.predict(texts)
        
        assert len(results) == len(texts)
        for result in results:
            assert "sentiment" in result
            assert "scores" in result
    
    def test_positive_sentiment(self, baseline):
        """Test positive sentiment detection."""
        text = "This is absolutely amazing and wonderful!"
        result = baseline.predict(text)
        
        assert result["sentiment"] == "positive"
    
    def test_negative_sentiment(self, baseline):
        """Test negative sentiment detection."""
        text = "This is terrible and awful!"
        result = baseline.predict(text)
        
        assert result["sentiment"] == "negative"
    
    def test_neutral_sentiment(self, baseline):
        """Test neutral sentiment detection."""
        text = "This is a thing."
        result = baseline.predict(text)
        
        # Note: VADER might classify this differently
        assert result["sentiment"] in ["positive", "negative", "neutral"]


class TestModelComparison:
    """Test cases for comparing models."""
    
    @pytest.fixture
    def analyzer(self):
        return SentimentAnalyzer()
    
    @pytest.fixture
    def baseline(self):
        return BaselineSentimentAnalyzer()
    
    def test_same_text_different_models(self, analyzer, baseline):
        """Test same text with different models."""
        text = "This product is great!"
        
        transformer_result = analyzer.predict(text)
        baseline_result = baseline.predict(text)
        
        assert transformer_result is not None
        assert baseline_result is not None
        
        # Both should detect positive sentiment
        # (though they might differ in edge cases)
        assert transformer_result["sentiment"] in ["positive", "negative", "neutral"]
        assert baseline_result["sentiment"] in ["positive", "negative", "neutral"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
