"""
FastAPI Application for Sentiment Analysis

This module provides a REST API for sentiment analysis using transformer models.

Author: Gabriel Demetrios Lafis
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
import sys
import os
from loguru import logger

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sentiment_analyzer import SentimentAnalyzer, BaselineSentimentAnalyzer

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="Advanced sentiment analysis using transformer models (BERT, RoBERTa, DistilBERT)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
analyzer: Optional[SentimentAnalyzer] = None
baseline_analyzer: Optional[BaselineSentimentAnalyzer] = None


# Pydantic models for request/response
class TextInput(BaseModel):
    """Single text input for sentiment analysis."""
    text: str = Field(..., min_length=1, max_length=5000, description="Text to analyze")
    return_all_scores: bool = Field(False, description="Return scores for all sentiment classes")
    return_attention: bool = Field(False, description="Return attention weights")
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty or only whitespace')
        return v


class BatchTextInput(BaseModel):
    """Batch text input for sentiment analysis."""
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to analyze")
    return_all_scores: bool = Field(False, description="Return scores for all sentiment classes")
    batch_size: int = Field(32, ge=1, le=64, description="Batch size for processing")
    
    @validator('texts')
    def texts_not_empty(cls, v):
        if any(not text.strip() for text in v):
            raise ValueError('All texts must be non-empty')
        return v


class SentimentResponse(BaseModel):
    """Response model for sentiment prediction."""
    sentiment: str = Field(..., description="Predicted sentiment (positive, negative, neutral)")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score for prediction")
    scores: Optional[Dict[str, float]] = Field(None, description="Scores for all classes")
    attention: Optional[List[List[float]]] = Field(None, description="Attention weights")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class BatchSentimentResponse(BaseModel):
    """Response model for batch sentiment prediction."""
    results: List[SentimentResponse]
    total_texts: int
    total_processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_name: Optional[str]
    device: Optional[str]


class ModelInfo(BaseModel):
    """Model information response."""
    model_name: str
    device: str
    num_labels: int
    max_length: int


class ComparisonResponse(BaseModel):
    """Response for model comparison."""
    text: str
    transformer_result: SentimentResponse
    baseline_result: Dict


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global analyzer, baseline_analyzer
    
    try:
        logger.info("Loading sentiment analysis models...")
        
        # Load transformer model
        model_name = os.getenv(
            "MODEL_NAME",
            "distilbert-base-uncased-finetuned-sst-2-english"
        )
        analyzer = SentimentAnalyzer(model_name=model_name)
        
        # Load baseline model
        baseline_analyzer = BaselineSentimentAnalyzer()
        
        logger.success("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down API...")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Sentiment Analysis API",
        "version": "1.0.0",
        "author": "Gabriel Demetrios Lafis",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if analyzer is not None else "unhealthy",
        "model_loaded": analyzer is not None,
        "model_name": analyzer.model_name if analyzer else None,
        "device": analyzer.device if analyzer else None
    }


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """Get information about the loaded model."""
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": analyzer.model_name,
        "device": analyzer.device,
        "num_labels": analyzer.num_labels,
        "max_length": analyzer.max_length
    }


@app.post("/predict", response_model=SentimentResponse, tags=["Prediction"])
async def predict_sentiment(input_data: TextInput):
    """
    Predict sentiment for a single text.
    
    Args:
        input_data: Text input with optional parameters
        
    Returns:
        Sentiment prediction with confidence and optional scores/attention
    """
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = analyzer.predict(
            input_data.text,
            return_all_scores=input_data.return_all_scores,
            return_attention=input_data.return_attention
        )
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch_predict", response_model=BatchSentimentResponse, tags=["Prediction"])
async def batch_predict_sentiment(input_data: BatchTextInput):
    """
    Predict sentiment for multiple texts.
    
    Args:
        input_data: Batch of texts with optional parameters
        
    Returns:
        List of sentiment predictions
    """
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        import time
        start_time = time.time()
        
        results = analyzer.batch_predict(
            input_data.texts,
            batch_size=input_data.batch_size,
            show_progress=False
        )
        
        total_time = (time.time() - start_time) * 1000
        
        return {
            "results": results,
            "total_texts": len(input_data.texts),
            "total_processing_time_ms": total_time
        }
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.post("/explain", tags=["Explainability"])
async def explain_prediction(input_data: TextInput):
    """
    Explain sentiment prediction with token importance.
    
    Args:
        input_data: Text input
        
    Returns:
        Prediction with explanation (top important tokens)
    """
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = analyzer.explain_prediction(input_data.text, top_k=10)
        return result
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@app.post("/compare", response_model=ComparisonResponse, tags=["Comparison"])
async def compare_models(input_data: TextInput):
    """
    Compare transformer model with VADER baseline.
    
    Args:
        input_data: Text input
        
    Returns:
        Predictions from both models for comparison
    """
    if analyzer is None or baseline_analyzer is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        transformer_result = analyzer.predict(
            input_data.text,
            return_all_scores=True
        )
        baseline_result = baseline_analyzer.predict(input_data.text)
        
        return {
            "text": input_data.text,
            "transformer_result": transformer_result,
            "baseline_result": baseline_result
        }
    except Exception as e:
        logger.error(f"Comparison error: {e}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@app.get("/examples", tags=["Examples"])
async def get_examples():
    """Get example texts for testing the API."""
    return {
        "examples": [
            {
                "text": "This product is absolutely amazing! Best purchase ever!",
                "expected_sentiment": "positive"
            },
            {
                "text": "Terrible experience. Waste of money. Very disappointed.",
                "expected_sentiment": "negative"
            },
            {
                "text": "It's okay, nothing special. Average quality.",
                "expected_sentiment": "neutral"
            },
            {
                "text": "The customer service was excellent, but the product quality was poor.",
                "expected_sentiment": "mixed"
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn
    
    # Run the API
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
