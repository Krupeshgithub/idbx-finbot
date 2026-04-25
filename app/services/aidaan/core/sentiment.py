import logging
from typing import Dict, Any, List

logger = logging.getLogger("aidaan.sentiment")

class NewsSentimentAnalyzer:
    """
    A professional-grade sentiment analysis module using FinBERT.
    Implemented as a Singleton with Lazy Loading to avoid blocking backend startup.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NewsSentimentAnalyzer, cls).__new__(cls)
            cls._instance.is_loaded = False
            cls._instance.nlp_pipeline = None
        return cls._instance

    def load_model(self):
        """Lazy load the model only when first needed."""
        if self.is_loaded:
            return

        try:
            from transformers import pipeline
            import torch
            
            logger.info("Initializing FinBERT Pipeline. This might take a moment to load weights...")
            
            # Use GPU if available (good for high-frequency trading latency), otherwise CPU
            device = 0 if torch.cuda.is_available() else -1
            
            self.nlp_pipeline = pipeline(
                "sentiment-analysis", 
                model="ProsusAI/finbert", 
                tokenizer="ProsusAI/finbert",
                device=device
            )
            self.is_loaded = True
            logger.info("FinBERT Model initialized successfully.")
        except ImportError as e:
            logger.error(f"FinBERT dependencies missing. Please install torch and transformers: {e}")
        except Exception as e:
            logger.error(f"Error loading FinBERT: {e}")

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Process a list of headlines and assign financial sentiment scores natively.
        """
        if not texts:
            return []
            
        import time
        start_time = time.monotonic()
            
        # Ensure model is loaded before inference
        self.load_model()
        
        if not self.is_loaded or not self.nlp_pipeline:
            logger.warning("[FinBERT] Model unavailable. Defaulting to Neutral sentiment.")
            return [{"label": "neutral", "score": 0.0, "confidence": 0.0} for _ in texts]
            
        try:
            # FinBERT returns: [{'label': 'positive', 'score': 0.89}, ...]
            results = self.nlp_pipeline(texts)
            processed_results = []
            
            pos_count = neg_count = neu_count = 0
            
            for res in results:
                label = res['label'].lower() 
                confidence = float(res['score'])
                
                # Polarity mapping (-1.0 to 1.0) for the Visual State Machine
                polarity = 0.0
                if label == "positive":
                    polarity = confidence
                    pos_count += 1
                elif label == "negative":
                    polarity = -confidence
                    neg_count += 1
                else:
                    neu_count += 1
                    
                processed_results.append({
                    "label": label,
                    "score": polarity,
                    "confidence": confidence
                })
                
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.info(
                f"[FinBERT] Batch Analysis Complete | "
                f"Headlines: {len(texts)} | Latency: {elapsed_ms:.2f}ms | "
                f"Impact/Spread: POS:{pos_count} NEG:{neg_count} NEU:{neu_count}"
            )
            return processed_results
            
        except Exception as e:
            logger.error(f"[FinBERT] Error in batch sentiment analysis: {e}")
            return [{"label": "neutral", "score": 0.0, "confidence": 0.0} for _ in texts]

# Export a single global analyzer
sentiment_analyzer = NewsSentimentAnalyzer()
