import logging
from typing import Dict, Any, List
import os

logger = logging.getLogger("aidaan.sentiment")

class NewsSentimentAnalyzer:
    """
    GPU-Accelerated FinBERT sentiment analysis with optimizations for NVIDIA L4.
    Implemented as a Singleton with pre-loading and GPU memory optimization.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NewsSentimentAnalyzer, cls).__new__(cls)
            cls._instance.is_loaded = False
            cls._instance.nlp_pipeline = None
            cls._instance.device = None
            cls._instance.model = None
            cls._instance.tokenizer = None
        return cls._instance

    def load_model(self):
        """Load model with GPU optimization and pre-warming."""
        if self.is_loaded:
            return

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            from accelerate import Accelerator
            
            logger.info("🚀 Initializing GPU-Accelerated FinBERT Pipeline...")
            
            # Force GPU usage
            if not torch.cuda.is_available():
                logger.warning("⚠️ CUDA not available! Falling back to CPU (will be slow)")
                self.device = torch.device("cpu")
            else:
                self.device = torch.device("cuda:0")
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"✓ GPU Detected: {gpu_name} ({gpu_memory:.1f}GB VRAM)")
            
            # Load model and tokenizer - use local path if available
            model_name = "ProsusAI/finbert"
            local_model_path = "/models/finbert"
            
            import os
            if os.path.exists(local_model_path) and os.listdir(local_model_path):
                model_source = local_model_path
                logger.info(f"Loading {model_name} from local path: {local_model_path}")
            else:
                model_source = model_name
                logger.info(f"Loading {model_name} from HuggingFace Hub...")
            
            logger.info(f"Loading {model_name} to {self.device}...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_source)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_source,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  # FP16 for speed
                use_safetensors=True,  # Use safetensors for secure loading
            )
            
            # Move model to GPU
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            # Enable GPU optimizations
            if torch.cuda.is_available():
                # Enable TF32 for faster matmul on Ampere+ GPUs (L4 is Ampere)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
                # Enable cuDNN auto-tuner for optimal performance
                torch.backends.cudnn.benchmark = True
                
                # Pre-allocate GPU memory
                torch.cuda.empty_cache()
                
                logger.info("✓ GPU optimizations enabled (TF32, cuDNN benchmark)")
            
            # Warm-up inference (pre-compile CUDA kernels)
            logger.info("Warming up GPU with dummy inference...")
            dummy_texts = ["This is a test sentence for GPU warm-up."] * 8
            self._inference(dummy_texts)
            
            self.is_loaded = True
            
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
                memory_reserved = torch.cuda.memory_reserved(0) / 1024**2
                logger.info(
                    f"✓ FinBERT loaded on GPU | "
                    f"Memory: {memory_allocated:.1f}MB allocated, {memory_reserved:.1f}MB reserved"
                )
            else:
                logger.info("✓ FinBERT loaded on CPU")
                
        except ImportError as e:
            logger.error(f"❌ FinBERT dependencies missing: {e}")
        except Exception as e:
            logger.error(f"❌ Error loading FinBERT: {e}")

    def _inference(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Internal inference method with GPU optimization."""
        import torch
        
        # Tokenize with padding and truncation
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Move inputs to GPU
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference with no gradient computation (faster)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Convert to CPU for processing
        predictions = predictions.cpu().numpy()
        
        # Map predictions to labels
        label_map = {0: "positive", 1: "negative", 2: "neutral"}
        results = []
        
        for pred in predictions:
            label_idx = pred.argmax()
            confidence = float(pred[label_idx])
            label = label_map[label_idx]
            
            results.append({
                "label": label,
                "confidence": confidence,
                "scores": {
                    "positive": float(pred[0]),
                    "negative": float(pred[1]),
                    "neutral": float(pred[2])
                }
            })
        
        return results

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        GPU-accelerated batch sentiment analysis.
        Optimized for high throughput with NVIDIA L4.
        """
        if not texts:
            return []
            
        import time
        start_time = time.monotonic()
            
        # Ensure model is loaded
        self.load_model()
        
        if not self.is_loaded or not self.model:
            logger.warning("[FinBERT] Model unavailable. Defaulting to Neutral sentiment.")
            return [{"label": "neutral", "score": 0.0, "confidence": 0.0} for _ in texts]
            
        try:
            # Run GPU inference
            results = self._inference(texts)
            
            # Process results
            processed_results = []
            pos_count = neg_count = neu_count = 0
            
            for res in results:
                label = res['label'].lower()
                confidence = res['confidence']
                
                # Polarity mapping (-1.0 to 1.0)
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
            
            # Log with GPU stats
            gpu_info = ""
            if self.device.type == "cuda":
                import torch
                gpu_util = torch.cuda.utilization(0) if hasattr(torch.cuda, 'utilization') else 0
                gpu_mem = torch.cuda.memory_allocated(0) / 1024**2
                gpu_info = f"GPU: {gpu_util}% util, {gpu_mem:.1f}MB | "
            
            logger.info(
                f"[FinBERT] ⚡ Batch Analysis | "
                f"{gpu_info}"
                f"Headlines: {len(texts)} | Latency: {elapsed_ms:.2f}ms | "
                f"Speed: {len(texts)/elapsed_ms*1000:.1f} texts/sec | "
                f"Sentiment: POS:{pos_count} NEG:{neg_count} NEU:{neu_count}"
            )
            
            return processed_results
            
        except Exception as e:
            logger.error(f"[FinBERT] ❌ Error in batch sentiment analysis: {e}")
            return [{"label": "neutral", "score": 0.0, "confidence": 0.0} for _ in texts]

# Export a single global analyzer
sentiment_analyzer = NewsSentimentAnalyzer()
