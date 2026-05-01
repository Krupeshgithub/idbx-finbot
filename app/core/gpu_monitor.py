"""
GPU Monitoring and Optimization Utilities for NVIDIA L4
"""
import logging
from typing import Dict, Any, Optional
import os

logger = logging.getLogger(__name__)


class GPUMonitor:
    """Monitor and optimize GPU usage for AIDANN."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GPUMonitor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.gpu_available = False
        self.gpu_name = None
        self.gpu_memory_total = 0
        self._initialized = True
        
        self._check_gpu()
    
    def _check_gpu(self):
        """Check GPU availability and specs."""
        try:
            import torch
            
            if torch.cuda.is_available():
                self.gpu_available = True
                self.gpu_name = torch.cuda.get_device_name(0)
                self.gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                
                logger.info(
                    f"✓ GPU Available: {self.gpu_name} | "
                    f"Memory: {self.gpu_memory_total:.1f}GB | "
                    f"CUDA Version: {torch.version.cuda}"
                )
                
                # Enable optimizations
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
                
                logger.info("✓ GPU optimizations enabled (TF32, cuDNN benchmark)")
            else:
                logger.warning("⚠️ No GPU detected - running on CPU (will be slower)")
                
        except ImportError:
            logger.warning("⚠️ PyTorch not installed - GPU monitoring unavailable")
        except Exception as e:
            logger.error(f"❌ Error checking GPU: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current GPU statistics."""
        if not self.gpu_available:
            return {
                "available": False,
                "message": "GPU not available"
            }
        
        try:
            import torch
            
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**2  # MB
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**2  # MB
            memory_free = (self.gpu_memory_total * 1024) - memory_reserved  # MB
            
            # Try to get utilization (may not work on all systems)
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = utilization.gpu
                mem_util = utilization.memory
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                pynvml.nvmlShutdown()
            except:
                gpu_util = None
                mem_util = None
                temperature = None
            
            stats = {
                "available": True,
                "name": self.gpu_name,
                "memory": {
                    "total_gb": round(self.gpu_memory_total, 2),
                    "allocated_mb": round(memory_allocated, 2),
                    "reserved_mb": round(memory_reserved, 2),
                    "free_mb": round(memory_free, 2),
                    "utilization_percent": mem_util
                },
                "compute": {
                    "utilization_percent": gpu_util,
                    "temperature_c": temperature
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting GPU stats: {e}")
            return {
                "available": True,
                "error": str(e)
            }
    
    def clear_cache(self):
        """Clear GPU cache to free memory."""
        if not self.gpu_available:
            return
        
        try:
            import torch
            torch.cuda.empty_cache()
            logger.info("✓ GPU cache cleared")
        except Exception as e:
            logger.error(f"Error clearing GPU cache: {e}")
    
    def log_stats(self):
        """Log current GPU statistics."""
        stats = self.get_stats()
        
        if not stats.get("available"):
            return
        
        if "error" in stats:
            logger.warning(f"GPU stats error: {stats['error']}")
            return
        
        mem = stats.get("memory", {})
        compute = stats.get("compute", {})
        
        log_msg = (
            f"[GPU Stats] "
            f"Memory: {mem.get('allocated_mb', 0):.0f}MB / {mem.get('total_gb', 0)*1024:.0f}MB"
        )
        
        if compute.get("utilization_percent") is not None:
            log_msg += f" | Compute: {compute['utilization_percent']}%"
        
        if compute.get("temperature_c") is not None:
            log_msg += f" | Temp: {compute['temperature_c']}°C"
        
        logger.info(log_msg)


# Global instance
gpu_monitor = GPUMonitor()
