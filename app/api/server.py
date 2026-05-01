"""
FastAPI Server initialization for AIDAAN
"""
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config.settings import settings as app_settings
from app.core.logging_config import setup_logging
from app.db.bootstrap import initialize_database
from app.db.config import get_database_config
from app.services.kill_switch import kill_switch_service

# Configure structured logging before anything else
setup_logging()
logger = logging.getLogger(__name__)
db_config = get_database_config()

# Initialize FastAPI app
app = FastAPI(
    title=app_settings.PROJECT_NAME,
    version=app_settings.VERSION
)

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """
    Initialize operational storage and pre-load GPU models.
    """
    initialize_database()
    logger.info("[Startup] Database bootstrap complete.")
    
    # Initialize GPU monitoring
    try:
        from app.core.gpu_monitor import gpu_monitor
        gpu_monitor.log_stats()
        logger.info("[Startup] GPU monitoring initialized.")
    except Exception as e:
        logger.warning(f"[Startup] GPU monitoring failed (non-critical): {e}")
    
    # Pre-initialize DLP client to avoid first-request latency
    try:
        from app.core.dlp_client import dlp_client
        dlp_client.initialize()
        logger.info("[Startup] DLP client pre-initialized successfully.")
    except Exception as e:
        logger.warning(f"[Startup] DLP client pre-initialization failed (non-critical): {e}")
    
    # Pre-load FinBERT model on GPU to avoid cold start
    try:
        import asyncio
        from app.services.aidaan.core.sentiment import sentiment_analyzer
        logger.info("[Startup] 🚀 Pre-loading FinBERT model on GPU...")
        await asyncio.to_thread(sentiment_analyzer.load_model)
        logger.info("[Startup] ✓ FinBERT model pre-loaded on GPU successfully.")
        
        # Log GPU stats after model loading
        from app.core.gpu_monitor import gpu_monitor
        gpu_monitor.log_stats()
    except Exception as e:
        logger.warning(f"[Startup] FinBERT pre-load failed (non-critical): {e}")


@app.get("/health", tags=["health"])
async def health_check():
    """
    Simple health check endpoint to verify the server is running.
    """
    return {
        "status": "healthy",
        "database_backend": db_config.backend,
        "alloydb_enabled": db_config.alloydb_enabled,
        "kill_switch_active": kill_switch_service.is_active(),
    }


@app.get("/gpu-stats", tags=["health"])
async def gpu_stats():
    """
    Get current GPU statistics and utilization.
    """
    try:
        from app.core.gpu_monitor import gpu_monitor
        stats = gpu_monitor.get_stats()
        return stats
    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        }


# Include API routes under the /v1 prefix
app.include_router(api_router, prefix="/v1")
