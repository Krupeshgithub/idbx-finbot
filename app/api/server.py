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
    Initialize operational storage after the app boots.
    """
    initialize_database()
    logger.info("[Startup] Database bootstrap complete.")


@app.get("/health", tags=["health"])
async def health_check():
    """
    Simple health check endpoint to verify the server is running.
    """
    return {
        "status": "healthy",
        "database_backend": db_config.backend,
        "alloydb_enabled": db_config.alloydb_enabled,
    }


# Include API routes under the /v1 prefix
app.include_router(api_router, prefix="/v1")
