"""
FastAPI Server initialization for AIDAAN
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config.settings import settings as app_settings
from app.core.logging_config import setup_logging

# Configure structured logging before anything else
setup_logging()

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


@app.get("/health", tags=["health"])
async def health_check():
    """
    Simple health check endpoint to verify the server is running.
    """
    return {"status": "healthy"}


# Include API routes under the /v1 prefix
app.include_router(api_router, prefix="/v1")
