"""
FastAPI Server initialization for AIDANN
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config.settings import settings as app_settings


# Initialize FastAPI app
app = FastAPI(
    title=app_settings.PROJECT_NAME,
    version=app_settings.VERSION
)


# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed for production
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health", tags=["health"])
async def health_check():
    """
    Simple health check endpoint to verify the server is running.
    """
    return {"status": "healthy"}


# Include API routes under the institutional /v1 prefix
app.include_router(api_router, prefix="/v1")
