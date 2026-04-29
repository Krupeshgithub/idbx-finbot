"""
Centralized Configuration for AIDAAN - Institutional Standard.

This module handles all application settings using Pydantic-Settings.
Configurations are loaded from environment variables or the local .env file.
"""
import sys
import os
import tempfile
import logging

from typing import Optional
from pathlib import Path
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict
)

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    General application settings.
    Sensitive keys (API keys, project IDs) are loaded from .env.
    """

    # --- Project JWT ---
    JWT_SECRET_KEY: str = "change-me-in-production"

    # --- Project Metadata ---
    PROJECT_NAME: str = "AIDANN-Backend"
    VERSION: str = "1.0.0"

    # --- API Credentials (Loaded from Environment) ---
    GOOGLE_API_KEY: Optional[str] = None
    ALPHA_VANTAGE_API_KEY: Optional[str] = None
    HF_TOKEN: Optional[str] = None

    # --- Google Cloud Platform ---
    GOOGLE_CLOUD_PROJECT: Optional[str] = None
    GOOGLE_CLOUD_LOCATION: str = "global"
    
    # --- Cloud DLP (Sensitive Data Protection) ---
    GCP_DLP_LOCATION: str = "global"
    GCP_DLP_DEIDENTIFY_TEMPLATE: Optional[str] = None
    GCP_DLP_INSPECT_TEMPLATE: Optional[str] = None
    
    # --- Cloud Logging (Audit Trails) ---
    GCP_AUDIT_LOG_NAME: str = "aidann-audit-log"
    
    # --- Vertex AI ---
    VERTEX_AI_MODEL_NAME: str = "gemini-2.5-flash"
    VERTEX_AI_REASONING_MODEL_NAME: str = "gemini-2.5-pro"
    VERTEX_AI_ROUTER_MODEL_NAME: str = "gemini-2.5-flash-lite"
    VERTEX_AI_API_VERSION: str = "v1"
    VERTEX_AI_TEMPERATURE: float = 0.2
    VERTEX_AI_MAX_OUTPUT_TOKENS: int = 8192
    VERTEX_AI_MAX_WORKERS: int = 12
    VERTEX_AI_MAX_CONCURRENT_REQUESTS: int = 4
    VERTEX_AI_USE_EXPRESS_MODE: bool = True
    VERTEX_AI_SERVICE_ACCOUNT_FILE: Optional[str] = None
    VERTEX_AI_SERVICE_ACCOUNT_JSON: Optional[str] = None
    VERTEX_AI_ENABLE_GOOGLE_SEARCH: bool = False
    VERTEX_AI_ENABLE_URL_CONTEXT: bool = False
    VERTEX_AI_ENABLE_CODE_EXECUTION: bool = False
    VERTEX_AI_ENABLE_CONTEXT_CACHING: bool = True
    VERTEX_AI_CONTEXT_CACHE_TTL_SECONDS: int = 3600
    FINBERT_ENDPOINT_ID: Optional[str] = None

    # --- System & Safety ---
    ENABLE_KILL_SWITCH: bool = True
    KILL_SWITCH_ACTIVE: bool = False
    KILL_SWITCH_REASON: str = "Kill switch is not active."
    MAX_PASSWORD_BYTES: int = 72

    # --- Multi-Language Support ---
    SUPPORTED_LANGUAGES: str = "All"

    # --- Database / AlloyDB ---
    DB_BACKEND: str = "postgresql"
    DATABASE_URL: Optional[str] = None
    DB_ECHO: bool = False
    DB_AUTO_CREATE: bool = True
    DB_AUTO_SEED: bool = True
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_TIMEOUT: float = 2.0
    DB_POOL_RECYCLE_SECONDS: int = 1800
    DB_CONNECT_TIMEOUT_SECONDS: int = 3
    DB_STATEMENT_TIMEOUT_MS: int = 2500
    DB_PUBLIC_SCHEMA: str = "public"
    DB_AIDAAN_SCHEMA: str = "aidaan"
    AIDAAN_HISTORY_WINDOW: int = 5
    ALLOYDB_ENABLED: bool = False
    ALLOYDB_USE_AUTH_PROXY: bool = False
    ALLOYDB_HOST: str = "127.0.0.1"
    ALLOYDB_PORT: int = 5432
    ALLOYDB_DATABASE: str = "aidann"
    ALLOYDB_USER: str = "aidann_app"
    ALLOYDB_PASSWORD: Optional[str] = None
    ALLOYDB_PROJECT_ID: Optional[str] = None
    ALLOYDB_REGION: Optional[str] = None
    ALLOYDB_CLUSTER_ID: Optional[str] = None
    ALLOYDB_INSTANCE_ID: Optional[str] = None
    ALLOYDB_INSTANCE_URI: Optional[str] = None
    ALLOYDB_SSL_MODE: str = "require"

    # --- Infrastructure (Redis/Cache) ---
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_EXPIRE: int = 14400  # 4 hours
    MARKET_DATA_CACHE_EXPIRE: int = 300
    LLM_CACHE_EXPIRE: int = 1800

    # --- Market Data Providers (LSEG) ---
    LSEG_API_BASE_URL: str = "https://api.lseg.com/data"
    LSEG_API_SNAPSHOT_PATH: str = "snapshot"
    LSEG_API_KEY: Optional[str] = None
    LSEG_API_TIMEOUT: float = 5.0

    # --- MCP (Model Context Protocol) ---
    MCP_SERVER_COMMAND: str = sys.executable
    MCP_SERVER_ARGS: str = "app/services/aidaan/mcp/server.py"
    MCP_TRANSPORT: str = "inprocess"  # "inprocess" (default) | "stdio"

    # --- BigQuery Data Warehouse ---
    LDL_BIGQUERY_DATASET: str = "ldl_v2"
    LDL_BIGQUERY_TIMESTAMP_FIELD: str = "event_ts"
    LDL_BIGQUERY_BID_FIELD: str = "bid"
    LDL_BIGQUERY_ASK_FIELD: str = "ask"
    LDL_BIGQUERY_INSTRUMENT_FIELD: str = "instrument"

    # Pydantic Configuration
    model_config = SettingsConfigDict(
        env_file=".env", 
        case_sensitive=True, 
        extra="ignore"
    )


# Instantiate the global settings object.
# Pydantic will automatically load values from .env on initialization.
settings = Settings()


def setup_infrastructure():
    """
    Ensures environment variables like GOOGLE_APPLICATION_CREDENTIALS and HF_TOKEN 
    are set correctly for underlying SDKs.
    """
    # 1. Hugging Face Authentication
    if settings.HF_TOKEN:
        os.environ["HF_TOKEN"] = settings.HF_TOKEN
        # Also set the older variant just in case
        os.environ["HUGGING_FACE_HUB_TOKEN"] = settings.HF_TOKEN
        logger.info("[Auth] Hugging Face Token exported to environment.")

    # 2. Google Cloud Credentials
    if settings.VERTEX_AI_SERVICE_ACCOUNT_JSON:
        try:
            credential_dir = Path(tempfile.gettempdir())
            credential_dir.mkdir(parents=True, exist_ok=True)
            credential_path = credential_dir / "aidaan-vertex-service-account.json"
            credential_path.write_text(settings.VERTEX_AI_SERVICE_ACCOUNT_JSON, encoding="utf-8")
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credential_path)
            logger.info("[GCPAuth] Using Service Account from JSON content.")
            return
        except Exception as exc:
            logger.error("[GCPAuth] Failed to write JSON credentials: %s", exc)

    if settings.VERTEX_AI_SERVICE_ACCOUNT_FILE:
        resolved_path = Path(settings.VERTEX_AI_SERVICE_ACCOUNT_FILE).expanduser()
        if resolved_path.is_file():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(resolved_path)
            logger.info("[GCPAuth] Set GOOGLE_APPLICATION_CREDENTIALS to: %s", resolved_path)
        else:
            # Check relative to /app in Docker
            docker_path = Path("/app") / settings.VERTEX_AI_SERVICE_ACCOUNT_FILE
            if docker_path.is_file():
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(docker_path)
                logger.info("[GCPAuth] Set GOOGLE_APPLICATION_CREDENTIALS")


# Run setup immediately on module load
setup_infrastructure()
