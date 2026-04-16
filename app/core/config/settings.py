"""
Centralized Configuration for AIDAAN - Institutional Standard.
=============================================================
This module handles all application settings using Pydantic-Settings.
Configurations are loaded from environment variables or the local .env file.
"""
from typing import Optional

from pydantic_settings import (
    BaseSettings, 
    SettingsConfigDict
)


class Settings(BaseSettings):
    """
    General application settings.
    Sensitive keys (API keys, project IDs) are loaded from .env.
    """

    # --- Project Metadata ---
    PROJECT_NAME: str = "AIDANN-Backend"
    VERSION: str = "1.0.0"

    # --- API Credentials (Loaded from Environment) ---
    GOOGLE_API_KEY: Optional[str] = None
    ALPHA_VANTAGE_API_KEY: Optional[str] = None

    # --- Google Cloud Platform ---
    GOOGLE_CLOUD_PROJECT: Optional[str] = None
    GOOGLE_CLOUD_LOCATION: str = "global"
    VERTEX_AI_MODEL_NAME: str = "gemini-2.5-flash"
    VERTEX_AI_REASONING_MODEL_NAME: str = "gemini-2.5-pro"
    VERTEX_AI_API_VERSION: str = "v1"
    VERTEX_AI_TEMPERATURE: float = 0.2
    VERTEX_AI_MAX_OUTPUT_TOKENS: int = 8192
    VERTEX_AI_USE_EXPRESS_MODE: bool = True
    VERTEX_AI_SERVICE_ACCOUNT_FILE: Optional[str] = None
    VERTEX_AI_SERVICE_ACCOUNT_JSON: Optional[str] = None
    VERTEX_AI_ENABLE_GOOGLE_SEARCH: bool = False
    VERTEX_AI_ENABLE_URL_CONTEXT: bool = False
    VERTEX_AI_ENABLE_CODE_EXECUTION: bool = False
    FINBERT_ENDPOINT_ID: Optional[str] = None

    # --- System & Safety ---
    ENABLE_KILL_SWITCH: bool = True
    MAX_PASSWORD_BYTES: int = 72

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
    MCP_SERVER_COMMAND: str = "python3"
    MCP_SERVER_ARGS: str = "app/services/aidaan/mcp/market.py"

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
