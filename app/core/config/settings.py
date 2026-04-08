"""
Database and general application configuration for AIDANN.
"""
from typing import Optional

from pydantic_settings import (
    BaseSettings, 
    SettingsConfigDict
)


class Settings(BaseSettings):
    """
    General application settings.

    Attributes:
        PROJECT_NAME: Name of the application.
        VERSION: Current version.
        GOOGLE_CLOUD_PROJECT: GCP project ID.
        GOOGLE_CLOUD_LOCATION: GCP region.
        GOOGLE_API_KEY: API key for Google services.
        ALPHA_VANTAGE_API_KEY: API key for Alpha Vantage.
        VERTEX_AI_MODEL_NAME: Gemini model identifier.
        FINBERT_ENDPOINT_ID: FinBERT model endpoint.
        ENABLE_KILL_SWITCH: Venue-level safety switch toggle.
        REDIS_URL: Redis connection string.
        REDIS_EXPIRE: Default cache expiration in seconds.
        LSEG_API_BASE_URL: Base URL for LSEG Data Library.
        LSEG_API_SNAPSHOT_PATH: Path for snapshot queries against LSEG.
        LSEG_API_KEY: Optional API key for authenticating with LSEG.
        LSEG_API_TIMEOUT: HTTP timeout when calling LSEG.
    """
    MAX_PASSWORD_BYTES: int = 72

    GOOGLE_API_KEY: str = "AIzaSyAnNgB7Ir2i9DSYV05TNfM8ktxBgWeojYc"
    ALPHA_VANTAGE_API_KEY: str = "PJBIMYS7Q8AYB7NJ"

    PROJECT_NAME: str = "AIDANN-Backend"
    VERSION: str = "1.0.0"
    GOOGLE_CLOUD_PROJECT: str = "your-project-id"
    GOOGLE_CLOUD_LOCATION: str = "us-central1"
    VERTEX_AI_MODEL_NAME: str = "gemini-2.5-flash"
    FINBERT_ENDPOINT_ID: Optional[str] = None
    ENABLE_KILL_SWITCH: bool = True

    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_EXPIRE: int = 14400  # 4 hours

    LSEG_API_BASE_URL: str = "https://api.lseg.com/data"
    LSEG_API_KEY: Optional[str] = None
    LSEG_API_TIMEOUT: float = 5.0

    LDL_BIGQUERY_DATASET: str = "ldl_v2"
    LDL_BIGQUERY_TIMESTAMP_FIELD: str = "event_ts"
    LDL_BIGQUERY_BID_FIELD: str = "bid"
    LDL_BIGQUERY_ASK_FIELD: str = "ask"
    LDL_BIGQUERY_INSTRUMENT_FIELD: str = "instrument"

    model_config = SettingsConfigDict(
        env_file=".env", 
        case_sensitive=True, 
        extra="ignore"
    )


settings = Settings()
