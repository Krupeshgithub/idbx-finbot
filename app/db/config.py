"""
Database configuration helpers.
Keeps local Postgres and AlloyDB wiring in one place.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.core.config.settings import settings


@dataclass(frozen=True)
class DatabaseConfig:
    backend: str
    url: str
    alloydb_enabled: bool
    alloydb_instance_uri: str | None


def build_database_url() -> str:
    """
    Build a SQLAlchemy URL.
    For AlloyDB via Auth Proxy the standard Postgres URL is sufficient.
    """
    if settings.DATABASE_URL:
        return settings.DATABASE_URL

    if settings.ALLOYDB_ENABLED:
        host = settings.ALLOYDB_HOST
        port = settings.ALLOYDB_PORT
        database = settings.ALLOYDB_DATABASE
        user = settings.ALLOYDB_USER
        password = settings.ALLOYDB_PASSWORD
        return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"

    # Open-source/dev default: zero-dependency local SQLite.
    # Docker deployments should set DATABASE_URL explicitly (e.g., postgres service).
    return "sqlite:///./aidann.db"


def get_database_config() -> DatabaseConfig:
    return DatabaseConfig(
        backend=settings.DB_BACKEND,
        url=build_database_url(),
        alloydb_enabled=settings.ALLOYDB_ENABLED,
        alloydb_instance_uri=settings.ALLOYDB_INSTANCE_URI or None,
    )
