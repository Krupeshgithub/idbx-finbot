"""
Engine and session management for operational storage.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
from sqlalchemy.orm import Session, sessionmaker

from app.core.config.settings import settings
from app.db.config import get_database_config

logger = logging.getLogger(__name__)

db_config = get_database_config()

connect_args = {}
if db_config.url.startswith("sqlite:"):
    # Keep SQLite responsive even under accidental concurrent access during demos/tests.
    connect_args = {"check_same_thread": False, "timeout": 1.0}
elif db_config.url.startswith("postgresql"):
    connect_args = {
        "connect_timeout": settings.DB_CONNECT_TIMEOUT_SECONDS,
        "sslmode": settings.ALLOYDB_SSL_MODE,
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 5,
        # NOTE: Cloud SQL Proxy does not support statement_timeout in options parameter.
        # Use SET commands after connection instead if needed.
    }

engine_kwargs = {
    "echo": settings.DB_ECHO,
    "future": True,
    "pool_pre_ping": True,
    "connect_args": connect_args,
}

if db_config.url.startswith("postgresql") and settings.DB_POOL_ENABLED:
    engine_kwargs.update(
        {
            "pool_size": settings.DB_POOL_SIZE,
            "max_overflow": settings.DB_MAX_OVERFLOW,
            "pool_timeout": settings.DB_POOL_TIMEOUT,
            "pool_recycle": settings.DB_POOL_RECYCLE_SECONDS,
            "pool_use_lifo": True,  # Prefer recently used connections.
        }
    )
elif db_config.url.startswith("postgresql"):
    engine_kwargs["poolclass"] = NullPool

engine = create_engine(db_config.url, **engine_kwargs)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def warm_database_pool() -> None:
    """
    Pre-warm multiple connections in the pool during startup.
    This moves first-connect latency away from the first user messages.
    Opens 5 connections to ensure pool is ready for concurrent requests.
    """
    connections = []
    try:
        # Open 5 connections to pre-warm the pool
        for i in range(5):
            conn = engine.connect()
            conn.execute(text("SELECT 1"))
            connections.append(conn)
        
        logger.info("[Database] Connection pool warmed with 5 connections.")
    finally:
        # Close all connections to return them to the pool
        for conn in connections:
            conn.close()


@contextmanager
def get_db_session() -> Session:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def dispose_engine() -> None:
    """
    Dispose of the SQLAlchemy engine and close all connections in the pool.
    Call this during application shutdown.
    """
    engine.dispose()
    logger.info("[Database] Connection pool disposed.")
