"""
Engine and session management for operational storage.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager

from sqlalchemy import create_engine, text
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
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 5,
        "options": (
            f"-c statement_timeout={settings.DB_STATEMENT_TIMEOUT_MS} "
            f"-c idle_in_transaction_session_timeout={settings.DB_STATEMENT_TIMEOUT_MS * 2}"
        ),
    }

engine_kwargs = {
    "echo": settings.DB_ECHO,
    "future": True,
    "pool_pre_ping": True,
    "connect_args": connect_args,
}

if db_config.url.startswith("postgresql"):
    engine_kwargs.update(
        {
            "pool_size": settings.DB_POOL_SIZE,
            "max_overflow": settings.DB_MAX_OVERFLOW,
            "pool_timeout": settings.DB_POOL_TIMEOUT,
            "pool_recycle": 60,  # Reduced to 60s to aggressively prevent stale SSL sessions
            "pool_use_lifo": True, # Prefer recently used connections
        }
    )

engine = create_engine(db_config.url, **engine_kwargs)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def warm_database_pool() -> None:
    """
    Open and return one connection to the pool during startup.
    This moves first-connect latency away from the first user message.
    """
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    logger.info("[Database] Connection pool warmed.")


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
