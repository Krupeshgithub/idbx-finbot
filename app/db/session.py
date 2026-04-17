"""
Engine and session management for operational storage.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config.settings import settings
from app.db.config import get_database_config

logger = logging.getLogger(__name__)

db_config = get_database_config()

connect_args = {}
if db_config.url.startswith("sqlite:"):
    # Keep SQLite responsive even under accidental concurrent access during demos/tests.
    connect_args = {"check_same_thread": False, "timeout": 1.0}

engine = create_engine(
    db_config.url,
    echo=settings.DB_ECHO,
    future=True,
    pool_pre_ping=True,
    connect_args=connect_args,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


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
