"""
Schema initialization and synthetic seed bootstrap.
"""
from __future__ import annotations

import logging

from sqlalchemy import select

from app.core.config.settings import settings
from app.db.base import Base
from app.db.models import User
from app.db.seed import seed_synthetic_data
from app.db.session import engine, get_db_session

logger = logging.getLogger(__name__)


def initialize_database() -> None:
    if settings.DB_AUTO_CREATE:
        Base.metadata.create_all(bind=engine)
        logger.info("[Database] Schema ensured.")

    if not settings.DB_AUTO_SEED:
        return

    with get_db_session() as session:
        has_user = session.execute(select(User.id).limit(1)).scalar_one_or_none()
        if has_user:
            logger.info("[Database] Seed skipped; operational data already exists.")
            return

    seed_synthetic_data()

