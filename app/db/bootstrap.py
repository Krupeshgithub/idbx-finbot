"""
Schema initialization and synthetic seed bootstrap.
"""
from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.schema import CreateSchema

from app.core.config.settings import settings
from app.db.base import Base
from app.db.models import User
from app.db.schema_config import is_read_only_table, schema_for_table, schemas_enabled
from app.db.seed import seed_synthetic_data
from app.db.session import engine, get_db_session

logger = logging.getLogger(__name__)


def initialize_database() -> None:
    if settings.DB_AUTO_CREATE:
        if schemas_enabled():
            schema_name = schema_for_table("conversations")
            if schema_name:
                with engine.begin() as connection:
                    connection.execute(CreateSchema(schema_name, if_not_exists=True))
            writable_tables = [
                table
                for table in Base.metadata.sorted_tables
                if not is_read_only_table(table.name)
            ]
            Base.metadata.create_all(bind=engine, tables=writable_tables)
        else:
            Base.metadata.create_all(bind=engine)
        logger.info("[Database] Schema ensured.")

    if not settings.DB_AUTO_SEED:
        return

    if schemas_enabled():
        logger.info("[Database] Seed skipped; schema-routed deployment uses existing reference data.")
        return

    with get_db_session() as session:
        has_user = session.execute(select(User.id).limit(1)).scalar_one_or_none()
        if has_user:
            logger.info("[Database] Seed skipped; operational data already exists.")
            return

    seed_synthetic_data()
