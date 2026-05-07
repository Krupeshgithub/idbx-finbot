"""
Schema initialization and synthetic seed bootstrap.
"""
from __future__ import annotations

import logging

from sqlalchemy import select, text
from sqlalchemy.schema import CreateSchema

from app.core.config.settings import settings
from app.db.base import Base
from app.db.models import User
from app.db.schema_config import is_read_only_table, schema_for_table, schemas_enabled
from app.db.seed import seed_synthetic_data
from app.db.session import engine, get_db_session, warm_database_pool

logger = logging.getLogger(__name__)


def _is_postgresql() -> bool:
    return str(engine.url).lower().startswith("postgresql")


def _ensure_vector_infrastructure() -> None:
    """
    Ensure pgvector-backed AIDAAN tables and columns exist.
    This keeps semantic-search features available without manual SQL steps.
    """
    if not _is_postgresql():
        return

    aidaan_schema = settings.DB_AIDAAN_SCHEMA
    with engine.begin() as connection:
        # pgvector extension is required for vector columns and <=> distance operator.
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        # Messages semantic memory column (used by hybrid history retrieval).
        connection.execute(
            text(
                f"""
                ALTER TABLE IF EXISTS {aidaan_schema}.messages
                ADD COLUMN IF NOT EXISTS content_vector vector(768)
                """
            )
        )

        # Corporate knowledge table for institutional Q&A.
        connection.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS {aidaan_schema}.corporate_knowledge (
                    id TEXT PRIMARY KEY,
                    category VARCHAR(64) NOT NULL,
                    question TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json JSONB DEFAULT '{{}}'::jsonb,
                    embedding_vector vector(768),
                    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        )

        # Read-only public counterparties shadow vectors in aidaan schema.
        connection.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS {aidaan_schema}.counterparty_embeddings (
                    counterparty_id UUID PRIMARY KEY,
                    embedding_vector vector(768) NOT NULL,
                    last_updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        )

        # Alias table used by ticker resolution.
        connection.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS {aidaan_schema}.ticker_aliases (
                    id TEXT PRIMARY KEY,
                    alias VARCHAR(128) NOT NULL,
                    ticker VARCHAR(32) NOT NULL,
                    priority INTEGER DEFAULT 100,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        )

        connection.execute(
            text(
                f"CREATE UNIQUE INDEX IF NOT EXISTS idx_ticker_aliases_alias ON {aidaan_schema}.ticker_aliases (LOWER(alias))"
            )
        )
        connection.execute(
            text(
                f"CREATE INDEX IF NOT EXISTS idx_messages_content_vector_ivfflat ON {aidaan_schema}.messages USING ivfflat (content_vector vector_cosine_ops) WITH (lists = 100)"
            )
        )
        connection.execute(
            text(
                f"CREATE INDEX IF NOT EXISTS idx_corporate_knowledge_vector_ivfflat ON {aidaan_schema}.corporate_knowledge USING ivfflat (embedding_vector vector_cosine_ops) WITH (lists = 100)"
            )
        )
        connection.execute(
            text(
                f"CREATE INDEX IF NOT EXISTS idx_corporate_knowledge_category ON {aidaan_schema}.corporate_knowledge (category)"
            )
        )

    logger.info("[Database] pgvector infrastructure ensured for schema '%s'.", aidaan_schema)


def initialize_database() -> None:
    warm_database_pool()

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
        _ensure_vector_infrastructure()
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
