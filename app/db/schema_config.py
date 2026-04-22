"""
Schema routing helpers for operational storage.
"""
from __future__ import annotations

from app.core.config.settings import settings


PUBLIC_USER_TABLE = "users"
READ_ONLY_TABLES = {
    "users",
    "desks",
    "desk_memberships",
    "desk_limits",
}


def schemas_enabled() -> bool:
    database_url = (settings.DATABASE_URL or "").lower()
    return settings.ALLOYDB_ENABLED or database_url.startswith("postgresql")


def schema_for_table(table_name: str) -> str | None:
    if not schemas_enabled():
        return None
    if table_name == PUBLIC_USER_TABLE:
        return settings.DB_PUBLIC_SCHEMA
    return settings.DB_AIDAAN_SCHEMA


def table_args_for(table_name: str) -> dict[str, str]:
    schema = schema_for_table(table_name)
    return {"schema": schema} if schema else {}


def qualified_table_name(table_name: str) -> str:
    schema = schema_for_table(table_name)
    return f"{schema}.{table_name}" if schema else table_name


def is_read_only_table(table_name: str) -> bool:
    return table_name in READ_ONLY_TABLES
