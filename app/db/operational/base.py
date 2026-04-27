"""
Shared reflection-based helpers for operational schema access.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy import MetaData, Table, desc, func, insert, select
from sqlalchemy.orm import Session


class ReflectedTableRegistry:
    """
    Lazily reflects tables so we can safely work with heterogeneous schemas.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], Table] = {}

    def table(self, session: Session, *, schema: str, name: str) -> Table:
        key = (schema, name)
        if key in self._cache:
            return self._cache[key]

        metadata = MetaData()
        table = Table(name, metadata, schema=schema, autoload_with=session.get_bind())
        self._cache[key] = table
        return table


registry = ReflectedTableRegistry()


def normalize_scalar(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    if isinstance(value, (list, dict, str, int, float, bool)) or value is None:
        return value
    return str(value)


def row_to_dict(row: Any, *, columns: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    mapping = row._mapping
    keys = columns or mapping.keys()
    return {key: normalize_scalar(mapping.get(key)) for key in keys}


def fetch_all(
    session: Session,
    *,
    schema: str,
    table_name: str,
    filters: Optional[Dict[str, Any]] = None,
    order_by: Optional[str] = None,
    order_desc: bool = False,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    table = registry.table(session, schema=schema, name=table_name)
    query = select(table)

    for column_name, value in (filters or {}).items():
        if value is None or column_name not in table.c:
            continue
        query = query.where(table.c[column_name] == value)

    if order_by and order_by in table.c:
        ordering = desc(table.c[order_by]) if order_desc else table.c[order_by]
        query = query.order_by(ordering)
    if limit:
        query = query.limit(limit)

    return [row_to_dict(row) for row in session.execute(query).fetchall()]


def fetch_one(
    session: Session,
    *,
    schema: str,
    table_name: str,
    filters: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    rows = fetch_all(
        session,
        schema=schema,
        table_name=table_name,
        filters=filters,
        limit=1,
    )
    return rows[0] if rows else None


def count_rows(session: Session, *, schema: str, table_name: str) -> int:
    table = registry.table(session, schema=schema, name=table_name)
    return int(session.execute(select(func.count()).select_from(table)).scalar_one())


def insert_row(
    session: Session,
    *,
    schema: str,
    table_name: str,
    values: Dict[str, Any],
) -> Dict[str, Any]:
    table = registry.table(session, schema=schema, name=table_name)
    payload = {key: value for key, value in values.items() if key in table.c}
    session.execute(insert(table).values(**payload))
    return payload
