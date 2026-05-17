from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable


@dataclass(frozen=True)
class ColumnSpec:
    source_name: str
    name: str
    sqlite_type: str
    nullable: bool


def normalize_column_name(name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if not normalized:
        return "column"
    if normalized[0].isdigit():
        return f"col_{normalized}"
    return normalized


def infer_sqlite_type(values: Iterable[str]) -> str:
    observed = [str(value).strip() for value in values if str(value).strip()]
    if not observed:
        return "TEXT"

    if all(_is_int(value) for value in observed):
        return "INTEGER"
    if all(_is_float(value) for value in observed):
        return "REAL"
    if all(_is_date(value) for value in observed):
        return "TEXT"
    return "TEXT"


def infer_schema(rows: list[dict[str, str]]) -> list[ColumnSpec]:
    if not rows:
        return []

    columns = list(rows[0].keys())
    used_names: set[str] = set()
    schema: list[ColumnSpec] = []

    for source_name in columns:
        base_name = normalize_column_name(source_name)
        name = base_name
        suffix = 2
        while name in used_names:
            name = f"{base_name}_{suffix}"
            suffix += 1
        used_names.add(name)

        values = [row.get(source_name, "") for row in rows]
        sqlite_type = infer_sqlite_type(values)
        nullable = any(str(value).strip() == "" for value in values)
        schema.append(
            ColumnSpec(
                source_name=source_name,
                name=name,
                sqlite_type=sqlite_type,
                nullable=nullable,
            )
        )

    return schema


def create_table_sql(table_name: str, columns: list[ColumnSpec]) -> str:
    safe_table = normalize_column_name(table_name)
    column_defs = ["_adf_row_id INTEGER PRIMARY KEY AUTOINCREMENT"]
    for column in columns:
        null_part = "" if column.nullable else " NOT NULL"
        column_defs.append(f"{column.name} {column.sqlite_type}{null_part}")
    return f"CREATE TABLE IF NOT EXISTS {safe_table} ({', '.join(column_defs)});"


def _is_int(value: str) -> bool:
    try:
        int(value)
        return True
    except ValueError:
        return False


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _is_date(value: str) -> bool:
    formats = ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y")
    return any(_matches_date_format(value, fmt) for fmt in formats)


def _matches_date_format(value: str, fmt: str) -> bool:
    try:
        datetime.strptime(value, fmt)
        return True
    except ValueError:
        return False

