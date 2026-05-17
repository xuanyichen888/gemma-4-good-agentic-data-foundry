from __future__ import annotations

import csv
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .schema import ColumnSpec, create_table_sql, infer_schema, normalize_column_name


def load_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def build_sqlite_from_csv(
    csv_path: str | Path,
    db_path: str | Path,
    table_name: str = "records",
) -> dict[str, Any]:
    rows = load_csv(csv_path)
    schema = infer_schema(rows)
    safe_table = normalize_column_name(table_name)

    db = sqlite3.connect(db_path)
    try:
        db.execute(f"DROP TABLE IF EXISTS {safe_table}")
        db.execute("DROP TABLE IF EXISTS adf_provenance")
        db.execute(create_table_sql(safe_table, schema))
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS adf_provenance (
                provenance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_name TEXT NOT NULL,
                row_id INTEGER NOT NULL,
                source_file TEXT NOT NULL,
                source_row_number INTEGER NOT NULL
            );
            """
        )
        _insert_rows(db, safe_table, schema, rows, str(csv_path))
        validation = validate_rows(schema, rows)
        db.commit()
    finally:
        db.close()

    return {
        "table_name": safe_table,
        "row_count": len(rows),
        "columns": [asdict(column) for column in schema],
        "validation": validation,
        "database_path": str(db_path),
    }


def validate_rows(schema: list[ColumnSpec], rows: list[dict[str, str]]) -> list[str]:
    warnings: list[str] = []
    for column in schema:
        values = [str(row.get(column.source_name, "")).strip() for row in rows]
        missing = sum(1 for value in values if not value)
        if missing:
            warnings.append(f"{column.name}: {missing} missing values")
    return warnings


def run_sql(db_path: str | Path, sql: str) -> list[dict[str, Any]]:
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    try:
        cursor = db.execute(sql)
        return [dict(row) for row in cursor.fetchall()]
    finally:
        db.close()


def run_readonly_sql(db_path: str | Path, sql: str) -> list[dict[str, Any]]:
    absolute_path = Path(db_path).resolve().as_posix()
    db = sqlite3.connect(f"file:{absolute_path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        cursor = db.execute(sql)
        return [dict(row) for row in cursor.fetchall()]
    finally:
        db.close()


def get_provenance_for_rows(
    db_path: str | Path,
    table_name: str,
    row_ids: list[int],
) -> list[dict[str, Any]]:
    if not row_ids:
        return []

    placeholders = ", ".join("?" for _ in row_ids)
    sql = f"""
        SELECT row_id, source_file, source_row_number
        FROM adf_provenance
        WHERE table_name = ? AND row_id IN ({placeholders})
        ORDER BY row_id
    """
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    try:
        cursor = db.execute(sql, [table_name, *row_ids])
        return [dict(row) for row in cursor.fetchall()]
    finally:
        db.close()


def _insert_rows(
    db: sqlite3.Connection,
    table_name: str,
    schema: list[ColumnSpec],
    rows: list[dict[str, str]],
    source_file: str,
) -> None:
    if not schema:
        return

    column_names = [column.name for column in schema]
    placeholders = ", ".join("?" for _ in column_names)
    columns_sql = ", ".join(column_names)
    insert_sql = f"INSERT INTO {table_name} ({columns_sql}) VALUES ({placeholders})"

    for source_row_number, row in enumerate(rows, start=2):
        values = [row.get(column.source_name, "") for column in schema]
        cursor = db.execute(insert_sql, values)
        db.execute(
            """
            INSERT INTO adf_provenance
                (table_name, row_id, source_file, source_row_number)
            VALUES (?, ?, ?, ?)
            """,
            (table_name, cursor.lastrowid, source_file, source_row_number),
        )
