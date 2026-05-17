from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .database import get_provenance_for_rows, run_readonly_sql

if TYPE_CHECKING:
    from .llm import OllamaGemmaClient


class UnsafeQueryError(ValueError):
    """Raised when generated SQL is not safe enough to execute."""


@dataclass(frozen=True)
class QuestionAnswer:
    question: str
    sql: str
    rows: list[dict[str, Any]]
    provenance: list[dict[str, Any]]
    source: str


EXAMPLE_QUESTIONS = [
    "Which clients need follow-up this week?",
    "What are the most common unmet needs?",
    "Which ZIP codes have the most transportation requests?",
    "How many open cases are older than 14 days?",
    "Which languages should we prioritize for translated outreach?",
    "Which records are missing follow-up dates?",
    "How many unique ZIP codes are represented?",
    "Which open cases have been waiting the longest?",
    "How many clients are there per household size?",
    "Show all closed cases with their notes.",
]


FORBIDDEN_SQL_TERMS = {
    "alter",
    "attach",
    "create",
    "delete",
    "detach",
    "drop",
    "insert",
    "pragma",
    "reindex",
    "replace",
    "truncate",
    "update",
    "vacuum",
}


def answer_question(
    db_path: str | Path,
    table_name: str,
    question: str,
    generated_sql: str | None = None,
) -> QuestionAnswer:
    source = "Gemma SQL" if generated_sql else "deterministic fallback"
    sql = clean_sql(generated_sql or fallback_sql_for_question(question, table_name))
    validate_select_sql(sql, table_name)
    rows = run_readonly_sql(db_path, sql)
    provenance = provenance_for_answer(db_path, table_name, rows)
    return QuestionAnswer(
        question=question,
        sql=sql,
        rows=rows,
        provenance=provenance,
        source=source,
    )


def clean_sql(sql: str) -> str:
    cleaned = sql.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:sql)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    return cleaned.rstrip(";").strip()


def validate_select_sql(sql: str, table_name: str) -> None:
    cleaned = clean_sql(sql)
    lowered = cleaned.lower()

    if not cleaned:
        raise UnsafeQueryError("The generated SQL is empty.")
    if not re.match(r"^(select|with)\b", lowered):
        raise UnsafeQueryError("Only SELECT queries are allowed.")
    if ";" in cleaned:
        raise UnsafeQueryError("Multiple SQL statements are not allowed.")
    if "--" in cleaned or "/*" in cleaned or "*/" in cleaned:
        raise UnsafeQueryError("SQL comments are not allowed.")

    found_terms = {
        term
        for term in FORBIDDEN_SQL_TERMS
        if re.search(rf"\b{re.escape(term)}\b", lowered)
    }
    if found_terms:
        blocked = ", ".join(sorted(found_terms))
        raise UnsafeQueryError(f"Forbidden SQL terms: {blocked}")

    referenced_tables = re.findall(
        r"\b(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
        lowered,
    )
    allowed_tables = {table_name.lower(), "adf_provenance"}
    unknown_tables = [table for table in referenced_tables if table not in allowed_tables]
    if unknown_tables:
        blocked = ", ".join(sorted(set(unknown_tables)))
        raise UnsafeQueryError(f"Unknown table reference: {blocked}")


def answer_with_gemma_repair(
    db_path: str | Path,
    table_name: str,
    columns: list[dict[str, Any]],
    question: str,
    client: "OllamaGemmaClient",
    max_repair: int = 2,
) -> tuple["QuestionAnswer", int]:
    """Generate SQL with Gemma, auto-repair on safety or execution errors.

    Returns (answer, repair_count) where repair_count is the number of repairs needed.
    """
    from .llm import build_nl2sql_prompt, build_sql_repair_prompt

    prompt = build_nl2sql_prompt(table_name, columns, question)
    sql = clean_sql(client.generate(prompt))
    last_error: Exception = RuntimeError("no attempts made")

    for attempt in range(max_repair + 1):
        try:
            validate_select_sql(sql, table_name)
            rows = run_readonly_sql(db_path, sql)
            provenance = provenance_for_answer(db_path, table_name, rows)
            repairs = attempt
            source = "Gemma SQL" if repairs == 0 else f"Gemma SQL (auto-repaired in {repairs} step{'s' if repairs > 1 else ''})"
            return (
                QuestionAnswer(question=question, sql=sql, rows=rows, provenance=provenance, source=source),
                repairs,
            )
        except (UnsafeQueryError, sqlite3.OperationalError, sqlite3.DatabaseError) as error:
            last_error = error
            if attempt == max_repair:
                raise last_error
            repair_prompt = build_sql_repair_prompt(table_name, columns, question, sql, str(error))
            sql = clean_sql(client.generate(repair_prompt))

    raise last_error


def provenance_for_answer(
    db_path: str | Path,
    table_name: str,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    row_ids = sorted(
        {
            int(row["_adf_row_id"])
            for row in rows
            if "_adf_row_id" in row and str(row["_adf_row_id"]).isdigit()
        }
    )
    return get_provenance_for_rows(db_path, table_name, row_ids)


def fallback_sql_for_question(question: str, table_name: str) -> str:
    q = question.lower()

    if "missing" in q and ("follow" in q or "date" in q):
        return f"""
            SELECT _adf_row_id, client_id, zip_code, primary_need, status, follow_up_date
            FROM {table_name}
            WHERE follow_up_date IS NULL OR follow_up_date = ''
            ORDER BY client_id
            LIMIT 25
        """

    if "transport" in q and ("zip" in q or "where" in q):
        return f"""
            SELECT zip_code, COUNT(*) AS transportation_requests
            FROM {table_name}
            WHERE lower(primary_need) LIKE '%transport%'
            GROUP BY zip_code
            ORDER BY transportation_requests DESC, zip_code ASC
            LIMIT 10
        """

    if ("common" in q or "top" in q or "most" in q) and (
        "need" in q or "unmet" in q
    ):
        return f"""
            SELECT primary_need, COUNT(*) AS cases
            FROM {table_name}
            GROUP BY primary_need
            ORDER BY cases DESC, primary_need ASC
            LIMIT 10
        """

    if "older" in q and "14" in q:
        return f"""
            SELECT COUNT(*) AS open_cases_older_than_14_days
            FROM {table_name}
            WHERE status = 'open'
              AND date(intake_date) <= date('now', '-14 days')
        """

    if "language" in q or "translated" in q or "translation" in q:
        return f"""
            SELECT language, COUNT(*) AS clients
            FROM {table_name}
            GROUP BY language
            ORDER BY clients DESC, language ASC
            LIMIT 10
        """

    if "unique" in q and "zip" in q:
        return f"""
            SELECT COUNT(DISTINCT zip_code) AS unique_zip_codes
            FROM {table_name}
        """

    if ("waiting" in q or "longest" in q or "earliest" in q) and "open" in q:
        return f"""
            SELECT _adf_row_id, client_id, intake_date, zip_code, primary_need, status
            FROM {table_name}
            WHERE status = 'open'
            ORDER BY intake_date ASC
            LIMIT 10
        """

    if "household" in q and ("size" in q or "per" in q or "many" in q):
        return f"""
            SELECT household_size, COUNT(*) AS clients
            FROM {table_name}
            GROUP BY household_size
            ORDER BY household_size ASC
        """

    if "closed" in q and ("note" in q or "show" in q or "all" in q):
        return f"""
            SELECT _adf_row_id, client_id, intake_date, primary_need, status, notes
            FROM {table_name}
            WHERE status = 'closed'
            ORDER BY intake_date DESC
            LIMIT 20
        """

    return f"""
        SELECT _adf_row_id, client_id, zip_code, primary_need, follow_up_date, status
        FROM {table_name}
        WHERE status = 'open'
          AND follow_up_date IS NOT NULL
          AND follow_up_date != ''
          AND date(follow_up_date) <= date('now', '+7 days')
        ORDER BY follow_up_date ASC
        LIMIT 10
    """
