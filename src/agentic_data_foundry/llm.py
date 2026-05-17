from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests


DEFAULT_OLLAMA_MODEL = os.getenv("ADF_OLLAMA_MODEL", "gemma4:e4b")
DEFAULT_OLLAMA_URL = os.getenv("ADF_OLLAMA_URL", "http://localhost:11434")


@dataclass(frozen=True)
class OllamaStatus:
    available: bool
    models: list[str]
    gemma_models: list[str]
    selected_model: str
    error: str | None = None


@dataclass(frozen=True)
class OllamaGemmaClient:
    model: str = DEFAULT_OLLAMA_MODEL
    base_url: str = DEFAULT_OLLAMA_URL

    def generate(self, prompt: str) -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                },
            },
            timeout=600,
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        return str(payload.get("response", "")).strip()


def get_ollama_status(base_url: str = DEFAULT_OLLAMA_URL) -> OllamaStatus:
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=3)
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
    except requests.RequestException as error:
        return OllamaStatus(
            available=False,
            models=[],
            gemma_models=[],
            selected_model=DEFAULT_OLLAMA_MODEL,
            error=str(error),
        )

    models = sorted(
        str(model.get("name", ""))
        for model in payload.get("models", [])
        if model.get("name")
    )
    gemma_models = [model for model in models if is_gemma_model(model)]
    selected_model = choose_default_model(models, gemma_models)
    return OllamaStatus(
        available=True,
        models=models,
        gemma_models=gemma_models,
        selected_model=selected_model,
    )


def choose_default_model(models: list[str], gemma_models: list[str]) -> str:
    if DEFAULT_OLLAMA_MODEL in models:
        return DEFAULT_OLLAMA_MODEL
    if gemma_models:
        return gemma_models[0]
    if models:
        return models[0]
    return DEFAULT_OLLAMA_MODEL


def is_gemma_model(model_name: str) -> bool:
    return model_name.lower().startswith("gemma")


def build_nl2sql_prompt(table_name: str, columns: list[dict[str, str]], question: str) -> str:
    column_lines = "\n".join(
        f"- {column['name']} ({column['sqlite_type']})"
        for column in columns
    )
    return f"""
You are a careful data assistant. Convert the user's question into one SQLite SELECT query.

Rules:
- Return SQL only. No explanation, no markdown.
- Use only the table and columns listed below.
- Do not write INSERT, UPDATE, DELETE, DROP, ALTER, or PRAGMA.
- If the question asks about specific records or individuals (not an aggregate count or group-by), always include _adf_row_id in the SELECT list so source evidence can be traced.
- If the question is a count, sum, or group-by aggregation, do NOT include _adf_row_id.
- If the question cannot be answered, return: SELECT 'Question cannot be answered from this table' AS answer;

Table: {table_name}
Columns:
- _adf_row_id (INTEGER) — internal row identifier, include for row-level queries
{column_lines}

Question: {question}
""".strip()


def build_sql_repair_prompt(
    table_name: str,
    columns: list[dict[str, str]],
    question: str,
    failed_sql: str,
    error_message: str,
) -> str:
    column_lines = "\n".join(
        f"- {column['name']} ({column['sqlite_type']})"
        for column in columns
    )
    return f"""
You are a careful data assistant. Your previous SQL query failed. Fix it.

Rules:
- Return SQL only. No explanation, no markdown.
- Use only SELECT. Never use INSERT, UPDATE, DELETE, DROP, ALTER, or PRAGMA.
- Only reference the table and columns listed below.
- No semicolons inside the query. No SQL comments.

Table: {table_name}
Columns:
{column_lines}

Question: {question}

Your previous SQL (which failed):
{failed_sql}

Error message:
{error_message}

Write the corrected SQL query now:
""".strip()


def build_validation_agent_prompt(
    warnings: list[str],
    columns: list[dict[str, str]],
    sample_rows: list[dict[str, Any]],
) -> str:
    column_summary = ", ".join(column["name"] for column in columns)
    row_preview = str(sample_rows[:5])
    warning_text = "\n".join(f"- {w}" for w in warnings) if warnings else "- No warnings found."
    return f"""
You are a data quality advisor helping a small community organization understand their records.

The system flagged these data quality warnings after importing:
{warning_text}

Database columns: {column_summary}

Sample records (first 5 rows):
{row_preview}

Write exactly 3 bullet points:
- What each warning likely means for day-to-day service operations
- Which warning poses the highest risk to client follow-up and why
- One concrete action the organization can take this week to address the most critical gap

Be specific, compassionate, and practical. Avoid technical jargon.
""".strip()


def build_schema_review_prompt(
    columns: list[dict[str, str]],
    sample_rows: list[dict[str, Any]],
) -> str:
    column_lines = "\n".join(
        f"- {col['name']}: {col['sqlite_type']} (nullable={col.get('nullable', True)})"
        for col in columns
    )
    row_preview = str(sample_rows[:3])
    return f"""
You are a database design advisor reviewing an automatically inferred schema for a community service organization.

Inferred schema:
{column_lines}

Sample data:
{row_preview}

Write exactly 3 bullet points:
- Flag any columns where the inferred type may be wrong (e.g. ZIP codes or phone numbers stored as INTEGER lose leading zeros)
- Identify one column whose values suggest a data entry consistency problem
- Suggest one additional column this organization should track for better reporting outcomes

Be specific and reference the actual column names above.
""".strip()


def build_answer_summary_prompt(
    question: str,
    sql: str,
    rows: list[dict[str, Any]],
    provenance: list[dict[str, Any]],
) -> str:
    preview_rows = rows[:8]
    preview_provenance = provenance[:8]
    return f"""
You are helping a small community organization audit a database answer.

Write a concise, careful explanation in 3 bullets:
- what the answer says
- what source evidence supports it
- one limitation or data quality caveat

Do not invent facts. Use only the question, SQL, rows, and provenance below.

Question:
{question}

SQL:
{sql}

Rows:
{preview_rows}

Provenance:
{preview_provenance}
""".strip()
