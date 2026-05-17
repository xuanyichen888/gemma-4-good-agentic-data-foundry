from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterator

import requests


DEFAULT_OLLAMA_MODEL = os.getenv("ADF_OLLAMA_MODEL", "gemma4:e4b")
DEFAULT_OLLAMA_URL = os.getenv("ADF_OLLAMA_URL", "http://localhost:11434")

# ── Inference defaults ────────────────────────────────────────
# num_ctx: context window size for KV cache.
#   gemma4:e4b default is 128 K — allocating that much KV cache for a
#   ~500-token prompt wastes ~60× memory and makes every decode step slow.
#   Our longest prompt (answer summary with 8 rows) is well under 1 500 tokens.
NUM_CTX = 2048

# num_predict: max tokens to generate.
#   SQL queries: ≤100 tokens.  3-bullet answers: ≤400 tokens.
NUM_PREDICT_SQL   = 150
NUM_PREDICT_TEXT  = 512   # schema review, validation, answer summary


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
    num_predict: int = NUM_PREDICT_TEXT
    num_ctx: int = NUM_CTX

    def _options(self) -> dict[str, Any]:
        return {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": self.num_predict,
            "num_ctx": self.num_ctx,
        }

    def generate(self, prompt: str) -> str:
        """Blocking generation. Returns full response string."""
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": self._options(),
            },
            timeout=600,
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        return str(payload.get("response", "")).strip()

    def stream_generate(self, prompt: str) -> Iterator[str]:
        """Streaming generation. Yields tokens as they arrive.

        Use this in the UI so the user sees output immediately instead of
        waiting for the full response.  The HTTP connection stays open until
        the model is done or num_predict is reached.
        """
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": True,
                "options": self._options(),
            },
            stream=True,
            timeout=600,
        )
        response.raise_for_status()
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            chunk: dict[str, Any] = json.loads(raw_line)
            token = chunk.get("response", "")
            if token:
                yield token
            if chunk.get("done"):
                break


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
    row_preview = str(sample_rows[:3])          # 3 rows is enough context
    warning_text = "\n".join(f"- {w}" for w in warnings) if warnings else "- No warnings found."
    return f"""
You are a data quality advisor. A small community organization imported a CSV.

Warnings:
{warning_text}

Columns: {column_summary}

Sample records (3 rows):
{row_preview}

Write exactly 3 bullet points:
- What this warning means for day-to-day service operations
- Which gap poses the highest risk to client follow-up and why
- One concrete action staff can take this week

Be specific and practical. No jargon.
""".strip()


def build_schema_review_prompt(
    columns: list[dict[str, str]],
    sample_rows: list[dict[str, Any]],
) -> str:
    column_lines = "\n".join(
        f"- {col['name']}: {col['sqlite_type']}"
        for col in columns
    )
    row_preview = str(sample_rows[:2])          # 2 rows is enough to spot type issues
    return f"""
You are a database design advisor reviewing an auto-inferred schema.

Schema:
{column_lines}

Sample data (2 rows):
{row_preview}

Write exactly 3 bullet points:
- Any columns where the inferred type looks wrong (e.g. ZIP as INTEGER loses leading zeros)
- One column whose values suggest a data entry consistency problem
- One additional column this organization should track for better reporting

Reference actual column names.
""".strip()


def build_answer_summary_prompt(
    question: str,
    sql: str,
    rows: list[dict[str, Any]],
    provenance: list[dict[str, Any]],
) -> str:
    preview_rows = rows[:5]
    preview_provenance = provenance[:5]
    return f"""
You are helping a community organization audit a database answer.

Write exactly 3 bullet points:
- What the answer says
- What source evidence supports it
- One data quality caveat

Use only the information below. Do not invent facts.

Question: {question}
SQL: {sql}
Rows: {preview_rows}
Provenance: {preview_provenance}
""".strip()
