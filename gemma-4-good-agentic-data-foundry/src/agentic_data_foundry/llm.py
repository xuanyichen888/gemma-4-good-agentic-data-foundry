from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests


DEFAULT_OLLAMA_MODEL = os.getenv("ADF_OLLAMA_MODEL", "gemma3n:e4b")
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
            timeout=120,
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
- Return SQL only.
- Use only the table and columns listed below.
- Do not write INSERT, UPDATE, DELETE, DROP, ALTER, or PRAGMA.
- If the question cannot be answered, return: SELECT 'Question cannot be answered from this table' AS answer;

Table: {table_name}
Columns:
{column_lines}

Question: {question}
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
