from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agentic_data_foundry.database import build_sqlite_from_csv, run_sql
from agentic_data_foundry.llm import (
    OllamaGemmaClient,
    build_answer_summary_prompt,
    build_nl2sql_prompt,
    get_ollama_status,
    is_gemma_model,
)
from agentic_data_foundry.query import (
    EXAMPLE_QUESTIONS,
    UnsafeQueryError,
    answer_question,
)


st.set_page_config(page_title="Agentic Data Foundry", layout="wide")

st.title("Agentic Data Foundry")
st.caption("Trusted local database construction for community support records")

example_path = ROOT / "examples" / "community_intake.csv"
db_path = ROOT / "agentic_data_foundry_demo.sqlite"

with st.sidebar:
    st.header("Data Source")
    uploaded = st.file_uploader("CSV file", type=["csv"])
    use_example = uploaded is None
    if use_example:
        st.caption("Using synthetic community intake records.")
        csv_path = example_path
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix="adf_"))
        csv_path = temp_dir / uploaded.name
        csv_path.write_bytes(uploaded.getvalue())

    if st.button("Build database", type="primary", use_container_width=True):
        result = build_sqlite_from_csv(csv_path, db_path, table_name="community_intake")
        st.session_state["build_result"] = result
        st.session_state["last_answer"] = None

    st.divider()
    st.header("Local Model")
    ollama_status = get_ollama_status()
    if ollama_status.available:
        model_options = ollama_status.models or [ollama_status.selected_model]
        selected_model = st.selectbox(
            "Ollama model",
            model_options,
            index=model_options.index(ollama_status.selected_model)
            if ollama_status.selected_model in model_options
            else 0,
        )
        if is_gemma_model(selected_model):
            st.success("Gemma model detected.")
        else:
            st.warning("No Gemma model is selected. This is OK for local debugging, but the hackathon submission should use Gemma.")
    else:
        selected_model = ollama_status.selected_model
        st.error("Ollama service is not reachable.")
        st.caption("Run `ollama serve` and pull a Gemma model to enable local model generation.")

result = st.session_state.get("build_result")

if not result:
    st.info("Build the database from the sidebar to start the demo.")
    st.stop()

metric_cols = st.columns(4)
metric_cols[0].metric("Rows", result["row_count"])
metric_cols[1].metric("Columns", len(result["columns"]))
metric_cols[2].metric("Validation Warnings", len(result["validation"]))
metric_cols[3].metric("Storage", "SQLite")

build_tab, ask_tab, evidence_tab = st.tabs(["Build", "Ask", "Evidence"])

with build_tab:
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Inferred Schema")
        st.dataframe(pd.DataFrame(result["columns"]), use_container_width=True)

    with right:
        st.subheader("Validation Report")
        if result["validation"]:
            for warning in result["validation"]:
                st.warning(warning)
        else:
            st.success("No missing-value warnings.")

    st.subheader("Database Preview")
    rows = run_sql(db_path, f"SELECT * FROM {result['table_name']} LIMIT 20")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

with ask_tab:
    question_choice = st.selectbox("Question", EXAMPLE_QUESTIONS)
    question = st.text_input("Edit question", value=question_choice)
    use_gemma = st.checkbox("Generate SQL with local Ollama model", value=False)
    generate_summary = st.checkbox("Generate model explanation", value=use_gemma)

    if st.button("Run trusted query", type="primary"):
        generated_sql = None
        if use_gemma:
            if not ollama_status.available:
                st.error("Ollama is not reachable. Start Ollama or turn off local model generation.")
                st.stop()
            prompt = build_nl2sql_prompt(
                result["table_name"],
                result["columns"],
                question,
            )
            with st.spinner(f"{selected_model} is drafting SQL..."):
                generated_sql = OllamaGemmaClient(model=selected_model).generate(prompt)

        try:
            answer = answer_question(
                db_path=db_path,
                table_name=result["table_name"],
                question=question,
                generated_sql=generated_sql,
            )
            st.session_state["last_answer"] = answer
            st.session_state["last_summary"] = None
            if generate_summary and ollama_status.available:
                summary_prompt = build_answer_summary_prompt(
                    question=answer.question,
                    sql=answer.sql,
                    rows=answer.rows,
                    provenance=answer.provenance,
                )
                with st.spinner(f"{selected_model} is explaining the answer..."):
                    st.session_state["last_summary"] = OllamaGemmaClient(
                        model=selected_model
                    ).generate(summary_prompt)
        except UnsafeQueryError as error:
            st.error(f"Blocked unsafe SQL: {error}")
        except Exception as error:
            st.error(f"Query failed: {error}")

    answer = st.session_state.get("last_answer")
    if answer:
        st.caption(f"SQL source: {answer.source}")
        st.code(answer.sql, language="sql")
        st.dataframe(pd.DataFrame(answer.rows), use_container_width=True)
        summary = st.session_state.get("last_summary")
        if summary:
            st.subheader("Model Explanation")
            st.write(summary)

with evidence_tab:
    answer = st.session_state.get("last_answer")
    if not answer:
        st.info("Run a trusted query to see source evidence.")
    elif answer.provenance:
        st.subheader("Source Provenance")
        st.dataframe(pd.DataFrame(answer.provenance), use_container_width=True)
    else:
        st.info("This aggregate answer does not map to individual source rows.")
