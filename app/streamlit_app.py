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
    build_schema_review_prompt,
    build_validation_agent_prompt,
    get_ollama_status,
    is_gemma_model,
)
from agentic_data_foundry.query import (
    EXAMPLE_QUESTIONS,
    UnsafeQueryError,
    answer_question,
    answer_with_gemma_repair,
)


st.set_page_config(page_title="Agentic Data Foundry", layout="wide")

st.title("Agentic Data Foundry")
st.caption("Trusted local database construction for community support records · Gemma 4 Good Hackathon")

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
        st.session_state["schema_review"] = None
        st.session_state["validation_analysis"] = None

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
            st.success("Gemma model active.")
        else:
            st.warning("No Gemma model selected. For the hackathon demo, use a Gemma model.")
    else:
        selected_model = ollama_status.selected_model
        st.error("Ollama not reachable.")
        st.caption("Run `ollama serve` and pull `gemma3n:e4b` to enable all agent features.")

result = st.session_state.get("build_result")

if not result:
    st.info("Upload a CSV or use the synthetic example, then click **Build database** in the sidebar.")
    st.stop()

metric_cols = st.columns(4)
metric_cols[0].metric("Rows imported", result["row_count"])
metric_cols[1].metric("Columns", len(result["columns"]))
metric_cols[2].metric("Validation warnings", len(result["validation"]))
metric_cols[3].metric("Storage", "SQLite (local)")

build_tab, ask_tab, evidence_tab = st.tabs(["Build & Validate", "Ask", "Evidence"])

with build_tab:
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Inferred Schema")
        st.dataframe(pd.DataFrame(result["columns"]), use_container_width=True)

        # Gemma Agent 1: Schema Reviewer
        if ollama_status.available:
            if st.button("Run schema review agent", use_container_width=True):
                sample_rows = run_sql(db_path, f"SELECT * FROM {result['table_name']} LIMIT 5")
                with st.spinner(f"{selected_model} reviewing schema..."):
                    review = OllamaGemmaClient(model=selected_model).generate(
                        build_schema_review_prompt(result["columns"], sample_rows)
                    )
                st.session_state["schema_review"] = review

        schema_review = st.session_state.get("schema_review")
        if schema_review:
            st.info(schema_review)

    with right:
        st.subheader("Validation Report")
        if result["validation"]:
            for warning in result["validation"]:
                st.warning(warning)
        else:
            st.success("No missing-value warnings.")

        # Gemma Agent 2: Validation Analyst
        if ollama_status.available and result["validation"]:
            if st.button("Explain warnings with Gemma agent", use_container_width=True):
                sample_rows = run_sql(db_path, f"SELECT * FROM {result['table_name']} LIMIT 5")
                with st.spinner(f"{selected_model} analysing data quality..."):
                    analysis = OllamaGemmaClient(model=selected_model).generate(
                        build_validation_agent_prompt(result["validation"], result["columns"], sample_rows)
                    )
                st.session_state["validation_analysis"] = analysis

        validation_analysis = st.session_state.get("validation_analysis")
        if validation_analysis:
            st.info(validation_analysis)

    st.subheader("Database Preview")
    rows = run_sql(db_path, f"SELECT * FROM {result['table_name']} LIMIT 20")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

with ask_tab:
    question_choice = st.selectbox("Example question", EXAMPLE_QUESTIONS)
    question = st.text_input("Edit or type your own question", value=question_choice)
    use_gemma = st.checkbox("Generate SQL with local Ollama model (Gemma Agent 3)", value=False)
    generate_summary = st.checkbox("Generate Gemma explanation of the answer", value=use_gemma)

    if st.button("Run trusted query", type="primary"):
        st.session_state["last_answer"] = None
        st.session_state["last_summary"] = None
        st.session_state["repair_count"] = 0

        if use_gemma:
            if not ollama_status.available:
                st.error("Ollama is not reachable. Start Ollama or uncheck local model generation.")
                st.stop()
            client = OllamaGemmaClient(model=selected_model)
            try:
                with st.spinner(f"{selected_model} generating SQL (with auto-repair if needed)..."):
                    answer, repair_count = answer_with_gemma_repair(
                        db_path=db_path,
                        table_name=result["table_name"],
                        columns=result["columns"],
                        question=question,
                        client=client,
                    )
                st.session_state["last_answer"] = answer
                st.session_state["repair_count"] = repair_count
            except UnsafeQueryError as error:
                st.error(f"Blocked unsafe SQL after repair attempts: {error}")
                st.stop()
            except Exception as error:
                st.error(f"Query failed: {error}")
                st.stop()
        else:
            try:
                answer = answer_question(
                    db_path=db_path,
                    table_name=result["table_name"],
                    question=question,
                    generated_sql=None,
                )
                st.session_state["last_answer"] = answer
            except UnsafeQueryError as error:
                st.error(f"Blocked unsafe SQL: {error}")
                st.stop()
            except Exception as error:
                st.error(f"Query failed: {error}")
                st.stop()

        if generate_summary and ollama_status.available:
            answer = st.session_state["last_answer"]
            summary_prompt = build_answer_summary_prompt(
                question=answer.question,
                sql=answer.sql,
                rows=answer.rows,
                provenance=answer.provenance,
            )
            with st.spinner(f"{selected_model} explaining the answer..."):
                st.session_state["last_summary"] = OllamaGemmaClient(
                    model=selected_model
                ).generate(summary_prompt)

    answer = st.session_state.get("last_answer")
    if answer:
        repair_count = st.session_state.get("repair_count", 0)
        source_label = answer.source
        if repair_count > 0:
            st.success(f"SQL auto-repaired in {repair_count} step{'s' if repair_count > 1 else ''} by Gemma.")
        st.caption(f"SQL source: {source_label}")
        st.code(answer.sql, language="sql")
        st.dataframe(pd.DataFrame(answer.rows), use_container_width=True)

        summary = st.session_state.get("last_summary")
        if summary:
            st.subheader("Gemma Explanation")
            st.write(summary)

with evidence_tab:
    answer = st.session_state.get("last_answer")
    if not answer:
        st.info("Run a trusted query in the Ask tab to see source evidence here.")
    elif answer.provenance:
        st.subheader("Source Provenance")
        st.caption("Each row below traces back to a specific row in your original CSV file.")
        st.dataframe(pd.DataFrame(answer.provenance), use_container_width=True)
    else:
        st.info("This aggregate query does not map to individual source rows.")
