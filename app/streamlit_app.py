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

st.set_page_config(
    page_title="Agentic Data Foundry",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Main background ── */
.stApp { background-color: #F0F4F8; }

/* ── Hero header ── */
.hero {
    background: linear-gradient(135deg, #0F4C75 0%, #1B6CA8 60%, #00A896 100%);
    border-radius: 16px;
    padding: 36px 40px 28px;
    margin-bottom: 28px;
    color: white;
}
.hero h1 { font-size: 2rem; font-weight: 700; margin: 0 0 6px; }
.hero p  { font-size: 0.95rem; opacity: 0.85; margin: 0; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: white;
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    border-left: 4px solid #1B6CA8;
}

/* ── Tabs ── */
[data-testid="stTabs"] button {
    font-weight: 600;
    font-size: 0.9rem;
    padding: 10px 20px;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #1B6CA8;
    border-bottom-color: #1B6CA8;
}

/* ── Section cards ── */
.card {
    background: white;
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

/* ── Section headings ── */
.section-title {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #64748B;
    margin-bottom: 12px;
}

/* ── Agent badge ── */
.agent-badge {
    display: inline-block;
    background: #EFF6FF;
    color: #1B6CA8;
    font-size: 0.72rem;
    font-weight: 600;
    border-radius: 999px;
    padding: 3px 10px;
    margin-bottom: 10px;
    letter-spacing: 0.04em;
}

/* ── Gemma output box ── */
.gemma-output {
    background: #F0FDF9;
    border-left: 4px solid #00A896;
    border-radius: 0 8px 8px 0;
    padding: 16px 18px;
    font-size: 0.9rem;
    line-height: 1.7;
    color: #1E293B;
    margin-top: 12px;
}

/* ── SQL block ── */
[data-testid="stCode"] {
    border-radius: 10px;
    font-size: 0.85rem;
}

/* ── Primary buttons ── */
[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, #0F4C75, #1B6CA8);
    border: none;
    border-radius: 8px;
    font-weight: 600;
    letter-spacing: 0.02em;
    padding: 10px 24px;
    transition: opacity 0.15s;
}
[data-testid="stButton"] button[kind="primary"]:hover { opacity: 0.88; }

/* ── Secondary buttons ── */
[data-testid="stButton"] button:not([kind="primary"]) {
    border-radius: 8px;
    font-weight: 500;
    border-color: #CBD5E1;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: white;
    border-right: 1px solid #E2E8F0;
}
[data-testid="stSidebar"] .stButton button {
    border-radius: 8px;
    font-weight: 600;
}

/* ── Status pill ── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 0.8rem;
    font-weight: 600;
}
.status-ok   { background: #DCFCE7; color: #166534; }
.status-warn { background: #FEF9C3; color: #854D0E; }
.status-err  { background: #FEE2E2; color: #991B1B; }

/* ── Provenance table ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Hide Streamlit branding ── */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Hero header ──────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🏗️ Agentic Data Foundry</h1>
  <p>Local-first Gemma 4 agents that turn messy community records into a trusted, queryable database &nbsp;·&nbsp; Gemma 4 Good Hackathon</p>
</div>
""", unsafe_allow_html=True)

example_path = ROOT / "examples" / "community_intake.csv"
db_path = ROOT / "agentic_data_foundry_demo.sqlite"

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Data Source")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    use_example = uploaded is None
    if use_example:
        st.caption("No file uploaded — using the synthetic community intake dataset.")
        csv_path = example_path
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix="adf_"))
        csv_path = temp_dir / uploaded.name
        csv_path.write_bytes(uploaded.getvalue())

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if st.button("Build database", type="primary", use_container_width=True):
        with st.spinner("Building database…"):
            result = build_sqlite_from_csv(csv_path, db_path, table_name="community_intake")
        st.session_state["build_result"] = result
        st.session_state["last_answer"] = None
        st.session_state["schema_review"] = None
        st.session_state["validation_analysis"] = None
        st.success("Database ready.")

    st.markdown("---")
    st.markdown("### Local Gemma 4 Model")
    ollama_status = get_ollama_status()

    if ollama_status.available:
        model_options = ollama_status.models or [ollama_status.selected_model]
        selected_model = st.selectbox(
            "Model",
            model_options,
            index=model_options.index(ollama_status.selected_model)
            if ollama_status.selected_model in model_options else 0,
            label_visibility="collapsed",
        )
        if is_gemma_model(selected_model):
            st.markdown('<div class="status-pill status-ok">● Gemma 4 active</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-pill status-warn">⚠ Non-Gemma model</div>', unsafe_allow_html=True)
    else:
        selected_model = ollama_status.selected_model
        st.markdown('<div class="status-pill status-err">✕ Ollama unreachable</div>', unsafe_allow_html=True)
        st.caption("Run `ollama serve` and pull `gemma4:e4b`.")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem; color:#94A3B8; line-height:1.6'>
    <b>Agent pipeline</b><br>
    1 · Schema Reviewer<br>
    2 · Validation Analyst<br>
    3 · SQL Generator + Auto-repair<br>
    4 · Answer Explainer
    </div>
    """, unsafe_allow_html=True)

# ── Gate: database must be built ─────────────────────────────
result = st.session_state.get("build_result")

if not result:
    st.markdown("""
    <div style='text-align:center; padding: 60px 20px; color:#64748B'>
        <div style='font-size:3rem'>📂</div>
        <div style='font-size:1.1rem; font-weight:600; margin-top:12px'>Upload a CSV or use the example dataset</div>
        <div style='font-size:0.9rem; margin-top:6px'>Then click <b>Build database</b> in the sidebar to start.</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Metrics row ───────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows imported", result["row_count"])
c2.metric("Columns", len(result["columns"]))
c3.metric("Validation warnings", len(result["validation"]))
c4.metric("Storage", "SQLite · local")

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────
build_tab, ask_tab, evidence_tab = st.tabs(["🗂  Build & Validate", "💬  Ask", "🔍  Evidence"])

# ── BUILD TAB ─────────────────────────────────────────────────
with build_tab:
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown('<div class="section-title">Inferred Schema</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(result["columns"]), use_container_width=True, height=240)

        if ollama_status.available:
            st.markdown('<div class="agent-badge">Agent 1 · Schema Reviewer</div>', unsafe_allow_html=True)
            if st.button("Run schema review", use_container_width=True):
                sample_rows = run_sql(db_path, f"SELECT * FROM {result['table_name']} LIMIT 5")
                with st.spinner(f"{selected_model} reviewing schema…"):
                    review = OllamaGemmaClient(model=selected_model).generate(
                        build_schema_review_prompt(result["columns"], sample_rows)
                    )
                st.session_state["schema_review"] = review

        schema_review = st.session_state.get("schema_review")
        if schema_review:
            st.markdown(f'<div class="gemma-output">{schema_review}</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-title">Validation Report</div>', unsafe_allow_html=True)
        if result["validation"]:
            for w in result["validation"]:
                st.warning(w)
        else:
            st.success("No missing-value warnings.")

        if ollama_status.available and result["validation"]:
            st.markdown('<div class="agent-badge">Agent 2 · Validation Analyst</div>', unsafe_allow_html=True)
            if st.button("Explain warnings with Gemma", use_container_width=True):
                sample_rows = run_sql(db_path, f"SELECT * FROM {result['table_name']} LIMIT 5")
                with st.spinner(f"{selected_model} analysing data quality…"):
                    analysis = OllamaGemmaClient(model=selected_model).generate(
                        build_validation_agent_prompt(result["validation"], result["columns"], sample_rows)
                    )
                st.session_state["validation_analysis"] = analysis

        validation_analysis = st.session_state.get("validation_analysis")
        if validation_analysis:
            st.markdown(f'<div class="gemma-output">{validation_analysis}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Database Preview</div>', unsafe_allow_html=True)
    rows = run_sql(db_path, f"SELECT * FROM {result['table_name']} LIMIT 20")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ── ASK TAB ───────────────────────────────────────────────────
with ask_tab:
    col_q, col_opts = st.columns([3, 2], gap="large")

    with col_q:
        st.markdown('<div class="section-title">Question</div>', unsafe_allow_html=True)
        question_choice = st.selectbox("Example questions", EXAMPLE_QUESTIONS, label_visibility="collapsed")
        question = st.text_input("Edit or type your own question", value=question_choice)

    with col_opts:
        st.markdown('<div class="section-title">Options</div>', unsafe_allow_html=True)
        use_gemma = st.checkbox("Generate SQL with Gemma 4 (Agent 3)", value=False)
        generate_summary = st.checkbox("Generate Gemma explanation", value=use_gemma)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    run_clicked = st.button("Run trusted query", type="primary")

    if run_clicked:
        st.session_state["last_answer"] = None
        st.session_state["last_summary"] = None
        st.session_state["repair_count"] = 0

        if use_gemma:
            if not ollama_status.available:
                st.error("Ollama is not reachable. Start Ollama or uncheck Gemma generation.")
                st.stop()
            client = OllamaGemmaClient(model=selected_model)
            try:
                with st.spinner(f"{selected_model} generating SQL…"):
                    answer, repair_count = answer_with_gemma_repair(
                        db_path=db_path,
                        table_name=result["table_name"],
                        columns=result["columns"],
                        question=question,
                        client=client,
                    )
                st.session_state["last_answer"] = answer
                st.session_state["repair_count"] = repair_count
            except UnsafeQueryError as e:
                st.error(f"Blocked unsafe SQL after repair: {e}")
                st.stop()
            except Exception as e:
                st.error(f"Query failed: {e}")
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
            except UnsafeQueryError as e:
                st.error(f"Blocked unsafe SQL: {e}")
                st.stop()
            except Exception as e:
                st.error(f"Query failed: {e}")
                st.stop()

        if generate_summary and ollama_status.available:
            answer = st.session_state["last_answer"]
            with st.spinner(f"{selected_model} explaining the answer…"):
                st.session_state["last_summary"] = OllamaGemmaClient(model=selected_model).generate(
                    build_answer_summary_prompt(
                        question=answer.question,
                        sql=answer.sql,
                        rows=answer.rows,
                        provenance=answer.provenance,
                    )
                )

    answer = st.session_state.get("last_answer")
    if answer:
        repair_count = st.session_state.get("repair_count", 0)
        if repair_count > 0:
            st.success(f"Gemma auto-repaired SQL in {repair_count} step{'s' if repair_count > 1 else ''}.")

        sql_col, _ = st.columns([3, 1])
        with sql_col:
            st.markdown(
                f'<div class="section-title">Generated SQL &nbsp;·&nbsp; '
                f'<span style="color:#1B6CA8;text-transform:none;letter-spacing:0">{answer.source}</span></div>',
                unsafe_allow_html=True,
            )
            st.code(answer.sql, language="sql")

        st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(answer.rows), use_container_width=True)

        summary = st.session_state.get("last_summary")
        if summary:
            st.markdown('<div class="agent-badge">Agent 4 · Answer Explainer</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="gemma-output">{summary}</div>', unsafe_allow_html=True)

# ── EVIDENCE TAB ──────────────────────────────────────────────
with evidence_tab:
    answer = st.session_state.get("last_answer")
    if not answer:
        st.markdown("""
        <div style='text-align:center; padding:48px; color:#94A3B8'>
            <div style='font-size:2.5rem'>🔍</div>
            <div style='font-size:1rem; font-weight:600; margin-top:10px'>Run a query first</div>
            <div style='font-size:0.85rem; margin-top:4px'>Go to the Ask tab, run a trusted query, then come back here.</div>
        </div>
        """, unsafe_allow_html=True)
    elif answer.provenance:
        st.markdown('<div class="section-title">Source Provenance</div>', unsafe_allow_html=True)
        st.caption("Every result row traces back to a specific line in your original CSV file.")
        st.dataframe(pd.DataFrame(answer.provenance), use_container_width=True)
    else:
        st.info("This aggregate query does not map to individual source rows.")
