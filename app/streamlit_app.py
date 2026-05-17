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
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Background */
.stApp { background: #FFFBF5; }

/* ── Hero badge ── */
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #DCFCE7;
    color: #166534;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    border-radius: 999px;
    padding: 5px 14px;
    margin-bottom: 8px;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: white;
    border-radius: 14px;
    padding: 20px 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.04);
    border-top: 3px solid #16A34A;
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-weight: 600;
    font-size: 0.88rem;
    color: #6B7280;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #16A34A !important;
}

/* ── Cards ── */
.section-card {
    background: white;
    border-radius: 16px;
    padding: 24px 26px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.04);
    margin-bottom: 20px;
}
.section-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #9CA3AF;
    margin-bottom: 14px;
}

/* ── Agent badge ── */
.agent-chip {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: #FFF7ED;
    border: 1px solid #FED7AA;
    color: #C2410C;
    font-size: 0.72rem;
    font-weight: 700;
    border-radius: 999px;
    padding: 4px 12px;
    margin-bottom: 12px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.agent-chip .dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #F97316;
}

/* ── Gemma output ── */
.gemma-box {
    background: linear-gradient(135deg, #F0FDF4, #ECFDF5);
    border: 1px solid #BBF7D0;
    border-left: 4px solid #16A34A;
    border-radius: 0 12px 12px 0;
    padding: 18px 20px;
    margin-top: 14px;
    line-height: 1.75;
    font-size: 0.9rem;
    color: #1C3829;
}
.gemma-box strong { color: #15803D; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: white;
    border-right: 1px solid #E5E7EB;
}
.sidebar-section {
    background: #F9FAFB;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 14px;
}
.sidebar-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #9CA3AF;
    margin-bottom: 10px;
}
.model-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    border-radius: 999px;
    padding: 5px 14px;
    font-size: 0.8rem;
    font-weight: 700;
    margin-top: 8px;
}
.model-ok   { background: #DCFCE7; color: #166534; }
.model-warn { background: #FEF3C7; color: #92400E; }
.model-err  { background: #FEE2E2; color: #991B1B; }

/* ── Buttons ── */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
}

/* ── SQL block ── */
[data-testid="stCode"] > div {
    border-radius: 12px !important;
    border: 1px solid #D1FAE5 !important;
}

/* ── Empty states ── */
.empty-state {
    text-align: center;
    padding: 56px 20px;
}
.empty-icon {
    font-size: 3.5rem;
    margin-bottom: 14px;
    display: block;
}
.empty-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #374151;
    margin-bottom: 6px;
}
.empty-sub {
    font-size: 0.88rem;
    color: #9CA3AF;
}

/* ── Dataframes ── */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* ── Warnings/success ── */
[data-testid="stAlert"] { border-radius: 10px !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
example_path = ROOT / "examples" / "community_intake.csv"
db_path = ROOT / "agentic_data_foundry_demo.sqlite"

with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 20px; text-align: center;">
      <div style="font-size:1.6rem; margin-bottom:4px">🌱</div>
      <div style="font-weight:800; font-size:1rem; color:#14532D">Agentic Data Foundry</div>
      <div style="font-size:0.72rem; color:#9CA3AF; margin-top:2px">Gemma 4 Good Hackathon</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Data Source</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload your CSV", type=["csv"], label_visibility="collapsed")
    use_example = uploaded is None
    if use_example:
        st.caption("Using synthetic community intake dataset.")
        csv_path = example_path
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix="adf_"))
        csv_path = temp_dir / uploaded.name
        csv_path.write_bytes(uploaded.getvalue())

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    if st.button("🗄️  Build Database", type="primary", use_container_width=True):
        with st.spinner("Importing records and building provenance table…"):
            result = build_sqlite_from_csv(csv_path, db_path, table_name="community_intake")
        st.session_state.update({
            "build_result": result,
            "last_answer": None,
            "schema_review": None,
            "validation_analysis": None,
        })
        st.success(f"✓ {result['row_count']} rows imported.")

    st.markdown("---")

    # Model status
    ollama_status = get_ollama_status()
    st.markdown('<div class="sidebar-label">Local Gemma 4 Model</div>', unsafe_allow_html=True)
    if ollama_status.available:
        model_options = ollama_status.models or [ollama_status.selected_model]
        selected_model = st.selectbox(
            "model", model_options,
            index=model_options.index(ollama_status.selected_model)
            if ollama_status.selected_model in model_options else 0,
            label_visibility="collapsed",
        )
        if is_gemma_model(selected_model):
            st.markdown(f'<div class="model-pill model-ok">● {selected_model} active</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="model-pill model-warn">⚠ {selected_model}</div>', unsafe_allow_html=True)
    else:
        selected_model = ollama_status.selected_model
        st.markdown('<div class="model-pill model-err">✕ Ollama unreachable</div>', unsafe_allow_html=True)
        st.caption("Run `ollama serve` and pull `gemma4:e4b`.")

    st.markdown("---")

    # Agent pipeline
    st.markdown('<div class="sidebar-label">Agent Pipeline</div>', unsafe_allow_html=True)
    st.markdown(
        "**1** Schema Reviewer — audits column types  \n"
        "**2** Validation Analyst — explains data gaps  \n"
        "**3** SQL Generator — NL → SQL + auto-repair  \n"
        "**4** Answer Explainer — summarises results"
    )

# ── Hero ──────────────────────────────────────────────────────
st.markdown(
    '<div class="hero-badge" style="display:inline-block;margin-bottom:10px">'
    '🌱 Gemma 4 Good Hackathon · Digital Equity &amp; Safety'
    '</div>',
    unsafe_allow_html=True,
)
st.markdown("## Turn messy records into trusted databases")
st.markdown(
    "Small community organizations track hundreds of families in spreadsheets — "
    "but lack a data engineer. **Agentic Data Foundry** uses local Gemma 4 agents "
    "to build a clean, queryable database with source evidence for every answer."
)
st.markdown("🔒 Data stays local &nbsp;·&nbsp; 🤖 4 Gemma 4 agents &nbsp;·&nbsp; 📎 Row-level provenance &nbsp;·&nbsp; 🛡 SQL safety validation", unsafe_allow_html=True)
st.markdown("---")

# ── Gate ──────────────────────────────────────────────────────
result = st.session_state.get("build_result")

if not result:
    st.markdown("""
    <div class="empty-state">
      <span class="empty-icon">📂</span>
      <div class="empty-title">No database yet</div>
      <div class="empty-sub">Upload a CSV or use the synthetic example,<br>then click <b>Build Database</b> in the sidebar.</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Metrics ───────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows imported", result["row_count"])
c2.metric("Columns", len(result["columns"]))
c3.metric("Validation warnings", len(result["validation"]))
c4.metric("Storage", "SQLite · local")

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────
build_tab, ask_tab, evidence_tab = st.tabs(["🗂  Build & Validate", "💬  Ask", "🔍  Evidence"])

# ─────────────────────────────────────────────────────────────
# BUILD TAB
# ─────────────────────────────────────────────────────────────
with build_tab:
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown('<div class="section-label">Inferred Schema</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(result["columns"]), use_container_width=True, height=220)

        if ollama_status.available:
            st.markdown('<div class="agent-chip"><span class="dot"></span>Agent 1 · Schema Reviewer</div>', unsafe_allow_html=True)
            if st.button("Run schema review", use_container_width=True):
                sample_rows = run_sql(db_path, f"SELECT * FROM {result['table_name']} LIMIT 5")
                with st.spinner(f"{selected_model} reviewing schema…"):
                    review = OllamaGemmaClient(model=selected_model).generate(
                        build_schema_review_prompt(result["columns"], sample_rows)
                    )
                st.session_state["schema_review"] = review

        if st.session_state.get("schema_review"):
            st.markdown(f'<div class="gemma-box">{st.session_state["schema_review"]}</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-label">Validation Report</div>', unsafe_allow_html=True)
        if result["validation"]:
            for w in result["validation"]:
                st.warning(w)
        else:
            st.success("No missing-value warnings detected.")

        if ollama_status.available and result["validation"]:
            st.markdown('<div class="agent-chip"><span class="dot"></span>Agent 2 · Validation Analyst</div>', unsafe_allow_html=True)
            if st.button("Explain warnings with Gemma", use_container_width=True):
                sample_rows = run_sql(db_path, f"SELECT * FROM {result['table_name']} LIMIT 5")
                with st.spinner(f"{selected_model} analysing data quality…"):
                    analysis = OllamaGemmaClient(model=selected_model).generate(
                        build_validation_agent_prompt(result["validation"], result["columns"], sample_rows)
                    )
                st.session_state["validation_analysis"] = analysis

        if st.session_state.get("validation_analysis"):
            st.markdown(f'<div class="gemma-box">{st.session_state["validation_analysis"]}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-label">Database Preview — first 20 rows</div>', unsafe_allow_html=True)
    rows = run_sql(db_path, f"SELECT * FROM {result['table_name']} LIMIT 20")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ─────────────────────────────────────────────────────────────
# ASK TAB
# ─────────────────────────────────────────────────────────────
with ask_tab:
    q_col, opt_col = st.columns([3, 2], gap="large")
    with q_col:
        st.markdown('<div class="section-label">Your Question</div>', unsafe_allow_html=True)
        choice = st.selectbox("Example", EXAMPLE_QUESTIONS, label_visibility="collapsed")
        question = st.text_input("Edit or write your own question", value=choice)
    with opt_col:
        st.markdown('<div class="section-label">Options</div>', unsafe_allow_html=True)
        use_gemma = st.checkbox("Generate SQL with Gemma 4 (Agent 3)", value=False)
        generate_summary = st.checkbox("Generate Gemma explanation (Agent 4)", value=use_gemma)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    if st.button("🔍  Run Trusted Query", type="primary"):
        st.session_state.update({"last_answer": None, "last_summary": None, "repair_count": 0})

        if use_gemma:
            if not ollama_status.available:
                st.error("Ollama unreachable. Start Ollama or uncheck Gemma generation.")
                st.stop()
            client = OllamaGemmaClient(model=selected_model)
            try:
                with st.spinner(f"{selected_model} generating SQL with auto-repair…"):
                    answer, repair_count = answer_with_gemma_repair(
                        db_path=db_path, table_name=result["table_name"],
                        columns=result["columns"], question=question, client=client,
                    )
                st.session_state.update({"last_answer": answer, "repair_count": repair_count})
            except UnsafeQueryError as e:
                st.error(f"Blocked unsafe SQL after repair: {e}"); st.stop()
            except Exception as e:
                st.error(f"Query failed: {e}"); st.stop()
        else:
            try:
                answer = answer_question(db_path=db_path, table_name=result["table_name"],
                                         question=question, generated_sql=None)
                st.session_state["last_answer"] = answer
            except UnsafeQueryError as e:
                st.error(f"Blocked unsafe SQL: {e}"); st.stop()
            except Exception as e:
                st.error(f"Query failed: {e}"); st.stop()

        if generate_summary and ollama_status.available:
            ans = st.session_state["last_answer"]
            with st.spinner(f"{selected_model} explaining the answer…"):
                st.session_state["last_summary"] = OllamaGemmaClient(model=selected_model).generate(
                    build_answer_summary_prompt(ans.question, ans.sql, ans.rows, ans.provenance)
                )

    answer = st.session_state.get("last_answer")
    if answer:
        if st.session_state.get("repair_count", 0) > 0:
            rc = st.session_state["repair_count"]
            st.success(f"Gemma auto-repaired SQL in {rc} step{'s' if rc > 1 else ''}.")

        st.markdown(
            f'<div class="section-label" style="margin-top:16px">Generated SQL &nbsp;·&nbsp; '
            f'<span style="color:#16A34A;text-transform:none;letter-spacing:0;font-weight:600">{answer.source}</span></div>',
            unsafe_allow_html=True,
        )
        st.code(answer.sql, language="sql")
        st.markdown('<div class="section-label">Results</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(answer.rows), use_container_width=True)

        if st.session_state.get("last_summary"):
            st.markdown('<div class="agent-chip"><span class="dot"></span>Agent 4 · Answer Explainer</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="gemma-box">{st.session_state["last_summary"]}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# EVIDENCE TAB
# ─────────────────────────────────────────────────────────────
with evidence_tab:
    answer = st.session_state.get("last_answer")
    if not answer:
        st.markdown("""
        <div class="empty-state">
          <span class="empty-icon">🔍</span>
          <div class="empty-title">No query run yet</div>
          <div class="empty-sub">Run a trusted query in the Ask tab,<br>then come back here to inspect the source evidence.</div>
        </div>
        """, unsafe_allow_html=True)
    elif answer.provenance:
        st.markdown('<div class="section-label">Source Provenance</div>', unsafe_allow_html=True)
        st.caption("Each row traces back to a specific line number in your original CSV file.")
        st.dataframe(pd.DataFrame(answer.provenance), use_container_width=True)
    else:
        st.info("This aggregate query does not map to individual source rows.")
