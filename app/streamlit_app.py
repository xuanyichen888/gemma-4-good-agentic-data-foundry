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

/* ── Animations ── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes floatY {
    0%, 100% { transform: translateY(0); }
    50%       { transform: translateY(-10px); }
}
@keyframes spin-slow {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}
@keyframes pulse-ring {
    0%   { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(45,153,102,0.4); }
    70%  { transform: scale(1);    box-shadow: 0 0 0 10px rgba(45,153,102,0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(45,153,102,0); }
}
@keyframes shimmer {
    0%   { background-position: -400px 0; }
    100% { background-position: 400px 0; }
}

/* ── Hero ── */
.hero-wrap {
    background: linear-gradient(135deg, #F0FDF4 0%, #FFF7ED 50%, #ECFDF5 100%);
    border: 1px solid #D1FAE5;
    border-radius: 20px;
    padding: 0;
    margin-bottom: 28px;
    overflow: hidden;
    display: flex;
    align-items: stretch;
    animation: fadeInUp 0.6s ease both;
}
.hero-left {
    flex: 1;
    padding: 40px 44px;
}
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
    margin-bottom: 18px;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #14532D;
    line-height: 1.2;
    margin: 0 0 12px;
}
.hero-title span { color: #F97316; }
.hero-sub {
    font-size: 1rem;
    color: #4B5563;
    line-height: 1.65;
    max-width: 480px;
    margin: 0 0 24px;
}
.hero-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}
.hero-pill {
    background: white;
    border: 1px solid #D1FAE5;
    border-radius: 999px;
    padding: 6px 16px;
    font-size: 0.82rem;
    font-weight: 600;
    color: #1F7A4A;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}
.hero-right {
    width: 320px;
    background: linear-gradient(135deg, #16A34A 0%, #15803D 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 30px;
    flex-shrink: 0;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: white;
    border-radius: 14px;
    padding: 20px 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.04);
    border-top: 3px solid #16A34A;
    animation: fadeInUp 0.5s ease both;
}

/* ── Tabs ── */
[data-testid="stTabs"] { animation: fadeInUp 0.5s 0.1s ease both; }
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: white;
    border-radius: 12px;
    padding: 4px;
    gap: 2px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    border-radius: 10px !important;
    font-weight: 600;
    font-size: 0.88rem;
    padding: 10px 20px;
    color: #6B7280;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: #16A34A !important;
    color: white !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { display: none; }

/* ── Cards ── */
.section-card {
    background: white;
    border-radius: 16px;
    padding: 24px 26px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.04);
    margin-bottom: 20px;
    animation: fadeInUp 0.5s ease both;
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
    animation: pulse-ring 2s infinite;
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
    animation: fadeInUp 0.4s ease both;
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
[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, #16A34A, #15803D) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 11px 28px !important;
    box-shadow: 0 4px 12px rgba(22,163,74,0.3) !important;
    transition: all 0.2s !important;
}
[data-testid="stButton"] button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(22,163,74,0.4) !important;
}
[data-testid="stButton"] button:not([kind="primary"]) {
    border-radius: 10px !important;
    font-weight: 600 !important;
    border-color: #D1FAE5 !important;
    color: #15803D !important;
    transition: all 0.2s !important;
}
[data-testid="stButton"] button:not([kind="primary"]):hover {
    background: #F0FDF4 !important;
}

/* ── SQL block ── */
[data-testid="stCode"] > div {
    border-radius: 12px !important;
    border: 1px solid #D1FAE5 !important;
}

/* ── Agent pipeline in sidebar ── */
.pipeline-step {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 8px 0;
}
.pipeline-num {
    width: 24px; height: 24px;
    border-radius: 50%;
    background: linear-gradient(135deg, #16A34A, #15803D);
    color: white;
    font-size: 0.7rem;
    font-weight: 800;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.pipeline-text {
    font-size: 0.78rem;
    color: #374151;
    font-weight: 500;
    line-height: 1.5;
    padding-top: 3px;
}
.pipeline-connector {
    width: 2px;
    height: 14px;
    background: #D1FAE5;
    margin-left: 11px;
}

/* ── Empty states ── */
.empty-state {
    text-align: center;
    padding: 56px 20px;
    animation: fadeInUp 0.5s ease both;
}
.empty-icon {
    font-size: 3.5rem;
    margin-bottom: 14px;
    display: block;
    animation: floatY 3s ease-in-out infinite;
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
    st.markdown("""
    <div>
      <div class="pipeline-step">
        <div class="pipeline-num">1</div>
        <div class="pipeline-text"><b>Schema Reviewer</b><br>Audits inferred column types</div>
      </div>
      <div class="pipeline-connector"></div>
      <div class="pipeline-step">
        <div class="pipeline-num">2</div>
        <div class="pipeline-text"><b>Validation Analyst</b><br>Explains data quality gaps</div>
      </div>
      <div class="pipeline-connector"></div>
      <div class="pipeline-step">
        <div class="pipeline-num">3</div>
        <div class="pipeline-text"><b>SQL Generator</b><br>NL → SQL with auto-repair</div>
      </div>
      <div class="pipeline-connector"></div>
      <div class="pipeline-step">
        <div class="pipeline-num">4</div>
        <div class="pipeline-text"><b>Answer Explainer</b><br>Summarises results + caveats</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-left">
    <div class="hero-badge">🌱 Gemma 4 Good Hackathon · Digital Equity & Safety</div>
    <div class="hero-title">Turn messy records into<br><span>trusted databases</span></div>
    <div class="hero-sub">
      Small community organizations track hundreds of families in spreadsheets—
      but lack a data engineer. Agentic Data Foundry uses local Gemma&nbsp;4 agents
      to build a clean, queryable database with source evidence for every answer.
    </div>
    <div class="hero-pills">
      <span class="hero-pill">🔒 Data stays local</span>
      <span class="hero-pill">🤖 4 Gemma 4 agents</span>
      <span class="hero-pill">📎 Row-level provenance</span>
      <span class="hero-pill">🛡 SQL safety validation</span>
    </div>
  </div>
  <div class="hero-right">
    <svg width="240" height="210" viewBox="0 0 240 210" fill="none" xmlns="http://www.w3.org/2000/svg">
      <!-- Floating papers (messy data) -->
      <rect x="8" y="30" width="52" height="38" rx="6" fill="white" opacity="0.25" transform="rotate(-12 8 30)"/>
      <rect x="18" y="34" width="32" height="4" rx="2" fill="white" opacity="0.5" transform="rotate(-12 8 30)"/>
      <rect x="18" y="42" width="22" height="4" rx="2" fill="white" opacity="0.4" transform="rotate(-12 8 30)"/>
      <rect x="18" y="50" width="28" height="4" rx="2" fill="white" opacity="0.3" transform="rotate(-12 8 30)"/>

      <rect x="2" y="80" width="52" height="38" rx="6" fill="white" opacity="0.2" transform="rotate(8 2 80)"/>
      <rect x="12" y="84" width="32" height="4" rx="2" fill="white" opacity="0.4" transform="rotate(8 2 80)"/>
      <rect x="12" y="92" width="18" height="4" rx="2" fill="white" opacity="0.3" transform="rotate(8 2 80)"/>

      <!-- Arrow -->
      <path d="M 75 105 L 105 105" stroke="white" stroke-width="2.5" stroke-dasharray="4 3" opacity="0.6"/>
      <polygon points="105,100 115,105 105,110" fill="white" opacity="0.8"/>

      <!-- Database cylinder -->
      <ellipse cx="158" cy="78" rx="46" ry="14" fill="white" opacity="0.25"/>
      <rect x="112" y="78" width="92" height="62" fill="white" opacity="0.15"/>
      <ellipse cx="158" cy="140" rx="46" ry="14" fill="white" opacity="0.2"/>
      <ellipse cx="158" cy="78" rx="46" ry="14" fill="white" opacity="0.9"/>
      <rect x="112" y="78" width="92" height="62" fill="white" opacity="0.15"/>
      <ellipse cx="158" cy="102" rx="46" ry="14" fill="white" opacity="0.25"/>
      <ellipse cx="158" cy="126" rx="46" ry="14" fill="white" opacity="0.3"/>
      <ellipse cx="158" cy="140" rx="46" ry="14" fill="white" opacity="0.85"/>
      <!-- DB stripes -->
      <rect x="126" y="84" width="64" height="3" rx="1.5" fill="#16A34A" opacity="0.4"/>
      <rect x="126" y="92" width="48" height="3" rx="1.5" fill="#16A34A" opacity="0.3"/>
      <rect x="126" y="100" width="56" height="3" rx="1.5" fill="#16A34A" opacity="0.25"/>

      <!-- Checkmark on top -->
      <circle cx="158" cy="78" r="18" fill="#16A34A"/>
      <polyline points="149,78 155,84 167,72" stroke="white" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>

      <!-- People icons -->
      <circle cx="34" cy="168" r="12" fill="white" opacity="0.9"/>
      <rect x="22" y="182" width="24" height="20" rx="8" fill="white" opacity="0.7"/>

      <circle cx="72" cy="165" r="10" fill="white" opacity="0.8"/>
      <rect x="61" y="177" width="22" height="18" rx="7" fill="white" opacity="0.6"/>

      <circle cx="108" cy="170" r="11" fill="white" opacity="0.85"/>
      <rect x="97" y="183" width="22" height="18" rx="7" fill="white" opacity="0.65"/>

      <!-- Connecting lines from people to DB -->
      <path d="M 34 156 Q 80 145 112 130" stroke="white" stroke-width="1.5" opacity="0.35" stroke-dasharray="3 3"/>
      <path d="M 72 155 Q 110 145 112 130" stroke="white" stroke-width="1.5" opacity="0.3" stroke-dasharray="3 3"/>
    </svg>
  </div>
</div>
""", unsafe_allow_html=True)

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
        st.markdown('<div class="section-card"><div class="section-label">Inferred Schema</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(result["columns"]), use_container_width=True, height=220)
        st.markdown('</div>', unsafe_allow_html=True)

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
        st.markdown('<div class="section-card"><div class="section-label">Validation Report</div>', unsafe_allow_html=True)
        if result["validation"]:
            for w in result["validation"]:
                st.warning(w)
        else:
            st.success("No missing-value warnings detected.")
        st.markdown('</div>', unsafe_allow_html=True)

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
