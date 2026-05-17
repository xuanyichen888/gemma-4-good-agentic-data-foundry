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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ── Page background ── */
.stApp { background: #F5F5F0; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: white;
    border-radius: 12px;
    padding: 20px 22px 18px;
    border: 1px solid #E5E7EB;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: #6B7280 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.7rem !important;
    font-weight: 800 !important;
    color: #111827 !important;
    line-height: 1.2 !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] {
    margin-top: 8px;
}
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: white;
    border-radius: 12px 12px 0 0;
    border: 1px solid #E5E7EB;
    border-bottom: none;
    padding: 0 8px;
    gap: 0;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-weight: 600;
    font-size: 0.85rem;
    color: #6B7280;
    padding: 12px 20px;
    border-bottom: 3px solid transparent;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #16A34A !important;
    border-bottom-color: #16A34A !important;
}
[data-testid="stTabs"] [data-baseweb="tab-panel"] {
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 0 0 12px 12px;
    padding: 28px 24px;
}

/* ── Section labels ── */
.lbl {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #9CA3AF;
    margin-bottom: 10px;
    margin-top: 4px;
}

/* ── Agent chips ── */
.chip {
    display: inline-block;
    background: #FFF7ED;
    border: 1px solid #FED7AA;
    color: #C2410C;
    font-size: 0.7rem;
    font-weight: 700;
    border-radius: 6px;
    padding: 3px 10px;
    margin-bottom: 10px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* ── Gemma output box ── */
.gbox {
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
    border-left: 4px solid #16A34A;
    border-radius: 0 10px 10px 0;
    padding: 16px 20px;
    margin-top: 12px;
    line-height: 1.75;
    font-size: 0.88rem;
    color: #14532D;
    white-space: pre-wrap;
}

/* ── Model status pill ── */
.mpill {
    display: inline-block;
    border-radius: 6px;
    padding: 4px 12px;
    font-size: 0.78rem;
    font-weight: 700;
    margin-top: 6px;
}
.mpill-ok   { background: #DCFCE7; color: #166534; }
.mpill-warn { background: #FEF3C7; color: #92400E; }
.mpill-err  { background: #FEE2E2; color: #991B1B; }

/* ── Sidebar brand header ── */
[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E5E7EB;
}
[data-testid="stSidebar"] hr {
    border-color: #F3F4F6;
    margin: 16px 0;
}

/* ── Sidebar label ── */
.slbl {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #9CA3AF;
    margin-bottom: 8px;
    display: block;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
}

/* ── Code block ── */
[data-testid="stCode"] {
    border-radius: 10px;
}
[data-testid="stCode"] > div {
    border-radius: 10px !important;
    border: 1px solid #E5E7EB !important;
    background: #F9FAFB !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #E5E7EB;
}

/* ── Alerts ── */
[data-testid="stAlert"] { border-radius: 8px !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border-radius: 10px;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
example_path = ROOT / "examples" / "community_intake.csv"
db_path = ROOT / "agentic_data_foundry_demo.sqlite"

with st.sidebar:
    # Brand header
    st.markdown(
        '<div style="'
        'background: linear-gradient(135deg,#14532D,#166534);'
        'border-radius:12px; padding:20px 18px 18px; margin-bottom:20px;'
        'text-align:center;'
        '">'
        '<div style="font-size:2rem; margin-bottom:6px;">🌱</div>'
        '<div style="font-weight:800; font-size:0.95rem; color:#fff; letter-spacing:-0.01em;">'
        'Agentic Data Foundry'
        '</div>'
        '<div style="font-size:0.68rem; color:rgba(255,255,255,0.6); margin-top:4px; letter-spacing:0.06em; text-transform:uppercase;">'
        'Gemma 4 Good Hackathon'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<span class="slbl">Data Source</span>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload your CSV", type=["csv"], label_visibility="collapsed")
    use_example = uploaded is None
    if use_example:
        st.caption("Using synthetic community intake dataset.")
        csv_path = example_path
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix="adf_"))
        csv_path = temp_dir / uploaded.name
        csv_path.write_bytes(uploaded.getvalue())

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
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

    st.divider()

    # Model status
    ollama_status = get_ollama_status()
    st.markdown('<span class="slbl">Local Gemma 4 Model</span>', unsafe_allow_html=True)
    if ollama_status.available:
        model_options = ollama_status.models or [ollama_status.selected_model]
        selected_model = st.selectbox(
            "model", model_options,
            index=model_options.index(ollama_status.selected_model)
            if ollama_status.selected_model in model_options else 0,
            label_visibility="collapsed",
        )
        if is_gemma_model(selected_model):
            st.markdown(f'<span class="mpill mpill-ok">● {selected_model} active</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="mpill mpill-warn">⚠ {selected_model}</span>', unsafe_allow_html=True)
    else:
        selected_model = ollama_status.selected_model
        st.markdown('<span class="mpill mpill-err">✕ Ollama unreachable</span>', unsafe_allow_html=True)
        st.caption("Run `ollama serve` and pull `gemma4:e4b`.")

    st.divider()

    # Agent pipeline
    st.markdown('<span class="slbl">Agent Pipeline</span>', unsafe_allow_html=True)
    for num, name, desc in [
        ("1", "Schema Reviewer", "Audits inferred column types"),
        ("2", "Validation Analyst", "Explains data quality gaps"),
        ("3", "SQL Generator", "NL → SQL with auto-repair loop"),
        ("4", "Answer Explainer", "Summarises results & caveats"),
    ]:
        st.markdown(
            f'<div style="display:flex;align-items:flex-start;gap:10px;padding:7px 0;border-bottom:1px solid #F3F4F6;">'
            f'<div style="min-width:22px;height:22px;border-radius:50%;background:#DCFCE7;color:#166534;'
            f'font-size:0.65rem;font-weight:800;display:flex;align-items:center;justify-content:center;">{num}</div>'
            f'<div style="line-height:1.4;">'
            f'<div style="font-size:0.8rem;font-weight:700;color:#111827;">{name}</div>'
            f'<div style="font-size:0.72rem;color:#6B7280;">{desc}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

# ── Hero ──────────────────────────────────────────────────────
st.markdown(
    '<div style="'
    'background:linear-gradient(135deg,#052E16 0%,#14532D 45%,#166534 100%);'
    'border-radius:16px; padding:44px 52px 40px; margin-bottom:24px; color:white;'
    '">'
    '<div style="font-size:0.7rem;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;'
    'color:rgba(255,255,255,0.55);margin-bottom:16px;">'
    '🌱 &nbsp;Gemma 4 Good Hackathon &nbsp;·&nbsp; Digital Equity &amp; Safety'
    '</div>'
    '<div style="font-size:2.5rem;font-weight:800;line-height:1.15;margin-bottom:16px;letter-spacing:-0.02em;">'
    'Turn messy records into<br>'
    '<span style="color:#86EFAC;">trusted databases</span>'
    '</div>'
    '<div style="font-size:1rem;color:rgba(255,255,255,0.8);line-height:1.7;max-width:580px;margin-bottom:28px;">'
    'Small community organizations track hundreds of families in spreadsheets — '
    'but lack a data engineer. <strong style="color:white;">Agentic Data Foundry</strong> uses '
    'local Gemma&nbsp;4 agents to build a clean, queryable database with source evidence for every answer.'
    '</div>'
    '<div style="display:flex;flex-wrap:wrap;gap:10px;">'
    '<span style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.25);'
    'border-radius:6px;padding:5px 14px;font-size:0.8rem;font-weight:600;">🔒 Data stays local</span>'
    '<span style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.25);'
    'border-radius:6px;padding:5px 14px;font-size:0.8rem;font-weight:600;">🤖 4 Gemma 4 agents</span>'
    '<span style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.25);'
    'border-radius:6px;padding:5px 14px;font-size:0.8rem;font-weight:600;">📎 Row-level provenance</span>'
    '<span style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.25);'
    'border-radius:6px;padding:5px 14px;font-size:0.8rem;font-weight:600;">🛡 SQL safety validation</span>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

# ── Gate ──────────────────────────────────────────────────────
result = st.session_state.get("build_result")

if not result:
    st.markdown(
        '<div style="background:white;border:1px solid #E5E7EB;border-radius:12px;'
        'padding:64px 32px;text-align:center;margin-top:8px;">'
        '<div style="font-size:3rem;margin-bottom:16px;">📂</div>'
        '<div style="font-size:1.15rem;font-weight:700;color:#111827;margin-bottom:8px;">No database yet</div>'
        '<div style="font-size:0.9rem;color:#6B7280;line-height:1.6;">'
        'Upload a CSV or use the synthetic example,<br>'
        'then click <strong>Build Database</strong> in the sidebar.</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# ── Metrics ───────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows imported", result["row_count"])
c2.metric("Columns", len(result["columns"]))
c3.metric("Validation warnings", len(result["validation"]))
c4.metric("Storage", "SQLite · local")

st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────
build_tab, ask_tab, evidence_tab = st.tabs(["🗂  Build & Validate", "💬  Ask", "🔍  Evidence"])

# ─────────────────────────────────────────────────────────────
# BUILD TAB
# ─────────────────────────────────────────────────────────────
with build_tab:
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown('<div class="lbl">Inferred Schema</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(result["columns"]), use_container_width=True, height=220)

        if ollama_status.available:
            st.markdown(
                '<span class="chip">⚡ Agent 1 · Schema Reviewer</span>',
                unsafe_allow_html=True,
            )
            if st.button("Run schema review", use_container_width=True):
                sample_rows = run_sql(db_path, f"SELECT * FROM {result['table_name']} LIMIT 5")
                with st.spinner(f"{selected_model} reviewing schema…"):
                    review = OllamaGemmaClient(model=selected_model).generate(
                        build_schema_review_prompt(result["columns"], sample_rows)
                    )
                st.session_state["schema_review"] = review

        if st.session_state.get("schema_review"):
            st.markdown(
                f'<div class="gbox">{st.session_state["schema_review"]}</div>',
                unsafe_allow_html=True,
            )

    with right:
        st.markdown('<div class="lbl">Validation Report</div>', unsafe_allow_html=True)
        if result["validation"]:
            for w in result["validation"]:
                st.warning(w)
        else:
            st.success("No missing-value warnings detected.")

        if ollama_status.available and result["validation"]:
            st.markdown(
                '<span class="chip">⚡ Agent 2 · Validation Analyst</span>',
                unsafe_allow_html=True,
            )
            if st.button("Explain warnings with Gemma", use_container_width=True):
                sample_rows = run_sql(db_path, f"SELECT * FROM {result['table_name']} LIMIT 5")
                with st.spinner(f"{selected_model} analysing data quality…"):
                    analysis = OllamaGemmaClient(model=selected_model).generate(
                        build_validation_agent_prompt(result["validation"], result["columns"], sample_rows)
                    )
                st.session_state["validation_analysis"] = analysis

        if st.session_state.get("validation_analysis"):
            st.markdown(
                f'<div class="gbox">{st.session_state["validation_analysis"]}</div>',
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown('<div class="lbl">Database Preview — first 20 rows</div>', unsafe_allow_html=True)
    rows = run_sql(db_path, f"SELECT * FROM {result['table_name']} LIMIT 20")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ─────────────────────────────────────────────────────────────
# ASK TAB
# ─────────────────────────────────────────────────────────────
with ask_tab:
    q_col, opt_col = st.columns([3, 2], gap="large")
    with q_col:
        st.markdown('<div class="lbl">Your Question</div>', unsafe_allow_html=True)
        choice = st.selectbox("Example", EXAMPLE_QUESTIONS, label_visibility="collapsed")
        question = st.text_input("Edit or write your own question", value=choice)
    with opt_col:
        st.markdown('<div class="lbl">Options</div>', unsafe_allow_html=True)
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

        st.divider()

        src_color = "#16A34A" if "gemma" in answer.source.lower() else "#6B7280"
        st.markdown(
            f'<div class="lbl" style="margin-top:4px;">Generated SQL &nbsp;'
            f'<span style="color:{src_color};text-transform:none;letter-spacing:0;'
            f'font-weight:700;font-size:0.75rem;">· {answer.source}</span></div>',
            unsafe_allow_html=True,
        )
        st.code(answer.sql, language="sql")

        st.markdown('<div class="lbl" style="margin-top:16px;">Results</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(answer.rows), use_container_width=True)

        if st.session_state.get("last_summary"):
            st.markdown(
                '<span class="chip">⚡ Agent 4 · Answer Explainer</span>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="gbox">{st.session_state["last_summary"]}</div>',
                unsafe_allow_html=True,
            )

# ─────────────────────────────────────────────────────────────
# EVIDENCE TAB
# ─────────────────────────────────────────────────────────────
with evidence_tab:
    answer = st.session_state.get("last_answer")
    if not answer:
        st.markdown(
            '<div style="text-align:center;padding:56px 20px;">'
            '<div style="font-size:3rem;margin-bottom:16px;">🔍</div>'
            '<div style="font-size:1.1rem;font-weight:700;color:#111827;margin-bottom:8px;">No query run yet</div>'
            '<div style="font-size:0.88rem;color:#6B7280;line-height:1.6;">'
            'Run a trusted query in the Ask tab,<br>then come back here to inspect the source evidence.'
            '</div></div>',
            unsafe_allow_html=True,
        )
    elif answer.provenance:
        st.markdown(
            '<div class="lbl">Source Provenance</div>'
            '<div style="font-size:0.85rem;color:#6B7280;margin-bottom:14px;">'
            'Each row traces back to a specific line number in your original CSV file.'
            '</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(pd.DataFrame(answer.provenance), use_container_width=True)
    else:
        st.info("This aggregate query does not map to individual source rows.")
