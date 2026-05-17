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

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,400;0,500;0,600;0,700;0,800;0,900;1,400&display=swap');

html, body, [class*="css"], p, span, div, label, input, textarea, button {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ── Page ── */
.stApp { background: #F0F2F0; }
.block-container { padding-top: 24px !important; padding-bottom: 40px !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: #FFFFFF !important; border-right: 1px solid #E2E8E2; }
[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }
[data-testid="stSidebar"] hr { border-color: #F0F0F0; margin: 14px 0; }

/* ── Buttons ── */
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    letter-spacing: -0.01em !important;
}

/* ── Tab list ── */
[data-baseweb="tab-list"] {
    background: #FFFFFF !important;
    border-radius: 10px 10px 0 0 !important;
    border-bottom: 2px solid #E5E7EB !important;
    gap: 0 !important;
    padding: 0 4px !important;
}
[data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 13.5px !important;
    font-weight: 600 !important;
    color: #6B7280 !important;
    padding: 12px 20px !important;
    border-bottom: 3px solid transparent !important;
    margin-bottom: -2px !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    color: #15803D !important;
    border-bottom-color: #15803D !important;
}
[data-baseweb="tab-panel"] {
    background: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
    padding: 28px 24px !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] { border-radius: 8px !important; font-size: 14px !important; }

/* ── Code ── */
[data-testid="stCode"] > div {
    border-radius: 8px !important;
    border: 1px solid #E5E7EB !important;
    font-size: 13px !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/* ── File uploader ── */
[data-testid="stFileUploader"] section {
    padding: 12px 16px !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploader"] section p {
    font-size: 13px !important;
}

/* ── Selectbox / inputs ── */
[data-baseweb="select"] {
    font-size: 14px !important;
}

/* ── Checkbox ── */
[data-testid="stCheckbox"] label {
    font-size: 14px !important;
    font-weight: 500 !important;
}

/* ── Caption ── */
[data-testid="stCaptionContainer"] p {
    font-size: 12px !important;
    color: #9CA3AF !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helper: custom metric card ─────────────────────────────────
def metric_card(label: str, value: str, color: str = "#15803D") -> str:
    return (
        f'<div style="background:white;border:1px solid #E5E7EB;border-radius:10px;'
        f'padding:18px 20px 16px;height:100%;">'
        f'<div style="font-size:11px;font-weight:700;letter-spacing:0.09em;'
        f'text-transform:uppercase;color:#9CA3AF;margin-bottom:10px;line-height:1;">'
        f'{label}</div>'
        f'<div style="font-size:28px;font-weight:800;color:#111827;line-height:1;'
        f'letter-spacing:-0.03em;">{value}</div>'
        f'<div style="margin-top:8px;height:3px;border-radius:2px;'
        f'background:{color};width:32px;"></div>'
        f'</div>'
    )


# ── Helper: section heading ────────────────────────────────────
def section_heading(text: str, tight: bool = False) -> None:
    mb = "10px" if tight else "14px"
    mt = "4px" if tight else "0"
    st.markdown(
        f'<div style="font-size:11px;font-weight:700;letter-spacing:0.09em;'
        f'text-transform:uppercase;color:#9CA3AF;margin-bottom:{mb};margin-top:{mt};">'
        f'{text}</div>',
        unsafe_allow_html=True,
    )


# ── Sidebar ───────────────────────────────────────────────────
example_path = ROOT / "examples" / "community_intake.csv"
db_path = ROOT / "agentic_data_foundry_demo.sqlite"

with st.sidebar:
    # Brand header
    st.markdown(
        '<div style="background:linear-gradient(160deg,#0F4C2A,#186E3A);'
        'padding:22px 20px 20px;margin:-1px -1px 0;">'
        '<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">'
        '<div style="background:rgba(255,255,255,0.15);border-radius:10px;'
        'width:36px;height:36px;display:flex;align-items:center;justify-content:center;'
        'font-size:18px;">🌱</div>'
        '<div>'
        '<div style="font-weight:800;font-size:14px;color:#fff;letter-spacing:-0.02em;line-height:1.2;">'
        'Agentic Data Foundry</div>'
        '<div style="font-size:10px;color:rgba(255,255,255,0.55);letter-spacing:0.08em;'
        'text-transform:uppercase;margin-top:2px;">Gemma 4 Good</div>'
        '</div></div>'
        '<div style="height:1px;background:rgba(255,255,255,0.12);"></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    section_heading("Data Source")
    uploaded = st.file_uploader("Upload your CSV", type=["csv"], label_visibility="collapsed")
    use_example = uploaded is None
    if use_example:
        st.caption("Using synthetic community intake dataset.")
        csv_path = example_path
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix="adf_"))
        csv_path = temp_dir / uploaded.name
        csv_path.write_bytes(uploaded.getvalue())

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
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
    section_heading("Gemma 4 Model")
    if ollama_status.available:
        model_options = ollama_status.models or [ollama_status.selected_model]
        selected_model = st.selectbox(
            "model", model_options,
            index=model_options.index(ollama_status.selected_model)
            if ollama_status.selected_model in model_options else 0,
            label_visibility="collapsed",
        )
        pill_bg = "#DCFCE7" if is_gemma_model(selected_model) else "#FEF3C7"
        pill_color = "#166534" if is_gemma_model(selected_model) else "#92400E"
        pill_icon = "●" if is_gemma_model(selected_model) else "⚠"
        st.markdown(
            f'<div style="display:inline-block;background:{pill_bg};color:{pill_color};'
            f'border-radius:6px;padding:4px 10px;font-size:12px;font-weight:700;'
            f'margin-top:6px;">{pill_icon} {selected_model}</div>',
            unsafe_allow_html=True,
        )
    else:
        selected_model = ollama_status.selected_model
        st.markdown(
            '<div style="display:inline-block;background:#FEE2E2;color:#991B1B;'
            'border-radius:6px;padding:4px 10px;font-size:12px;font-weight:700;">'
            '✕ Ollama unreachable</div>',
            unsafe_allow_html=True,
        )
        st.caption("Run `ollama serve` and pull `gemma4:e4b`.")

    st.divider()

    # Agent pipeline
    section_heading("Agent Pipeline")
    AGENTS = [
        ("1", "#DCFCE7", "#166534", "Schema Reviewer", "Audits column types, flags type risks"),
        ("2", "#FEF9C3", "#854D0E", "Validation Analyst", "Explains missing-field warnings"),
        ("3", "#DBEAFE", "#1E40AF", "SQL Generator", "NL → SQL with auto-repair loop"),
        ("4", "#F3E8FF", "#6B21A8", "Answer Explainer", "Summarises results & caveats"),
    ]
    for num, bg, fg, name, desc in AGENTS:
        st.markdown(
            f'<div style="display:flex;gap:10px;align-items:flex-start;padding:8px 0;'
            f'border-bottom:1px solid #F3F4F6;">'
            f'<div style="min-width:20px;height:20px;border-radius:50%;background:{bg};color:{fg};'
            f'font-size:10px;font-weight:800;display:flex;align-items:center;justify-content:center;'
            f'flex-shrink:0;margin-top:1px;">{num}</div>'
            f'<div>'
            f'<div style="font-size:12px;font-weight:700;color:#111827;line-height:1.3;">{name}</div>'
            f'<div style="font-size:11px;color:#6B7280;margin-top:2px;line-height:1.4;">{desc}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

# ── Hero ──────────────────────────────────────────────────────
st.markdown(
    '<div style="background:linear-gradient(135deg,#0B3D20 0%,#145A32 50%,#1A7A40 100%);'
    'border-radius:14px;padding:32px 44px 30px;margin-bottom:20px;">'

    # Eyebrow
    '<div style="display:flex;align-items:center;gap:8px;margin-bottom:14px;">'
    '<div style="background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.2);'
    'border-radius:999px;padding:3px 12px 4px;font-size:11px;font-weight:700;'
    'letter-spacing:0.08em;text-transform:uppercase;color:rgba(255,255,255,0.75);">'
    '🌱 &nbsp;Gemma 4 Good Hackathon</div>'
    '<div style="width:4px;height:4px;border-radius:50%;background:rgba(255,255,255,0.3);"></div>'
    '<div style="font-size:11px;font-weight:600;color:rgba(255,255,255,0.5);">Digital Equity &amp; Safety</div>'
    '</div>'

    # Title
    '<div style="font-size:34px;font-weight:900;line-height:1.15;letter-spacing:-0.03em;'
    'color:#fff;margin-bottom:12px;">'
    'Turn messy records into&nbsp;<span style="color:#6EE7A0;">trusted databases</span>'
    '</div>'

    # Subtitle
    '<div style="font-size:14.5px;color:rgba(255,255,255,0.72);line-height:1.65;'
    'max-width:560px;margin-bottom:22px;font-weight:400;">'
    'Small nonprofits track hundreds of families in spreadsheets — but lack a data engineer. '
    '<strong style="color:rgba(255,255,255,0.92);font-weight:600;">Agentic Data Foundry</strong> '
    'uses local Gemma&nbsp;4 agents to build a clean, queryable database with source evidence for every answer.'
    '</div>'

    # Pills
    '<div style="display:flex;flex-wrap:wrap;gap:8px;">'
    '<div style="background:rgba(255,255,255,0.1);border:1px solid rgba(255,255,255,0.2);'
    'border-radius:6px;padding:5px 13px;font-size:12px;font-weight:600;color:rgba(255,255,255,0.88);">🔒 Local-only inference</div>'
    '<div style="background:rgba(255,255,255,0.1);border:1px solid rgba(255,255,255,0.2);'
    'border-radius:6px;padding:5px 13px;font-size:12px;font-weight:600;color:rgba(255,255,255,0.88);">🤖 4 Gemma 4 agents</div>'
    '<div style="background:rgba(255,255,255,0.1);border:1px solid rgba(255,255,255,0.2);'
    'border-radius:6px;padding:5px 13px;font-size:12px;font-weight:600;color:rgba(255,255,255,0.88);">📎 Row-level provenance</div>'
    '<div style="background:rgba(255,255,255,0.1);border:1px solid rgba(255,255,255,0.2);'
    'border-radius:6px;padding:5px 13px;font-size:12px;font-weight:600;color:rgba(255,255,255,0.88);">🛡 SQL safety validation</div>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

# ── Gate ──────────────────────────────────────────────────────
result = st.session_state.get("build_result")

if not result:
    st.markdown(
        '<div style="background:white;border:1px solid #E5E7EB;border-radius:12px;'
        'padding:64px 32px;text-align:center;margin-top:4px;">'
        '<div style="font-size:40px;margin-bottom:16px;">📂</div>'
        '<div style="font-size:17px;font-weight:800;color:#111827;margin-bottom:8px;letter-spacing:-0.02em;">'
        'No database yet</div>'
        '<div style="font-size:14px;color:#6B7280;line-height:1.65;">'
        'Upload a CSV or use the synthetic example,<br>'
        'then click <strong style="color:#111827;">Build Database</strong> in the sidebar.</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# ── Metrics ───────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.markdown(metric_card("Rows imported", str(result["row_count"]), "#15803D"), unsafe_allow_html=True)
m2.markdown(metric_card("Columns", str(len(result["columns"])), "#2563EB"), unsafe_allow_html=True)
m3.markdown(metric_card("Data warnings", str(len(result["validation"])), "#D97706"), unsafe_allow_html=True)
m4.markdown(metric_card("Storage", "SQLite", "#7C3AED"), unsafe_allow_html=True)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────
build_tab, ask_tab, evidence_tab = st.tabs(["🗂  Build & Validate", "💬  Ask", "🔍  Evidence"])

# ─────────────────────────────────────────────────────────────
# BUILD TAB
# ─────────────────────────────────────────────────────────────
with build_tab:
    left, right = st.columns(2, gap="large")

    with left:
        section_heading("Inferred Schema")
        st.dataframe(pd.DataFrame(result["columns"]), use_container_width=True, height=220)

        if ollama_status.available:
            st.markdown(
                '<div style="display:inline-flex;align-items:center;gap:6px;'
                'background:#FFF7ED;border:1px solid #FED7AA;color:#C2410C;'
                'font-size:11px;font-weight:700;border-radius:6px;'
                'padding:3px 10px;margin-bottom:10px;letter-spacing:0.04em;text-transform:uppercase;">'
                '⚡ Agent 1 · Schema Reviewer</div>',
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
                f'<div style="background:#F0FDF4;border:1px solid #BBF7D0;'
                f'border-left:4px solid #16A34A;border-radius:0 8px 8px 0;'
                f'padding:14px 18px;margin-top:10px;font-size:13.5px;'
                f'line-height:1.75;color:#14532D;white-space:pre-wrap;">'
                f'{st.session_state["schema_review"]}</div>',
                unsafe_allow_html=True,
            )

    with right:
        section_heading("Validation Report")
        if result["validation"]:
            for w in result["validation"]:
                st.warning(w)
        else:
            st.success("No missing-value warnings detected.")

        if ollama_status.available and result["validation"]:
            st.markdown(
                '<div style="display:inline-flex;align-items:center;gap:6px;'
                'background:#FFF7ED;border:1px solid #FED7AA;color:#C2410C;'
                'font-size:11px;font-weight:700;border-radius:6px;'
                'padding:3px 10px;margin-bottom:10px;letter-spacing:0.04em;text-transform:uppercase;">'
                '⚡ Agent 2 · Validation Analyst</div>',
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
                f'<div style="background:#F0FDF4;border:1px solid #BBF7D0;'
                f'border-left:4px solid #16A34A;border-radius:0 8px 8px 0;'
                f'padding:14px 18px;margin-top:10px;font-size:13.5px;'
                f'line-height:1.75;color:#14532D;white-space:pre-wrap;">'
                f'{st.session_state["validation_analysis"]}</div>',
                unsafe_allow_html=True,
            )

    st.divider()
    section_heading("Database Preview — first 20 rows")
    rows = run_sql(db_path, f"SELECT * FROM {result['table_name']} LIMIT 20")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ─────────────────────────────────────────────────────────────
# ASK TAB
# ─────────────────────────────────────────────────────────────
with ask_tab:
    q_col, opt_col = st.columns([3, 2], gap="large")
    with q_col:
        section_heading("Your Question")
        choice = st.selectbox("Example", EXAMPLE_QUESTIONS, label_visibility="collapsed")
        question = st.text_input("Edit or write your own question", value=choice)
    with opt_col:
        section_heading("Options")
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

        src_label = answer.source
        src_color = "#15803D" if "gemma" in src_label.lower() else "#6B7280"
        st.markdown(
            f'<div style="font-size:11px;font-weight:700;letter-spacing:0.09em;'
            f'text-transform:uppercase;color:#9CA3AF;margin-bottom:10px;">'
            f'Generated SQL &nbsp;<span style="color:{src_color};text-transform:none;'
            f'letter-spacing:0;font-size:11px;">· {src_label}</span></div>',
            unsafe_allow_html=True,
        )
        st.code(answer.sql, language="sql")

        section_heading("Results")
        st.dataframe(pd.DataFrame(answer.rows), use_container_width=True)

        if st.session_state.get("last_summary"):
            st.markdown(
                '<div style="display:inline-flex;align-items:center;gap:6px;'
                'background:#FFF7ED;border:1px solid #FED7AA;color:#C2410C;'
                'font-size:11px;font-weight:700;border-radius:6px;'
                'padding:3px 10px;margin-bottom:10px;letter-spacing:0.04em;text-transform:uppercase;">'
                '⚡ Agent 4 · Answer Explainer</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="background:#F0FDF4;border:1px solid #BBF7D0;'
                f'border-left:4px solid #16A34A;border-radius:0 8px 8px 0;'
                f'padding:14px 18px;margin-top:4px;font-size:13.5px;'
                f'line-height:1.75;color:#14532D;white-space:pre-wrap;">'
                f'{st.session_state["last_summary"]}</div>',
                unsafe_allow_html=True,
            )

# ─────────────────────────────────────────────────────────────
# EVIDENCE TAB
# ─────────────────────────────────────────────────────────────
with evidence_tab:
    answer = st.session_state.get("last_answer")
    if not answer:
        st.markdown(
            '<div style="text-align:center;padding:60px 20px;">'
            '<div style="font-size:40px;margin-bottom:16px;">🔍</div>'
            '<div style="font-size:17px;font-weight:800;color:#111827;'
            'margin-bottom:8px;letter-spacing:-0.02em;">No query run yet</div>'
            '<div style="font-size:14px;color:#6B7280;line-height:1.65;">'
            'Run a trusted query in the Ask tab,<br>'
            'then return here to inspect source evidence.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    elif answer.provenance:
        section_heading("Source Provenance")
        st.markdown(
            '<div style="font-size:13px;color:#6B7280;margin-bottom:14px;line-height:1.6;">'
            'Each result row is traced back to a specific line in your original CSV file. '
            'Staff can verify any answer against the source data.'
            '</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(pd.DataFrame(answer.provenance), use_container_width=True)
    else:
        st.info("This aggregate query does not map to individual source rows.")
