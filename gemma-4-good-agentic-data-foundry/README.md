# Agentic Data Foundry

An impact-focused Gemma 4 Good Hackathon project for turning messy community data into a trusted, queryable database.

## Project thesis

Small nonprofits, clinics, schools, and local support teams often keep critical records across spreadsheets, PDFs, forms, and notes. They need structured data for reporting, grant writing, service planning, and follow-up, but they usually do not have a data engineer.

Agentic Data Foundry uses a local-first Gemma 4 agent to:

- inspect messy files and infer a database schema
- extract records into SQLite
- keep provenance links from each database row back to source evidence
- validate missing or inconsistent fields
- answer natural-language questions over the database with generated SQL

The project targets the Gemma 4 Good themes of **Digital Equity and Inclusivity** plus **Safety and Trust**.

## Why this is a strong fit

This is not a generic chatbot. The user-facing workflow is concrete:

1. A small organization uploads messy operational data.
2. The system proposes a clean schema.
3. The agent builds a database and flags uncertain fields.
4. Staff can ask questions such as "Which households need follow-up this week?" or "What are the top unmet service needs by ZIP code?"
5. Every answer links back to source rows or source snippets.

That gives the submission a useful demo, a clear social-impact story, and a research angle around reliable LLM agents for structured data systems.

## MVP scope

The first version should support:

- CSV upload
- automatic schema inference
- SQLite database creation
- provenance table
- rule-based validation report
- optional Gemma-backed natural-language to SQL query generation
- read-only SQL safety checks before execution
- evidence view linking answer rows to source CSV row numbers
- a simple Streamlit demo

Stretch goals:

- PDF/text extraction
- Gemma function-calling tool loop
- schema revision from user feedback
- confidence scoring for extracted fields
- before/after impact metrics for a realistic nonprofit workflow

## Demo scenario

Recommended demo story:

> A community resource center has messy intake records from multiple programs. Staff need to identify service gaps, follow-up cases, and reporting metrics without hiring a data engineer. Agentic Data Foundry converts the data into a clean database and produces evidence-backed answers.

The example dataset in `examples/community_intake.csv` is synthetic and privacy-safe.

## Repository layout

```text
app/
  streamlit_app.py
docs/
  project-brief.md
  submission-plan.md
  team-roles.md
examples/
  community_intake.csv
  evaluation_questions.csv
scripts/
  smoke_test.py
src/
  agentic_data_foundry/
    database.py
    llm.py
    query.py
    schema.py
```

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/smoke_test.py
streamlit run app/streamlit_app.py
```

Optional local Gemma path:

```bash
ollama serve
ollama pull gemma3n:e4b
export ADF_OLLAMA_MODEL=gemma3n:e4b
```

The app should still run without an LLM by using the deterministic schema and SQLite pipeline. You can check local model readiness with:

```bash
python3 scripts/check_local_model.py
```

If you use a non-Gemma Ollama model for local debugging, the app will still work, but the final hackathon version should run through a Gemma model.

## Demo flow

1. Build the database from the sidebar.
2. Review the inferred schema and validation warnings.
3. Ask one of the example questions.
4. Inspect generated SQL before execution.
5. Open the evidence tab to see source row provenance.
6. Enable local model generation once a Gemma model is installed.

The deterministic fallback supports the example questions in `examples/evaluation_questions.csv`. Local Gemma can be enabled for the same trusted SQL execution path.

## Hackathon deliverables

- working demo
- public GitHub repository
- 3-minute video
- 1,500-word Kaggle writeup
- cover image / gallery assets

## Public sharing

This app should be shared as a Streamlit deployment, not as GitHub Pages. Push the repo to GitHub, then deploy `app/streamlit_app.py` on Streamlit Community Cloud. See `docs/deployment.md`.

## Team split

- Technical lead: agent workflow, database construction, evaluation, code repo
- Business teammate 1: user research, nonprofit/education/health operations use case, impact metrics
- Business teammate 2: pitch narrative, demo script, video, market and deployment story

## Application positioning

For MS applications, this project shows applied ML engineering and product execution.

For PhD applications, the research framing is:

> Reliable LLM agents for automatic database construction from messy real-world data, with provenance, validation, and natural-language interfaces.
