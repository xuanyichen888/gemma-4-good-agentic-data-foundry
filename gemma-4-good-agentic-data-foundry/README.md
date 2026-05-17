# Agentic Data Foundry

**Gemma 4 Good Hackathon · Digital Equity and Inclusivity · Safety and Trust**

Small nonprofits, clinics, and community organizations collect critical records in spreadsheets and forms — but they rarely have a data engineer. Agentic Data Foundry uses a local-first Gemma 4 agent pipeline to turn messy CSV files into a trusted, queryable SQLite database with source provenance and natural-language access.

---

## The problem

A community resource center tracks hundreds of intake records across spreadsheets. Staff need to answer questions like "Which clients need follow-up this week?" or "What are the top unmet needs by ZIP code?" — but their data is inconsistent, has missing fields, and lives in files that change weekly. They cannot send this data to a cloud API. They have no SQL expertise.

Existing tools either require cloud data sharing (privacy risk), SQL knowledge (technical barrier), or produce answers with no source evidence (trust risk). All three problems hit underserved organizations hardest.

---

## Solution overview

```
CSV upload
    │
    ▼
Schema Inference ──→ [Gemma Agent 1: Schema Reviewer]
    │                  Reviews inferred types, flags ZIP-as-integer risks,
    │                  suggests missing fields for better reporting
    ▼
SQLite Build + Provenance Table
    │
    ▼
Validation Report ──→ [Gemma Agent 2: Validation Analyst]
    │                   Explains data quality warnings in plain language,
    │                   prioritizes which gaps pose the highest service risk
    ▼
Natural-Language Question
    │
    ▼
[Gemma Agent 3: SQL Generator with Auto-Repair Loop]
    │   1. Gemma generates SELECT query
    │   2. Safety checker validates (read-only, no forbidden terms,
    │      no unknown tables, no comments or multi-statements)
    │   3. If blocked → send error back to Gemma for repair (up to 2 retries)
    │   4. Execute against read-only SQLite connection
    ▼
Answer Table + Source Provenance
    │
    ▼
[Gemma Agent 4: Answer Explainer]
    Summarizes what the answer means, what evidence supports it,
    and what data quality caveats apply
```

Every answer links to source CSV row numbers via a provenance table. Staff can verify any result against the original file.

---

## Why this is not a generic CSV chatbot

| Generic chatbot | Agentic Data Foundry |
|---|---|
| Sends raw data to a cloud API | Local-only inference via Ollama |
| No schema curation | Gemma reviews inferred types and flags risks |
| No data quality feedback | Gemma explains missing-field warnings |
| SQL runs unchecked | Multi-rule safety validator + read-only connection |
| Answers have no source | Row-level provenance links every answer to CSV line numbers |
| Single prompt → answer | Iterative repair loop: Gemma fixes its own broken SQL |

---

## Gemma model roles

| Agent | Role | When it runs |
|---|---|---|
| Schema Reviewer | Reviews auto-inferred column types, flags type errors (ZIP as INTEGER), suggests additional fields | After DB build, on demand |
| Validation Analyst | Explains missing-field warnings in plain language, ranks risks for service operations | After DB build, on demand |
| SQL Generator | Translates natural-language questions into safe SQLite SELECT queries | On every NL query |
| SQL Repair | Receives error message + broken SQL, produces corrected query (up to 2 retries) | When SQL fails safety check or execution |
| Answer Explainer | Summarizes what the answer says, what evidence supports it, and one data quality caveat | After query, on demand |

All five roles run locally via `ollama` with `gemma3n:e4b`. No data leaves the user's machine.

---

## Evaluation

10 natural-language questions with expected SQL patterns are in `examples/evaluation_questions.csv`. The deterministic fallback path passes all 10:

```bash
python3 scripts/smoke_test.py
# smoke test passed for all evaluation questions
```

The Gemma path is tested interactively in the Streamlit demo. SQL safety validation runs on every query regardless of source.

---

## Local setup

```bash
git clone https://github.com/xuanyichen888/gemma-4-good-agentic-data-foundry.git
cd gemma-4-good-agentic-data-foundry

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Verify setup
python3 scripts/smoke_test.py

# Start the app
streamlit run app/streamlit_app.py
```

### Enable Gemma (local inference)

```bash
# Install Ollama: https://ollama.com
ollama pull gemma3n:e4b
# Ollama starts automatically on most systems; if not: ollama serve
```

Once Ollama is running, the sidebar will show "Gemma model active." All four agent features become available. The app works without Gemma (deterministic fallback), but the full agentic pipeline requires a local Gemma model.

---

## Demo flow

1. Click **Build database** in the sidebar (uses synthetic `community_intake.csv`)
2. In **Build & Validate** tab:
   - Review the inferred schema
   - Click **Run schema review agent** — Gemma flags potential type issues and suggests improvements
   - Click **Explain warnings with Gemma agent** — Gemma explains missing-field risks in plain language
3. In **Ask** tab:
   - Select an example question or type your own
   - Check **Generate SQL with local Ollama model**
   - Click **Run trusted query**
   - See the generated SQL, the result table, and (optionally) Gemma's explanation
4. In **Evidence** tab:
   - See source CSV row provenance for every answer row

---

## Safety and privacy

- All inference runs locally via Ollama. No data is sent to any external API.
- SQL is validated before execution: read-only connection, SELECT-only, no forbidden keywords, no unknown table references, no SQL comments, no multi-statement execution.
- If Gemma generates an unsafe query, the error is sent back to Gemma for repair (up to 2 retries). If repair fails, the query is blocked with an explanation.
- The provenance table is append-only and references source file paths and row numbers, not raw data.

---

## Repository layout

```
app/
  streamlit_app.py          Streamlit UI
docs/
  demo-script.md            3-minute video script
  deployment.md             Streamlit Community Cloud deploy guide
  project-brief.md          Project framing
  submission-plan.md        Hackathon milestone plan
examples/
  community_intake.csv      Synthetic privacy-safe demo dataset
  evaluation_questions.csv  10 NL questions with expected SQL signals
scripts/
  smoke_test.py             Deterministic pipeline regression test
  check_local_model.py      Ollama and Gemma availability check
src/
  agentic_data_foundry/
    schema.py               Column normalization and SQLite type inference
    database.py             CSV import, provenance table, read-only SQL execution
    query.py                NL fallback, SQL safety validation, Gemma repair loop
    llm.py                  Ollama client, 5 Gemma prompt builders
```

---

## Research framing

For academic audiences: this project explores **reliable LLM agents for automatic database construction from messy real-world records**, with four design properties rarely combined in prior work:

1. **Local-first inference** — privacy constraint drives architecture
2. **Multi-agent decomposition** — schema review, validation analysis, SQL generation, SQL repair, and answer explanation as separate agent roles with distinct prompts
3. **Iterative self-repair** — the SQL generator receives its own error as feedback and revises, a lightweight tool-calling loop
4. **Provenance-grounded answers** — every row answer links to source evidence, supporting human auditing

Target application domain: small organizations serving vulnerable populations, where data quality and answer trustworthiness matter most and technical capacity is lowest.

---

## Hackathon tracks

- **Digital Equity and Inclusivity** — lowers the data engineering barrier for under-resourced organizations
- **Safety and Trust** — SQL safety validation, local inference, source provenance, and iterative repair collectively make the system auditable and accountable
