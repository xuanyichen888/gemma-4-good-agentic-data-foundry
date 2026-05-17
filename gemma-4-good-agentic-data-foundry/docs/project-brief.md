# Project Brief

## Working title

Agentic Data Foundry: trusted local database construction for underserved organizations.

## Target user

Small community organizations that collect operational records but lack a dedicated data engineer.

Candidate users:

- community resource centers
- student support offices
- small public-health outreach teams
- volunteer-run legal aid or benefits-navigation groups

## Problem

These organizations often have useful data, but it is fragmented across spreadsheets, forms, notes, and PDFs. This makes it hard to answer operational questions, prepare reports, or track follow-up needs.

The common fallback is manual spreadsheet cleanup, which is slow, error-prone, and hard to audit.

## Solution

Build a local-first agentic workflow that turns messy source files into a structured SQLite database with source provenance and validation checks.

Core workflow:

1. Ingest messy records.
2. Infer entities, columns, and data types.
3. Build a database.
4. Store provenance for each imported row.
5. Run validation checks.
6. Use Gemma 4 to translate natural-language questions into SQL.
7. Return answers with evidence and warnings.

## Gemma 4 usage

Gemma 4 should be used meaningfully, not as a thin chatbot wrapper.

Proposed model roles:

- schema designer: propose normalized fields and human-readable descriptions
- extraction agent: map messy source columns into canonical fields
- validation agent: explain suspicious records and propose fixes
- query agent: generate SQL from user questions, then call the database tool
- explanation agent: summarize answer provenance and limitations

## Why local-first matters

Many target organizations handle sensitive records and cannot send raw data to a cloud API. Local inference makes the project more credible for privacy-sensitive education, health, and social-service workflows.

## Evaluation plan

Minimum evaluation:

- schema quality: compare inferred column types against a hand-labeled schema
- extraction accuracy: percent of rows imported without field loss
- SQL correctness: set of 10 natural-language questions with expected SQL/results
- trust quality: percent of answers that include correct provenance references
- usability: time saved versus manual spreadsheet cleanup in a scripted task

## MVP demo

Input:

- `examples/community_intake.csv`

User questions:

- Which clients need follow-up this week?
- What are the most common unmet needs?
- Which ZIP codes have the highest number of transportation requests?
- How many open cases are older than 14 days?

Demo output:

- inferred schema
- SQLite table preview
- validation report
- generated SQL
- answer table
- provenance row IDs

## Main risks

- Project becomes too broad.
- The demo looks like a generic CSV chatbot.
- The agent makes unsafe SQL or unsupported claims.
- The video spends too much time on architecture and not enough time on user pain.

## Anti-scope

Do not build:

- a generic PDF chatbot
- a medical advice tool
- a huge multi-domain data platform
- a polished UI without a reliable database pipeline

