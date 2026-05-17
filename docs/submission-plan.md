# Submission Plan

## Competition facts to track

- Competition: Gemma 4 Good Hackathon
- Organizer: Kaggle and Google DeepMind
- Publicly reported prize pool: $200,000
- Publicly reported final submission deadline: May 18, 2026
- Expected deliverables: working prototype, public code repository, video demo, writeup, demo or demo files

Confirm the final details on the Kaggle competition page before submission.

## Submission positioning

Primary track:

- Digital Equity and Inclusivity

Secondary track:

- Safety and Trust

Technology angle:

- Local Gemma 4 inference
- Function/tool calling
- SQLite database tools
- Evidence-backed answers

## 3-minute video outline

### 0:00-0:20 Problem

Show a messy spreadsheet and explain that small organizations need reliable data but cannot afford a data team.

### 0:20-0:45 User story

Introduce a community resource center that needs to identify unmet needs and follow-up cases.

### 0:45-1:45 Demo

Show the user uploading intake data, the agent inferring a schema, creating a database, and answering one or two natural-language questions.

### 1:45-2:20 Trust layer

Show provenance, validation warnings, and generated SQL.

### 2:20-2:45 Gemma 4 architecture

Briefly show the agent roles and local-first model usage.

### 2:45-3:00 Impact

End with measurable impact: faster reporting, lower technical barrier, better privacy, and reusable public code.

## Writeup outline

1. Problem and target user
2. Why existing tools fail this user
3. System overview
4. How Gemma 4 is used
5. Demo workflow
6. Evaluation and results
7. Safety, privacy, and limitations
8. Future work

## Milestones

### Day 1

- finalize user persona
- finalize demo dataset
- create repo skeleton
- build deterministic CSV-to-SQLite pipeline

### Days 2-4

- add Streamlit app
- add validation report
- add provenance table
- add first NL2SQL path

### Days 5-8

- integrate Gemma 4 via Ollama or Kaggle notebook environment
- add tool-calling loop
- prepare 10-question evaluation set

### Days 9-12

- improve UI
- write technical report draft
- record first demo footage

### Days 13-16

- finalize video
- polish README
- freeze demo flow
- submit early

## Definition of done

- A new user can run the demo from README instructions.
- The app works on the synthetic dataset without private data.
- At least 10 NL questions have expected answers.
- Each answer can show source evidence or row provenance.
- The video tells the story in under 3 minutes.

## Current demo status

- CSV-to-SQLite pipeline is implemented.
- Inferred schema and validation warnings are visible in Streamlit.
- The query path has read-only SQL safety checks.
- Row-level answers can show source CSV row provenance.
- Six example evaluation questions are listed in `examples/evaluation_questions.csv`.
- `scripts/smoke_test.py` verifies the deterministic demo path.
- `scripts/check_local_model.py` checks whether Ollama and a Gemma model are available.
- The current app can use any local Ollama model for debugging, but the submission should set `ADF_OLLAMA_MODEL` to a Gemma model.
