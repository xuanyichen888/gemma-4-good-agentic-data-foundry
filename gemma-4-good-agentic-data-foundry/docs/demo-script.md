# Demo Script

## One-sentence pitch

Agentic Data Foundry helps small community organizations turn messy intake records into a trusted local database, then ask operational questions with evidence-backed answers.

## Three-minute video draft

### 0:00-0:20

Small community organizations collect important service records, but those records often live in spreadsheets, notes, and exported forms. Without a data team, even simple questions can take hours of manual cleanup.

### 0:20-0:45

Our example is a community resource center tracking food assistance, transportation, housing referrals, benefits navigation, and follow-up cases. The records are synthetic, but the workflow mirrors common nonprofit operations.

### 0:45-1:25

We upload the intake data and build a local SQLite database. The system infers a schema, imports rows, and flags data quality issues such as missing follow-up dates.

### 1:25-1:55

Now staff can ask natural-language questions. For example: which clients need follow-up this week? The system generates SQL, checks that it is read-only and limited to approved tables, then returns the result.

### 1:55-2:20

Trust is the key layer. For row-level answers, each result links back to the original source file and source row number, so staff can audit the answer instead of blindly trusting a chatbot.

### 2:20-2:45

Gemma 4 fits into the agent loop as the local reasoning layer: schema design, SQL drafting, validation explanations, and natural-language summaries. The deterministic path keeps the demo reliable even without cloud APIs.

### 2:45-3:00

The impact is practical: faster reporting, lower technical barriers, better privacy, and a reusable open-source workflow for organizations that need data infrastructure but cannot hire a data engineer.

## Recording checklist

- Build database from sidebar.
- Show inferred schema.
- Show validation warning for missing follow-up date.
- Show local model status in the sidebar.
- Ask "Which clients need follow-up this week?"
- Show SQL and answer table.
- Open Evidence tab.
- Ask "What are the most common unmet needs?" to show aggregate reporting.
- If Gemma is installed, enable local model generation and show the model explanation.
