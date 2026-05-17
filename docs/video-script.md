---
output:
  pdf_document: default
  html_document: default
---
# 视频拍摄脚本 & 讲解稿
## Agentic Data Foundry — Gemma 4 Good Hackathon

> **总时长：3:00 分钟**  
> **语言：英语**  
> **原则：只说可验证的事实 — 演示中出现的数据、代码、技术细节全部真实存在于仓库中**

---

## 拍摄前准备

- 浏览器预先打开 `http://localhost:8501`，已完成 Judge Demo Mode
- 另开一个标签页展示 GitHub 仓库（备用）
- 屏幕分辨率 1280×800，字体放大到 110%
- 关闭所有通知、消息弹窗
- 录音环境安静，麦克风距离嘴部 20-30cm

---

## 完整讲解稿（逐句）

---

### 【段落 1：真实问题】0:00 – 0:25

**画面**：用文本编辑器或 Excel 打开 `examples/community_intake.csv`，让评委看到原始文件

**稿子**：

> "This is a CSV file — ten client intake records from a synthetic community resource center dataset. It has nine columns: client ID, intake date, ZIP code, primary need, household size, language, follow-up date, status, and notes.
>
> Look at row 8. Client C008 has an open housing case — but the follow-up date column is blank. There's no scheduled check-in. In a real intake system, this record would sit silently in a spreadsheet with no flag, no reminder, no alert.
>
> This is not a hypothetical. This is the actual file in our repository. And this is the problem we built Agentic Data Foundry to solve."

**拍摄提示**：打开 CSV 文件，用鼠标或高亮指向第 8 行的空白 follow_up_date 单元格。语气平静、直接。

---

### 【段落 2：建库 + Agent 1 & 2】0:25 – 1:10

**画面**：切换到 App，已通过 Judge Demo Mode 预加载所有结果

**稿子**：

> "Agentic Data Foundry takes that CSV and runs it through a pipeline of four local Gemma 4 agents — all inference happens on this machine, via Ollama. No data is sent to any external API.
>
> The database is built. Ten rows, nine columns, one data quality warning detected automatically.
>
> Agent 1 — the Schema Reviewer — reads the inferred column types and flags a real issue in this dataset:
>
> *(切换到 Build & Validate 标签，指向 Schema Review 输出)*
>
> ZIP code is stored as INTEGER. This CSV has ZIP codes like 90011 and 90037 — Los Angeles ZIP codes that happen to start with nine. But if any record had a ZIP starting with zero, the integer type would silently strip it. The agent flags this before it becomes a data integrity problem."

**操作**：指向 Schema Review 输出第一条 bullet 中的 `zip_code` 提及处。

> "Agent 2 — the Validation Analyst — takes the missing follow_up_date warning and translates it from a column name into a service implication:
>
> *(指向 Validation Analyst 输出)*
>
> 'One open case has no scheduled check-in. A client with an open housing referral and no follow-up date may fall through without a system alert.'
>
> That's what staff need to read — not 'null count: 1'."

**拍摄提示**：读 Gemma 输出时语速放慢，指向屏幕上对应文字。

---

### 【段落 3：自然语言查询 + Safety + Provenance】1:10 – 1:55

**画面**：切换到 Ask 标签，问题已预填

**稿子**：

> "Now — a staff member wants to know which clients need follow-up this week. They type that question in plain English.
>
> Agent 3 translates it into a SQLite SELECT query. But before that query touches the database, it goes through a safety validator with six rules:
>
> *(指向 SQL 代码块)*
>
> The query must start with SELECT. Forbidden keywords — DROP, DELETE, INSERT, UPDATE — are blocked at parse time. Only the imported table is accessible. No SQL comments. No multi-statements. And the SQLite connection itself is opened in read-only mode at the OS level — the model cannot modify the data even if it tried.
>
> Here are the results: nine open cases, ordered by follow-up date. Client C008 — the one with the blank date — is not in this list. That's correct. And that's exactly why the Validation Analyst flagged it separately."

**操作**：指向结果表格，特别指出 C008 不在结果里。

> "*(切换到 Evidence 标签)*
>
> Every row in this result includes a provenance record: the source file path and the exact row number in the original CSV. Staff can verify any answer directly against the source file. This is row-level audit trail — not a summary, not an approximation."

**操作**：指向 Evidence 表格中 `source_file` 和 `source_row` 列的具体值。

---

### 【段落 4：架构 + 技术选择的真实原因】1:55 – 2:40

**画面**：切换到 Architecture 标签

**稿子**：

> "Here's the full system.
>
> *(指向 Architecture 标签的 agent 列表)*
>
> Four agents. Each has a distinct prompt, a specific output contract, and a token budget. Schema Reviewer outputs up to 512 tokens. SQL Generator is capped at 150 — a SQL query doesn't need more. This is not arbitrary: smaller output budgets mean faster inference on local hardware.
>
> One specific technical decision worth explaining: we set num_ctx to 2048.
>
> Gemma 4's default context window is 128,000 tokens. Allocating that much KV-cache for a 400-token prompt wastes memory and slows every decode step. Our longest prompt — the answer summary with result rows and provenance — is under 600 tokens. Setting num_ctx to 2048 reduces KV-cache memory by roughly sixty times and makes local inference practical on a standard laptop.
>
> *(滚动到 Safety rules 区域)*
>
> The safety validator runs on every query — whether it came from Gemma or the deterministic fallback. If Gemma generates a query that fails validation, the error message is sent back to Gemma as context. It repairs its own output. Up to two retries."

---

### 【段落 5：为什么 Local-First + 结尾】2:40 – 3:00

**画面**：切换回首页，或讲话人出镜

**稿子**：

> "Community organizations handling intake records — food assistance, housing referrals, health navigation — often cannot use cloud tools. Their data includes income information, immigration status, health conditions. Sending that to an external API is not a policy question. For many organizations, it's a legal barrier.
>
> Agentic Data Foundry runs entirely on-device. The inference, the database, the provenance table — nothing leaves the machine.
>
> The code is open. The prompts are readable. Every answer includes a source citation. That's what trustworthy AI looks like for the organizations that need it most."

**拍摄提示**："nothing leaves the machine" 和 "every answer includes a source citation" 这两句停顿后再说。结尾不需要谢谢，直接停在最后一句。

---

## 时间轴对照表

| 时间点 | 画面 | 核心信息 | 可验证依据 |
|--------|------|----------|------------|
| 0:00 | CSV 文件原文 | C008 的 follow_up_date 为空 | community_intake.csv 第 8 行 |
| 0:25 | App 首页 | 4 个本地 Gemma 4 agent | llm.py, query.py |
| 0:45 | Build 标签 | ZIP 存为 INTEGER 的风险 | Schema Reviewer 输出 |
| 1:00 | Build 标签 | Validation Analyst 的服务风险描述 | Validation Analyst 输出 |
| 1:10 | Ask 标签 | 6 条安全规则 | query.py `validate_sql()` |
| 1:30 | Ask 标签 | C008 不在结果里，这是正确的 | 结果表格 |
| 1:45 | Evidence 标签 | source_file + source_row | provenance 表格 |
| 1:55 | Architecture 标签 | num_ctx=2048，减少 60 倍 KV-cache | llm.py `NUM_CTX = 2048` |
| 2:20 | Architecture 标签 | Gemma 自修复 SQL | query.py repair loop |
| 2:40 | 首页或出镜 | Local-only，无外部 API | Ollama 架构 |

---

## 拍摄注意事项

**关于数据**：演示中出现的所有数据来自仓库内的 `examples/community_intake.csv`，是合成数据（synthetic），不包含真实个人信息，可以完整展示在视频中。

**关于技术数字**：
- "60 times" — `128000 / 2048 ≈ 62.5`，代码中 `NUM_CTX = 2048`，可查
- "six rules" — `query.py` 中 `validate_sql()` 函数，有 6 条检查
- "up to two retries" — `query.py` 中 `max_repair=2`，可查
- "150 tokens for SQL" — `llm.py` 中 `NUM_PREDICT_SQL = 150`，可查

**如果 Gemma 运行缓慢**：提前用 Judge Demo Mode 预加载结果，演示时说：*"I've pre-run the Gemma agents locally so we can see the full output without waiting for inference in real time."* — 这是真实的，不是作弊。

---

## 备用开场（如果不想从 CSV 文件开始）

> "This repository contains a working pipeline that takes a CSV file and returns SQL-queryable results with row-level provenance — all using local Gemma 4 inference, with no external API calls.
>
> Let me show you what that actually means."

---

*脚本版本：2026-05-17 | 对应代码版本：main branch*
