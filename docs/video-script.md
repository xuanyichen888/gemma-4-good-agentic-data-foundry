# 视频拍摄脚本 & 讲解稿
## Agentic Data Foundry — Gemma 4 Good Hackathon

> **总时长：3:00 分钟**  
> **语言：英语（评委是国际评审）**  
> **风格：清晰、有情感、有证据 — 不是产品广告，是一个解决真实问题的故事**

---

## 拍摄前准备

- 浏览器预先打开 `http://localhost:8501`
- 侧边栏已加载，Judge Demo Mode 可见
- 屏幕分辨率 1280×800，字体放大到 110%
- 关闭所有通知、消息弹窗
- 录音环境安静，麦克风距离嘴部 20-30cm

---

## 完整讲解稿（逐句）

---

### 【段落 1：人物与问题】0:00 – 0:25

**画面**：讲话人出镜（或幻灯片：一张社区中心工作人员的照片，或一个打开了三十个 Excel 标签的截图）

**稿子**：

> "Meet Maria. She coordinates intake for a community resource center — helping families access food, housing, and health services.
>
> Every week, Maria opens four different spreadsheets, tries to remember which clients need follow-up, and wonders if she's missing anyone. Last month, a family was nearly evicted because their follow-up date was in a column that got accidentally deleted.
>
> Maria's organization can't use cloud tools. Their data is sensitive — health records, housing status, family income. It has to stay local."

**拍摄提示**：语速稍慢，"nearly evicted" 和 "has to stay local" 两句加重语气。

---

### 【段落 2：展示产品 — 建库与发现风险】0:25 – 1:10

**画面**：切换到浏览器全屏，展示 App 首页

**稿子**：

> "This is Agentic Data Foundry. It uses four local Gemma 4 agents to transform Maria's messy CSV into a trusted, queryable database — without sending a single byte of client data to the cloud.
>
> *(点击侧边栏 Judge Demo Mode 按钮)*
>
> I'll hit 'Run Demo' — this builds the database, runs a complete analysis, and populates every agent's output in under ten seconds."

**操作**：点击 "▶ Run 90-second Demo"，等页面刷新完成。

> "The database is built. Ten client records, nine columns. And immediately — Agent 1, the Schema Reviewer, flags a critical issue:
>
> *(指向 Build & Validate 标签 → Schema Reviewer 输出)*
>
> ZIP codes are stored as INTEGER. That silently strips leading zeros — ZIP 07102 becomes 7102. Geographic filtering would fail silently. This is exactly the kind of bug a data engineer catches on day one."

**操作**：指向 Schema Review 输出框中的第一条 bullet。

> "Agent 2, the Validation Analyst, explains the missing data in plain language — not as a column name, but as a service risk:
>
> *(指向 Validation 输出)*
>
> 'A single missed follow-up in a housing case can trigger eviction.' That's the sentence Maria needs to read. Not 'follow_up_date: 1 null value.'"

**拍摄提示**：读引号里的话要慢，有停顿，让评委听清楚。

---

### 【段落 3：自然语言查询与 Provenance】1:10 – 1:55

**画面**：切换到 Ask 标签

**稿子**：

> "Now Maria wants to know: which clients need follow-up this week?
>
> *(指向已预填的问题输入框)*
>
> She doesn't write SQL. She asks in plain English. Agent 3 — the SQL Generator — translates that question into a safe SQLite SELECT query.
>
> *(指向 SQL 代码块)*
>
> Notice the safety layer: this query went through six validation rules before execution. No INSERT, no DROP, no unknown table references. If Gemma had generated broken SQL, the system sends the error back to Gemma for automatic repair — up to two retries.
>
> And here are the results — three clients who need contact this week, ordered by urgency."

**操作**：指向结果表格中的数据行。

> "*(切换到 Evidence 标签)*
>
> Every single row in this result links back to a specific line in the original CSV file. Staff can verify any answer against the paper intake form. This is what we call row-level provenance — and it's what separates a trusted answer from a hallucinated one."

**操作**：指向 Evidence 表格中的 `source_file` 和 `source_row` 列。

**拍摄提示**：这是技术差异化最强的点，语气要确定，不要快读。

---

### 【段落 4：架构与 Local-First 的意义】1:55 – 2:40

**画面**：切换到 Architecture 标签

**稿子**：

> "*(指向 Architecture 标签)*
>
> Here's the full system: four Gemma 4 agents, each with a distinct role and a controlled output budget. Schema Reviewer. Validation Analyst. SQL Generator with auto-repair. Answer Explainer.
>
> All inference runs via Ollama on the local machine. We set num_ctx to 2048 — reducing the KV-cache by sixty times compared to Gemma's 128K default. That's what makes local inference fast enough to be practical on a laptop.
>
> And critically: no data leaves the machine. No API call to OpenAI. No CSV uploaded to a cloud service. For an organization handling health records or housing data, that's not a nice-to-have — it's a requirement."

**操作**：缓慢滚动 Architecture 标签，让评委看到 safety rules 列表。

> "The safety validator runs on every query, regardless of whether it came from Gemma or the deterministic fallback. Six rules. Read-only SQLite connection at the OS level. The model literally cannot modify the database."

---

### 【段落 5：愿景结尾】2:40 – 3:00

**画面**：讲话人出镜，或切回首页英雄区

**稿子**：

> "Maria doesn't need to wait for a data engineer to be hired. She doesn't need to send client records to a cloud API. She needs a system that's trustworthy, auditable, and works offline.
>
> Every small organization serving vulnerable populations deserves a data engineer. Agentic Data Foundry is that engineer — running locally, explaining itself, and leaving a paper trail for every answer.
>
> Thank you."

**拍摄提示**：最后一句"Thank you"前停顿一秒。整段保持平静、有力的语气，不要急。

---

## 时间轴对照表

| 时间点 | 段落 | 关键操作 | 核心词 |
|--------|------|----------|--------|
| 0:00 | 问题 | 出镜/PPT | "nearly evicted", "has to stay local" |
| 0:25 | 点 Demo | 点 Judge Demo 按钮 | "without sending a single byte" |
| 0:45 | Schema Review | 指向 ZIP INTEGER 警告 | "silently strips leading zeros" |
| 1:00 | Validation | 读出 bullet 原文 | "trigger eviction" |
| 1:10 | Ask 查询 | 指向问题输入框和 SQL | "six validation rules" |
| 1:35 | Provenance | 切换 Evidence 标签 | "row-level provenance" |
| 1:55 | Architecture | 切换 Architecture 标签 | "sixty times", "no data leaves" |
| 2:25 | Safety rules | 滚动展示规则列表 | "read-only at the OS level" |
| 2:40 | 结尾 | 出镜 | "Every small organization deserves..." |
| 2:58 | 结束 | 停顿 | "Thank you." |

---

## 常见拍摄问题处理

### Q: Gemma 太慢，演示时卡住怎么办？
**A**：用 Judge Demo Mode。所有 Gemma 输出已预计算，0 等待时间。演示时说：*"The Gemma agents run locally — I've pre-loaded the output so we can see the full flow without waiting for inference."*

### Q: 超时了怎么剪辑？
**A**：可以剪辑掉 Gemma 运行中的 spinner 等待画面，直接跳到结果。说明：*"In a live run, Gemma takes 30-60 seconds to generate this analysis locally."*

### Q: 需要字幕吗？
**A**：建议加英文字幕（YouTube 自动字幕通常足够），有助于非母语英语的评委理解。

### Q: 视频上传到哪里？
**A**：YouTube（设置为公开或"不公开但有链接可访问"），获取链接后填入 Kaggle 提交页。

---

## 备用开场白（如果不想出镜）

> "In the United States alone, there are over 1.5 million nonprofits. Most of them track client records in spreadsheets. Most of them cannot afford a data engineer. And most of them cannot send their data to the cloud.
>
> Agentic Data Foundry was built for them."

---

## 视频发布后检查清单

- [ ] 视频时长 ≤ 3:00
- [ ] YouTube 链接可以在无登录状态下访问
- [ ] 声音清晰，无背景噪音
- [ ] 关键数字清晰可见（10 rows, 1 warning, 3 follow-ups）
- [ ] Kaggle 提交页已填入视频链接
- [ ] GitHub 仓库链接已填入提交页
- [ ] Writeup 包含：impact, technical approach, Gemma usage, local-first rationale

---

*脚本版本：2026-05-17 | 对应代码版本：main branch ee2a417*
