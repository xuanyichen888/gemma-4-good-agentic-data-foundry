# 队友指南 — Agentic Data Foundry 提交冲刺

> 截止时间：**2026-05-18 23:59 UTC（北京时间 5月19日 07:59）**  
> 仓库：https://github.com/xuanyichen888/gemma-4-good-agentic-data-foundry  
> Kaggle 比赛页：https://www.kaggle.com/competitions/gemma-4-good-hackathon/overview

---

## 一、今日任务清单

| 优先级 | 任务 | 谁来做 | 状态 |
|--------|------|--------|------|
| 🔴 必须 | 录制 3 分钟演示视频 | - | 待做 |
| 🔴 必须 | 在 Kaggle 提交页填写 writeup | - | 待做 |
| 🟡 重要 | 测试 Judge Demo Mode 流程是否顺畅 | - | 待做 |
| 🟡 重要 | 确认 Ollama + gemma4:e4b 本地可用 | - | 待做 |
| 🟢 可选 | 润色 README 的 Research Framing 段落 | - | 待做 |

---

## 二、本地运行环境配置

### 最小依赖（无需 Gemma 也能跑完整 demo）

```bash
git clone https://github.com/xuanyichen888/gemma-4-good-agentic-data-foundry.git
cd gemma-4-good-agentic-data-foundry

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 快速验证
python3 scripts/smoke_test.py
# 预期输出：smoke test passed for 10 evaluation questions

# 启动 App
streamlit run app/streamlit_app.py
```

浏览器打开 `http://localhost:8501`

### 启用 Gemma 4（可选，演示更完整）

```bash
# 安装 Ollama：https://ollama.com
ollama pull gemma4:e4b       # 9.6 GB，下载一次

# 如果 Ollama 没有自动启动：
ollama serve
```

> **注意**：`gemma4:e4b` 在 MacBook Air M2 上每次推理约 30-120 秒。  
> **不需要等 Gemma**：Judge Demo Mode 用预计算结果，0 秒显示所有输出。

---

## 三、Judge Demo Mode（评委演示专用）

这是今天最重要的功能。点一个按钮完成整个演示流程：

1. 打开 `http://localhost:8501`
2. 侧边栏顶部有黄色区域 **"🎬 Judge Demo Mode"**
3. 点击 **"▶ Run 90-second Demo"**
4. 等约 3 秒（只是建库，不调 Gemma）
5. 页面自动展示：
   - ✅ 10 行数据已导入，发现 1 条数据质量警告
   - ✅ Schema Reviewer 的 3 条类型问题反馈（ZIP 存为 INTEGER 的风险）
   - ✅ Validation Analyst 的 3 条风险分析（follow_up 缺失的服务影响）
   - ✅ 自动提问"Which clients need follow-up this week?"并执行
   - ✅ 生成安全的 SQL 查询 + 结果表
   - ✅ Answer Explainer 的 3 条审计分析
6. 切换到 **Evidence 标签** 看行级 provenance（每行结果追溯到原始 CSV 行号）
7. 切换到 **Architecture 标签** 展示技术深度

**评委体验路径：约 2 分钟看完全部功能，不需要输入任何内容。**

---

## 四、录视频前的准备清单

### 环境检查
- [ ] `streamlit run app/streamlit_app.py` 能正常启动
- [ ] Judge Demo Mode 能运行（侧边栏黄色按钮）
- [ ] 浏览器窗口设置为 1280×800 以上
- [ ] 关闭通知、全屏、DND 模式
- [ ] 提前运行一遍 Judge Demo，确认页面状态正常

### 视频录制工具
- **Mac**：QuickTime Player → 新建屏幕录制
- **Windows**：Win+G 游戏录像
- **跨平台**：OBS Studio（免费）

### 视频规格要求
- 时长：**2:50 ~ 3:00 分钟**（不能超过 3 分钟）
- 分辨率：**1080p 或以上**
- 格式：MP4
- 必须能在 YouTube 或可访问链接播放

---

## 五、视频拍摄流程（分镜）

见 `docs/video-script.md` 获取完整稿子。

### 简版流程（30秒提示卡）

| 时间 | 画面 | 说什么 |
|------|------|--------|
| 0:00–0:25 | 人脸/PPT | 讲 Maria 的故事 |
| 0:25–0:55 | 浏览器全屏 | 打开 App，点 Judge Demo |
| 0:55–1:30 | Build & Validate 标签 | 展示 schema warning + Gemma 解释 |
| 1:30–2:00 | Ask 标签 | 问问题，展示 SQL + 结果 |
| 2:00–2:20 | Evidence 标签 | 展示 provenance 追溯 |
| 2:20–2:45 | Architecture 标签 | 快速讲 4 个 agent + 安全层 |
| 2:45–3:00 | 人脸 | 结尾愿景一句话 |

---

## 六、Kaggle 提交 Writeup 要点

在 Kaggle 比赛页提交时需要填写 notebook/writeup，建议包含以下内容：

### 标题
**Agentic Data Foundry: Local-First Gemma 4 Agents for Trustworthy Community Data**

### 摘要（约 200 字）
Small nonprofits and community organizations collect critical records in spreadsheets but lack data engineers. They cannot share this sensitive data with cloud APIs. Agentic Data Foundry uses a pipeline of four Gemma 4 agents running locally via Ollama to transform a messy CSV into a trusted, queryable SQLite database — with source provenance for every answer.

The system addresses three barriers: (1) **Privacy** — all inference is local, no data leaves the machine; (2) **Trust** — every row-level answer includes provenance linking back to the source CSV line; (3) **Safety** — a multi-rule SQL validator blocks all non-SELECT queries, and Gemma's broken SQL is auto-repaired in a retry loop.

### 技术要点（要提到）
1. **4 specialized Gemma 4 agent roles** — each with a distinct prompt and output contract
2. **num_ctx: 2048** — reduces KV-cache ~60× for short prompts, making local inference practical
3. **Streaming output** — token-by-token display; first output visible in ~2 seconds
4. **SQL safety validator** — 6 rules, runs on every query regardless of source
5. **Iterative self-repair** — Gemma receives its own error and produces corrected SQL (up to 2 retries)
6. **Row-level provenance** — `_adf_row_id` in every row-level query links to source CSV line numbers

### 社会影响（要提到）
- Target users: community intake coordinators, food bank staff, health clinic volunteers
- They handle HIPAA/FERPA-adjacent data — cloud APIs are not an option
- One missing follow-up date flagged by Gemma could mean the difference between a housed and unhoused client

### 链接
- GitHub: `https://github.com/xuanyichen888/gemma-4-good-agentic-data-foundry`
- Demo video: [填写 YouTube 链接]

---

## 七、如果 Gemma 在演示时很慢

**首选方案**：使用 Judge Demo Mode（不依赖 Gemma）

**备选方案**：
1. 只用 deterministic fallback 演示（取消勾选 "Generate SQL with Gemma"）
2. 演示时说：*"I'll show the deterministic path which is instant, and the Gemma path which adds AI-powered schema review and data quality explanation on top."*
3. 把 Gemma 的 canned response 截图放进视频

**调速方法**（如果有时间）：
```bash
# 用更小的模型测试
ADF_OLLAMA_MODEL=gemma3:1b streamlit run app/streamlit_app.py
```

---

## 八、评审维度与应对策略

| 评审维度 | 我们的应对 |
|----------|-----------|
| **Impact & Vision** | Maria 故事 + Before/After 区块 + 社会量化指标 |
| **Technical Depth** | Architecture 标签页：4 agents + 6 safety rules + privacy boundary |
| **Video Pitch** | 从人物故事出发，不是功能罗列；有具体数字 |
| **Demo & Code** | Judge Demo Mode 确保 100% 可演示；smoke_test 通过 |
| **Local-First** | num_ctx=2048、streaming、Ollama —— 全部本地 |

---

## 九、紧急联系

如果 App 无法启动，检查：
```bash
# 1. 确认在正确目录
pwd  # 应该是 gemma-4-good-agentic-data-foundry/

# 2. 确认依赖已安装
pip install -r requirements.txt

# 3. 清除 Python 缓存
find src -name "*.pyc" -delete

# 4. 重启
streamlit run app/streamlit_app.py
```

Smoke test 失败时：
```bash
python3 scripts/smoke_test.py
# 如果失败，说明 src/ 路径问题，确认 venv 已激活
```

---

*最后更新：2026-05-17 | 仓库：xuanyichen888/gemma-4-good-agentic-data-foundry*
