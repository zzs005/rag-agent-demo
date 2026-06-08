# KB Agent — 知识库智能问答系统

基于 RAG（Retrieval-Augmented Generation）架构的知识库问答 Agent，支持 PDF 文档上传、自动分块索引、混合检索与 LLM 驱动的多轮对话问答。

## 项目架构

```
kb-agent/
├── app/                        # 后端应用主体
│   ├── main.py                 # FastAPI 应用入口，注册路由
│   │
│   ├── api/                    # REST API 层
│   │   ├── chat.py             # POST /chat       — 多轮对话问答
│   │   ├── search.py           # POST /search     — 独立检索调试接口
│   │   ├── upload.py           # POST /upload     — PDF 上传 + 自动重建索引
│   │   ├── reindex.py          # POST /reindex    — 手动重建索引
│   │   ├── deps.py             # 依赖注入（单例：SearchService / Answerer / Memory / LLM）
│   │   └── schemas.py          # API 请求/响应 Pydantic 模型
│   │
│   ├── agent/                  # Agent 编排层
│   │   ├── orchestrator.py     # 核心编排器：协调检索→澄清→回答的完整流程
│   │   ├── clarifier.py        # 澄清引擎：判断问题是否模糊，生成追问
│   │   ├── planner.py          # 查询计划器：查询改写、子问题分解、多轮检索策略
│   │   ├── memory.py           # 会话记忆：内存中存储多轮对话历史
│   │   └── schemas.py          # Agent 内部数据结构（检索计划、对话轮次等）
│   │
│   ├── ingestion/              # 文档摄取管道
│   │   ├── pipeline.py         # 摄取主流程：加载→清洗→分块→保存
│   │   ├── loaders.py          # PDF 文档加载器（基于 pypdf）
│   │   ├── cleaner.py          # 文本清洗（规范化换行、压缩空格、去除页眉页脚伪影）
│   │   ├── chunker.py          # 句子感知的智能分块器（默认 1500 字，250 重叠）
│   │   └── schemas.py          # 文档/页面/Chunk 数据模型
│   │
│   ├── retrieval/              # 检索子系统
│   │   ├── search_service.py   # 检索服务入口：向量 + 关键词混合检索 + 重排序
│   │   ├── embedder.py         # 本地嵌入模型（BGE-small-zh-v1.5）
│   │   ├── vector_store.py     # 基于 NumPy 的向量存储（点积相似度检索）
│   │   ├── keyword_store.py    # 关键词匹配存储（中英文混合分词）
│   │   ├── hybrid_search.py    # 混合检索融合（向量 0.7 / 关键词 0.3 加权）
│   │   ├── reranker.py         # Cross-Encoder 重排序（BGE-reranker-base）
│   │   ├── load_chunks.py      # 从 JSONL 加载分块数据
│   │   └── schemas.py          # RetrievedChunk 数据模型
│   │
│   ├── qa/                     # 问答模块
│   │   ├── answerer.py         # 结构化回答器：裁判拒绝 + 模板/LLM 回答
│   │   ├── answer_builder.py   # 回答构建器（模板模式 + LLM 模式）
│   │   ├── refusal.py          # 规则裁判：基于分数阈值判断证据是否充分
│   │   ├── context_builder.py  # 证据上下文组装
│   │   ├── response_formatter.py # 响应格式化（grounded/refusal/clarification 三种类型）
│   │   └── schemas.py          # QAResponse / Citation 数据模型
│   │
│   └── services/               # 通用服务
│       ├── llm_service.py      # LLM 服务封装（OpenAI 兼容接口，默认 DeepSeek）
│       └── index_service.py    # 索引重建服务
│
├── frontend/
│   └── streamlit_app.py        # Streamlit 前端界面（上传/对话/检索调试三标签页）
│
├── scripts/                    # 命令行工具脚本
│   ├── build_index.py          # 构建向量索引
│   ├── build_keyword_index.py  # 构建关键词元数据索引
│   ├── ingest_docs.py          # 批量摄取 PDF 文档
│   ├── chat_debug.py           # 命令行交互式聊天调试
│   └── search_debug.py         # 命令行检索调试（分阶段输出向量/混合/重排结果）
│
├── tests/
│   ├── test_chunker.py         # 分块器单元测试
│   └── test_agent_orchestrator.py  # Agent 编排器测试
│
├── data/                       # 数据目录（被 .gitignore 排除）
│   ├── raw/                    # 原始上传的 PDF 文件
│   ├── processed/              # 处理后的 chunks JSONL 文件
│   └── index/                  # 向量索引 (.npy) + 元数据 (.json)
│
├── .env                        # 环境变量配置
├── requirements.txt            # Python 依赖
└── .gitignore
```

## 核心工作流程

```
用户提问
    │
    ▼
┌──────────────────────────────────────────────┐
│  1. Clarifier.decide_before_search()         │
│     检查问题是否过短/模糊/缺少上下文            │
│     → 如果是，返回澄清追问                     │
└──────────────────┬───────────────────────────┘
                   │ 问题通过
    ┌──────────────▼───────────────────────────┐
    │  2. QueryPlanner.build_plan()             │
    │     LLM 改写追问 / 拆解复合问题            │
    └──────────────┬───────────────────────────┘
                   │
    ┌──────────────▼───────────────────────────┐
    │  3. SearchService.search()                │
    │     Embedder → Vector + Keyword Hybrid    │
    │     → Reranker → Final Top-K              │
    └──────────────┬───────────────────────────┘
                   │
    ┌──────────────▼───────────────────────────┐
    │  4. RefusalJudge.judge()                  │
    │     基于分数阈值判断证据是否可用            │
    │     → 不足则返回拒绝回答                    │
    └──────────────┬───────────────────────────┘
                   │ 证据通过
    ┌──────────────▼───────────────────────────┐
    │  5. LLMService.generate_answer()          │
    │     基于证据片段生成带引用的答案            │
    └──────────────────────────────────────────┘
```

### 检索策略

- **向量检索**：使用 `BAAI/bge-small-zh-v1.5` 模型，对中文优化
- **关键词检索**：中英文混合分词，词频匹配
- **混合融合**：向量 0.7 + 关键词 0.3 加权合并
- **重排序**：使用 `BAAI/bge-reranker-base` Cross-Encoder 精排
- **迭代检索**：最多 2 轮，首轮无结果时 LLM 自动简化查询重试

### 答案类型

| 类型 | 说明 | 触发条件 |
|------|------|----------|
| `clarification` | 反问澄清 | 问题过于模糊、指代不明、缺少上下文 |
| `refusal` | 拒绝回答 | 证据评分低于阈值、检索无结果 |
| `grounded_answer` | 基于证据的答案 | 证据评分达标 |

## 快速开始

### 环境要求

- Python 3.9+
- pip

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：
| 包名 | 版本 | 用途 |
|------|------|------|
| `fastapi` | ≥0.110 | Web 框架 |
| `uvicorn` | ≥0.27 | ASGI 服务器 |
| `streamlit` | ≥1.32 | 前端界面 |
| `sentence-transformers` | ≥3.0.1 | 文本嵌入 & 重排序 |
| `numpy` | ≥1.26.4 | 向量存储与计算 |
| `pypdf` | ≥4.2.0 | PDF 解析 |
| `openai` | ≥1.0.0 | LLM API 调用（OpenAI 兼容接口） |
| `pydantic` | ≥2.7.0 | 数据模型校验 |
| `python-dotenv` | ≥1.0.1 | 环境变量管理 |

### 2. 配置环境变量

编辑项目根目录的 `.env` 文件：

```env
# 嵌入模型（可替换为其他 BGE/M3E 系列模型）
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5

# 检索参数
TOP_K=5
RETRIEVE_CANDIDATES=12

# 重排序模型
RERANK_MODEL=BAAI/bge-reranker-base

# LLM 配置（OpenAI 兼容接口，此处以 DeepSeek 为例）
LLM_API_BASE=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
LLM_API_KEY=sk-your-api-key-here
```

> **说明**：LLM 是可选的。不配置 `LLM_API_KEY` 时，系统会使用基于模板的答案生成器，但查询改写、追问生成、关键词提取等高级功能将不可用。

### 3. 准备文档

将 PDF 文档放入 `data/raw/` 目录：

```bash
mkdir -p data/raw
cp /path/to/your/documents/*.pdf data/raw/
```

### 4. 文档摄取 & 构建索引

```bash
# 步骤一：摄取 PDF → 分块 → 保存为 JSONL
python scripts/ingest_docs.py

# 步骤二：构建向量索引
python scripts/build_index.py

# 步骤三：构建关键词元数据索引
python scripts/build_keyword_index.py
```

或者通过 API 一步完成（上传即自动重建索引，见下文）。

### 5. 启动后端

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

API 文档自动生成，启动后访问：
- Swagger UI: http://127.0.0.1:8000/docs
- Health Check: http://127.0.0.1:8000/health

### 6. 启动前端（可选）

```bash
streamlit run frontend/streamlit_app.py
```

浏览器访问 http://localhost:8501，提供三个功能标签页：
- **Upload** — 上传 PDF 并自动重建索引
- **Agent Chat** — 多轮知识库对话，展示引用来源和 Agent 内部追踪
- **Search Debug** — 检索调试，查看命中结果和分数详情

## API 接口一览

### `GET /` — 根路径
返回服务状态。

### `GET /health` — 健康检查
```json
{"status": "ok"}
```

### `POST /upload` — 上传 PDF
- Content-Type: `multipart/form-data`
- 参数: `file` (PDF 文件)
- 上传后自动完成文档摄取 + 索引重建

```bash
curl -X POST http://127.0.0.1:8000/upload \
  -F "file=@document.pdf"
```

### `POST /search` — 知识库检索
```json
{
  "query": "奖学金评定标准是什么？",
  "top_k": 5
}
```

### `POST /chat` — 多轮对话问答
```json
{
  "query": "申请提前离校的条件是什么？",
  "top_k": 5,
  "conversation_id": "abc123"  // 可选，不传则自动创建新会话
}
```

响应示例：
```json
{
  "query": "申请提前离校的条件是什么？",
  "answer": "根据《2025学生手册》第X条规定...",
  "answer_type": "grounded_answer",
  "confidence": "high",
  "citations": [
    {
      "source": "2025学生手册.pdf",
      "page_start": 12,
      "page_end": 13,
      "chunk_id": "2025学生手册_abc12345_chunk_0023",
      "section_title": "第三章 学生日常管理规定"
    }
  ],
  "used_chunks": ["...chunk_0023", "..."],
  "refused": false,
  "refusal_reason": null,
  "conversation_id": "abc123",
  "needs_clarification": false,
  "clarification_question": null,
  "agent_trace": [
    "Round 1 retrieval plan: 直接检索用户问题。",
    "Executed search for '申请提前离校的条件' -> 5 results.",
    "LLM synthesized answer from retrieved evidence."
  ]
}
```

### `POST /reindex` — 手动重建索引
从 `data/processed/` 中所有 JSONL 文件重建向量索引。

## 脚本工具

### 文档摄取

```bash
python scripts/ingest_docs.py
```
遍历 `data/raw/` 中所有 PDF，执行 Load → Clean → Chunk → 保存到 `data/processed/`。

### 构建索引

```bash
# 向量索引（embedding + numpy 保存）
python scripts/build_index.py

# 关键词元数据索引
python scripts/build_keyword_index.py
```

### 交互式调试

```bash
# 检索调试：分别展示向量检索、混合检索、重排序三阶段结果
python scripts/search_debug.py

# 聊天调试：交互式问答 + 展示结构化响应和检索结果
python scripts/chat_debug.py
```
输入 `exit` 或 `quit` 退出。

## 运行测试

```bash
# 运行分块器测试
python -m pytest tests/test_chunker.py -v

# 或使用 unittest
python -m unittest tests.test_chunker -v
```

## 关键设计决策

### 分块策略
- 默认 1500 字符 / 250 字符重叠
- **句子感知**：优先在句子边界切分，保持语义完整性
- **成对定界符感知**：引号、书名号等成对标点内部不会被切分
- 超长句子回退到逗号/顿号等二级断点切分

### 检索流程
- **混合检索**：语义向量 (70%) + 关键词匹配 (30%)，互补长短
- **Cross-Encoder 重排序**：对候选结果做精细的相关性评估
- **迭代检索**：首轮无结果时 LLM 自动简化查询重试（最多 2 轮）
- **早停机制**：top1 得分 ≥ 0.9 时提前返回，不再继续检索

### 回答安全
- **规则裁判**：基于 top1 分数、top3 均分、分数差距三个维度的规则判断
- **拒绝阈值**：top1 < 0.58 或 top3 均分 < 0.52 时拒绝回答
- **模糊检测**：top1 < 0.65 且 top1-top2 < 0.03 时判定为模糊结果
- **置信度分级**：high / medium / low 三级

### 会话管理
- 基于 `conversation_id` 的多轮对话
- 内存存储（重启丢失），最多保留 8 轮历史
- 自动检测追问指代词（"这个"、"那个"、"它" 等）并结合历史改写

## 技术栈

| 层级 | 技术 |
|------|------|
| Web 框架 | FastAPI + Uvicorn |
| 前端 | Streamlit |
| 嵌入模型 | BAAI/bge-small-zh-v1.5 |
| 重排序 | BAAI/bge-reranker-base |
| 向量存储 | NumPy 内存存储 (dot product) |
| LLM | DeepSeek (兼容 OpenAI API) |
| PDF 解析 | pypdf |
| 数据校验 | Pydantic v2 |
