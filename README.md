# Multiagent Sentinel

多智能体金融舆情分析系统——基于 LangGraph + RAG，对上市公司进行三路并行检索、CRAG 纠错、跨源矛盾检测，最终输出结构化分析报告。

## 系统架构

```mermaid
graph TD
    START --> query_planner

    query_planner --> news_agent
    query_planner --> financial_agent
    query_planner --> macro_agent

    news_agent      --> evidence_evaluator
    financial_agent --> evidence_evaluator
    macro_agent     --> evidence_evaluator

    evidence_evaluator -->|retry_news|      news_agent
    evidence_evaluator -->|retry_financial| financial_agent
    evidence_evaluator -->|retry_macro|     macro_agent
    evidence_evaluator -->|continue|        contradiction_detector

    contradiction_detector --> report_writer
    report_writer --> END
```

```
START
  │
  ▼
query_planner          ← LLM 生成三路专属查询
  │         │         │
  ▼         ▼         ▼
news      financial  macro    ← 并行检索
agent     agent      agent
  │         │         │
  └────┬────┘─────────┘
       ▼
evidence_evaluator     ← CRAG：LLM 打质量分，分 < 0.5 则扩查重试（最多 2 次）
       │
  ┌────┴──── retry → 对应 agent
  ▼
contradiction_detector ← 跨源矛盾检测
       │
       ▼
report_writer          ← 生成结构化 JSON 报告
       │
      END
```

## 技术栈

| 组件 | 说明 |
|------|------|
| **LangGraph** | 多智能体编排，支持并行 fan-out / fan-in 与条件边 |
| **LangChain** | LLM 调用、文档加载、文本分割 |
| **DeepSeek API** | LLM 后端（`deepseek-chat`），通过 OpenAI 兼容接口调用 |
| **ChromaDB** | 向量数据库，持久化存储财报与宏观政策文本 |
| **BGE-small-zh** | 中文嵌入模型（`BAAI/bge-small-zh-v1.5`），本地运行 |
| **Serper API** | Google 新闻搜索（`gl=cn&hl=zh-cn`） |
| **BM25 + RRF** | 混合检索：关键词 BM25 + 向量语义，倒数排名融合 |
| **pdfminer.six** | PDF 年报文本提取 |

## 核心功能

### 三路并行检索
- **新闻路**：Serper API 实时搜索（Google News），中文结果限定
- **财报路**：RAG 混合检索（BM25 + 向量），检索本地年报知识库
- **宏观路**：RAG 混合检索，检索政策文件知识库

### CRAG 纠错（Corrective RAG）
每路检索完成后，LLM 对内容相关性打分（0–1）。分数 < 0.5 时自动扩宽查询词（追加"相关信息 最新"）并重新检索，最多重试 2 次。

### 跨源矛盾检测
三路证据汇总后，LLM 识别材料间的逻辑矛盾（如：政策利好 vs 财报利润下滑、新闻舆情负面 vs 官方数据正向）并结构化输出。

## 快速开始

### 1. 环境配置

```bash
conda create -n multiagent-sentinel python=3.12
conda activate multiagent-sentinel
pip install -r requirements.txt
```

复制并填写环境变量：

```bash
cp .env.example .env   # 或手动创建 .env
```

`.env` 内容：

```
DEEPSEEK_API_KEY=your_deepseek_key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
CHROMA_PATH=./chroma_db
SERPER_API_KEY=your_serper_key
```

### 2. 准备知识库

将财报 PDF（或 txt）放入 `knowledge_base/financials/`，政策文本放入 `knowledge_base/macro/`。

从 PDF 提取文本（可选）：

```bash
python -c "
import pdfminer.high_level
text = pdfminer.high_level.extract_text('年报.pdf')
open('knowledge_base/financials/company_2024.txt','w').write(text[:50000])
"
```

首次运行前加载向量库：

```bash
python main.py 宁德时代 --load-docs
```

### 3. 运行分析

```bash
# 分析单家公司
python main.py 宁德时代

# 保存结果
python main.py 比亚迪 > results/byd_result.json
```

### 4. 可视化图结构

```bash
python visualize_graph.py   # 输出 results/graph.png
```

## 示例输出

以下为宁德时代分析结果（实际运行）：

```json
{
  "company": "宁德时代",
  "overall_sentiment": "bullish",
  "confidence": 0.8,
  "evidence_quality": {
    "news": 0.0,
    "financial": 0.5,
    "macro": 0.3
  },
  "contradictions": [],
  "summary": "宁德时代作为全球动力电池龙头，2025年总资产同比增长23.92%，净资产增长36.52%，
财务基础稳固。宏观层面，新能源汽车产业政策明确2025年销量占比20%目标，财政部延续免征购置税
至2025年底，直接利好下游需求；同时动力电池回收新政强化龙头企业在回收体系中的优势地位。
中长期发展前景乐观。",
  "sources": [
    "宁德时代2024年年度报告",
    "宁德时代2026年第一季度报告",
    "六部门《新能源汽车废旧动力电池回收和综合利用管理暂行办法》",
    "工信部《新能源汽车产业发展规划》及财政部免征购置税政策"
  ]
}
```

## 项目结构

```
multiagent-sentinel/
├── agents/
│   ├── state.py               # LangGraph 共享状态定义
│   ├── graph.py               # 图结构组装
│   ├── query_planner.py       # LLM 生成三路查询
│   ├── news_agent.py          # Serper 新闻检索
│   ├── financial_agent.py     # RAG 财报检索
│   ├── macro_agent.py         # RAG 宏观政策检索
│   ├── evidence_evaluator.py  # CRAG 质量评估 + 路由
│   ├── contradiction_detector.py  # 跨源矛盾检测
│   └── report_writer.py       # 结构化报告生成
├── rag/
│   ├── loader.py              # 文档加载 & 向量化入库
│   └── retriever.py           # BM25 + 向量混合检索（RRF 融合）
├── knowledge_base/
│   ├── financials/            # 财报 txt
│   └── macro/                 # 政策文件 txt
├── results/                   # 分析结果输出
├── main.py                    # 入口
├── config.py                  # 环境变量读取
├── visualize_graph.py         # 图结构可视化
└── requirements.txt
```

## License

MIT
