from typing import TypedDict, Optional, Annotated
import operator


class EvidenceItem(TypedDict):
    content: str
    source: str
    quality_score: float  # 0-1
    retrieval_type: str   # "news" / "financial" / "macro"


class AnalysisReport(TypedDict):
    company: str
    overall_sentiment: str  # "bullish" / "bearish" / "neutral"
    confidence: float
    evidence_quality: dict  # {"news": 0.8, "financial": 0.7, "macro": 0.6}
    contradictions: list[str]
    summary: str
    sources: list[str]


class AgentState(TypedDict):
    # 输入
    company_name: str

    # QueryPlanner 输出
    news_query: str
    financial_query: str
    macro_query: str

    # 三路检索结果
    news_evidence: Optional[EvidenceItem]
    financial_evidence: Optional[EvidenceItem]
    macro_evidence: Optional[EvidenceItem]

    # CRAG 重试计数
    news_retry: int
    financial_retry: int
    macro_retry: int

    # CRAG 路由信号：evaluator 填写，路由函数读取，每轮重置
    needs_retry: list[str]  # e.g. ["news", "financial"]

    # 矛盾检测结果
    contradictions: list[str]

    # 最终报告
    report: Optional[AnalysisReport]
