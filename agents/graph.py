from langgraph.graph import StateGraph, END, START

from agents.state import AgentState
from agents.query_planner import query_planner
from agents.news_agent import news_agent
from agents.financial_agent import financial_agent
from agents.macro_agent import macro_agent
from agents.evidence_evaluator import evidence_evaluator, route_after_evaluation
from agents.contradiction_detector import contradiction_detector
from agents.report_writer import report_writer


def build_graph():
    workflow = StateGraph(AgentState)

    # ── 节点 ──────────────────────────────────────────────────────────────
    workflow.add_node("query_planner",        query_planner)
    workflow.add_node("news_agent",           news_agent)
    workflow.add_node("financial_agent",      financial_agent)
    workflow.add_node("macro_agent",          macro_agent)
    workflow.add_node("evidence_evaluator",   evidence_evaluator)
    workflow.add_node("contradiction_detector", contradiction_detector)
    workflow.add_node("report_writer",        report_writer)

    # ── 入口 ──────────────────────────────────────────────────────────────
    workflow.add_edge(START, "query_planner")

    # ── fan-out：query_planner → 三路并行检索 ─────────────────────────────
    workflow.add_edge("query_planner", "news_agent")
    workflow.add_edge("query_planner", "financial_agent")
    workflow.add_edge("query_planner", "macro_agent")

    # ── fan-in：三路检索完成后汇入 evaluator ──────────────────────────────
    workflow.add_edge("news_agent",      "evidence_evaluator")
    workflow.add_edge("financial_agent", "evidence_evaluator")
    workflow.add_edge("macro_agent",     "evidence_evaluator")

    # ── CRAG 条件边：质量不足 → 重试对应 agent；全部通过 → 矛盾检测 ────────
    workflow.add_conditional_edges(
        "evidence_evaluator",
        route_after_evaluation,
        {
            "retry_news":      "news_agent",
            "retry_financial": "financial_agent",
            "retry_macro":     "macro_agent",
            "continue":        "contradiction_detector",
        },
    )

    # ── 后续链路 ──────────────────────────────────────────────────────────
    workflow.add_edge("contradiction_detector", "report_writer")
    workflow.add_edge("report_writer", END)

    return workflow.compile()
