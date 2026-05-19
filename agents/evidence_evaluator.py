from langchain_openai import ChatOpenAI

from config import DEEPSEEK_API_KEY, LLM_BASE_URL, LLM_MODEL
from agents.state import AgentState

_CHECKS = [
    ("news_evidence",      "news_query",      "news_retry"),
    ("financial_evidence", "financial_query", "financial_retry"),
    ("macro_evidence",     "macro_query",     "macro_retry"),
]


def _score_evidence(llm, query: str, content: str) -> float:
    prompt = (
        f"以下文本是否与'{query}'相关且信息充分？"
        f"请只返回 0 到 1 之间的小数。文本：{content[:800]}"
    )
    response = llm.invoke(prompt)
    try:
        return max(0.0, min(1.0, float(response.content.strip())))
    except (ValueError, TypeError):
        return 0.5


def evidence_evaluator(state: AgentState) -> AgentState:
    llm = ChatOpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=LLM_BASE_URL,
        model=LLM_MODEL,
        temperature=0,
    )

    # 每轮重置，由下方逻辑重新填充
    state["needs_retry"] = []

    for ev_key, q_key, retry_key in _CHECKS:
        ev = state.get(ev_key)
        if not ev:
            continue

        score = _score_evidence(llm, state[q_key], ev["content"])
        ev["quality_score"] = score
        state[ev_key] = ev

        current_retry = state.get(retry_key, 0)
        if score < 0.5 and current_retry < 2:
            state[q_key] = state[q_key] + " 相关信息 最新"
            state[retry_key] = current_retry + 1
            state["needs_retry"].append(ev["retrieval_type"])

    return state


def route_after_evaluation(state: AgentState) -> str:
    """LangGraph 条件边路由函数，接在 evidence_evaluator 节点之后。

    返回值对应 graph.py 中 add_conditional_edges 的路径 key：
      "retry_news" | "retry_financial" | "retry_macro" | "continue"
    """
    retry_list = state.get("needs_retry", [])
    if "news" in retry_list:
        return "retry_news"
    if "financial" in retry_list:
        return "retry_financial"
    if "macro" in retry_list:
        return "retry_macro"
    return "continue"
