try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

from agents.state import AgentState, EvidenceItem


def news_agent(state: AgentState) -> AgentState:
    query = state["news_query"]

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5))

    if results:
        content = "\n\n".join(
            f"[{r.get('title', '')}]\n{r.get('body', '')}\n来源: {r.get('href', '')}"
            for r in results
        )
        sources = [r.get("href", "") for r in results if r.get("href")]
    else:
        content = "未找到相关新闻。"
        sources = []

    evidence: EvidenceItem = {
        "content": content,
        "source": "; ".join(sources),
        "quality_score": 0.5,
        "retrieval_type": "news",
    }

    state["news_evidence"] = evidence
    return state
