import os
import requests

from agents.state import AgentState, EvidenceItem

_SERPER_URL = "https://google.serper.dev/news"


def news_agent(state: AgentState) -> dict:
    query = state["news_query"]
    api_key = os.getenv("SERPER_API_KEY", "")

    try:
        resp = requests.post(
            _SERPER_URL,
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "gl": "cn", "hl": "zh-cn", "num": 5},
            timeout=10,
        )
        resp.raise_for_status()
        items = resp.json().get("news", [])
    except Exception:
        items = []

    if items:
        content = "\n\n".join(
            f"[{r.get('title', '')}]\n{r.get('snippet', '')}\n来源: {r.get('link', '')}"
            for r in items
        )
        sources = [r.get("link", "") for r in items if r.get("link")]
    else:
        content = "未找到相关新闻。"
        sources = []

    evidence: EvidenceItem = {
        "content": content,
        "source": "; ".join(sources),
        "quality_score": 0.5,
        "retrieval_type": "news",
    }

    return {"news_evidence": evidence}
