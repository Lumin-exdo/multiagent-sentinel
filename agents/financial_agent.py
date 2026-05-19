from rag.retriever import hybrid_search
from agents.state import AgentState, EvidenceItem


def financial_agent(state: AgentState) -> AgentState:
    query = state["financial_query"]
    chunks = hybrid_search(query, collection_name="financials", top_k=5)

    if chunks:
        content = "\n\n".join(chunks)
    else:
        content = "知识库中未找到相关财报数据。"

    evidence: EvidenceItem = {
        "content": content,
        "source": "financials knowledge base",
        "quality_score": 0.5,
        "retrieval_type": "financial",
    }

    state["financial_evidence"] = evidence
    return state
