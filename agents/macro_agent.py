from rag.retriever import hybrid_search
from agents.state import AgentState, EvidenceItem


def macro_agent(state: AgentState) -> AgentState:
    query = state["macro_query"]
    chunks = hybrid_search(query, collection_name="macro", top_k=5)

    if chunks:
        content = "\n\n".join(chunks)
    else:
        content = "知识库中未找到相关宏观政策数据。"

    evidence: EvidenceItem = {
        "content": content,
        "source": "macro knowledge base",
        "quality_score": 0.5,
        "retrieval_type": "macro",
    }

    state["macro_evidence"] = evidence
    return state
