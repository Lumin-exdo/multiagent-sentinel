import json
from langchain_openai import ChatOpenAI

from config import DEEPSEEK_API_KEY, LLM_BASE_URL, LLM_MODEL
from agents.state import AgentState


def contradiction_detector(state: AgentState) -> AgentState:
    company = state["company_name"]
    news_content     = (state.get("news_evidence")     or {}).get("content", "无数据")
    financial_content = (state.get("financial_evidence") or {}).get("content", "无数据")
    macro_content    = (state.get("macro_evidence")    or {}).get("content", "无数据")

    prompt = (
        f"你是一个金融分析师。以下是关于{company}的三份独立分析：\n\n"
        f"【新闻证据】{news_content}\n\n"
        f"【财报证据】{financial_content}\n\n"
        f"【宏观证据】{macro_content}\n\n"
        "请找出这三份材料中存在的矛盾或不一致之处。\n"
        "例如：新闻说销量增长但财报显示利润下滑；政策利好但公司市场份额下降等。\n"
        "如果没有明显矛盾，返回空列表。\n"
        '只返回 JSON 格式：{"contradictions": ["矛盾1", "矛盾2"]}'
    )

    llm = ChatOpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=LLM_BASE_URL,
        model=LLM_MODEL,
        temperature=0,
    )

    response = llm.invoke(prompt)
    raw = response.content.strip()

    # 去掉可能的 markdown 代码块
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        data = json.loads(raw)
        contradictions = data.get("contradictions", [])
        if not isinstance(contradictions, list):
            contradictions = []
    except (json.JSONDecodeError, AttributeError):
        contradictions = []

    return {"contradictions": contradictions}
