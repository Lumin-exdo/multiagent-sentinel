import json
from langchain_openai import ChatOpenAI

from config import DEEPSEEK_API_KEY, LLM_BASE_URL, LLM_MODEL
from agents.state import AgentState, AnalysisReport


def report_writer(state: AgentState) -> AgentState:
    company = state["company_name"]

    news_ev      = state.get("news_evidence")      or {}
    financial_ev = state.get("financial_evidence") or {}
    macro_ev     = state.get("macro_evidence")     or {}
    contradictions = state.get("contradictions", [])

    prompt = (
        f"基于以下信息，生成一份金融分析报告：\n"
        f"公司：{company}\n"
        f"新闻摘要：{news_ev.get('content','无数据')[:1500]}\n"
        f"财报信息：{financial_ev.get('content','无数据')[:1500]}\n"
        f"宏观环境：{macro_ev.get('content','无数据')[:1500]}\n"
        f"检测到的矛盾：{contradictions}\n\n"
        "请返回 JSON 格式（全部使用双引号）：\n"
        '{"overall_sentiment":"bullish/bearish/neutral",'
        '"confidence":0.0到1.0之间的数字,'
        '"evidence_quality":{"news":数字,"financial":数字,"macro":数字},'
        '"contradictions":[...],'
        '"summary":"200字以内的综合分析",'
        '"sources":["来源1","来源2"]}'
    )

    llm = ChatOpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=LLM_BASE_URL,
        model=LLM_MODEL,
        temperature=0.3,
    )

    response = llm.invoke(prompt)
    raw = response.content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    data = json.loads(raw)

    # 从 evidence 中汇总来源列表
    all_sources: list[str] = []
    for ev in [news_ev, financial_ev, macro_ev]:
        for s in ev.get("source", "").split(";"):
            s = s.strip()
            if s:
                all_sources.append(s)

    report: AnalysisReport = {
        "company": company,
        "overall_sentiment": data.get("overall_sentiment", "neutral"),
        "confidence": float(data.get("confidence", 0.5)),
        "evidence_quality": {
            "news":      news_ev.get("quality_score", 0.5),
            "financial": financial_ev.get("quality_score", 0.5),
            "macro":     macro_ev.get("quality_score", 0.5),
        },
        "contradictions": data.get("contradictions", contradictions),
        "summary": data.get("summary", ""),
        "sources": data.get("sources", all_sources),
    }

    return {"report": report}
