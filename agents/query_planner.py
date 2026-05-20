import json
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

from config import DEEPSEEK_API_KEY, LLM_BASE_URL, LLM_MODEL
from agents.state import AgentState


class QueryPlan(BaseModel):
    news_query: str
    financial_query: str
    macro_query: str


def query_planner(state: AgentState) -> AgentState:
    company = state["company_name"]

    prompt = (
        f"你是一个金融分析师。给定公司名称\"{company}\"，生成三个专门的检索查询：\n"
        "1. 新闻查询：关注最近3个月的重大事件、市场动态、舆情\n"
        "2. 财报查询：关注财务指标，如营收、利润、毛利率、负债等\n"
        "3. 宏观查询：关注该公司所在行业的政策、市场环境、竞争格局\n"
        "直接返回 JSON，格式：{\"news_query\": \"...\", \"financial_query\": \"...\", \"macro_query\": \"...\"}，不要其他内容。"
    )

    llm = ChatOpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=LLM_BASE_URL,
        model=LLM_MODEL,
        temperature=0,
    )

    response = llm.invoke(prompt)
    raw = response.content.strip()

    # 去掉可能的 markdown 代码块包裹
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    data = json.loads(raw)
    plan = QueryPlan(**data)

    return {
        "news_query":      plan.news_query,
        "financial_query": plan.financial_query,
        "macro_query":     plan.macro_query,
    }
