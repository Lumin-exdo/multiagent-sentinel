import json
import argparse

from agents.graph import build_graph


def run(company_name: str, load_docs: bool = False) -> None:
    if load_docs:
        from rag.loader import load_all_documents
        print("正在加载知识库文档...")
        load_all_documents()
        print("知识库加载完成。\n")

    graph = build_graph()

    initial_state = {
        "company_name": company_name,
        # query_planner 会填写这三个字段
        "news_query":      "",
        "financial_query": "",
        "macro_query":     "",
        # 三路证据（初始为 None）
        "news_evidence":      None,
        "financial_evidence": None,
        "macro_evidence":     None,
        # CRAG 重试计数
        "news_retry":      0,
        "financial_retry": 0,
        "macro_retry":     0,
        # 路由信号
        "needs_retry": [],
        # 矛盾 & 报告
        "contradictions": [],
        "report": None,
    }

    print(f"开始分析：{company_name}\n{'='*50}")
    result = graph.invoke(initial_state)

    report = result.get("report")
    if report:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("未生成报告，请检查日志。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多智能体金融舆情分析")
    parser.add_argument("company", nargs="?", default="宁德时代", help="公司名称")
    parser.add_argument("--load-docs", action="store_true", help="首次运行时加载知识库")
    args = parser.parse_args()

    run(args.company, load_docs=args.load_docs)
