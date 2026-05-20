from agents.graph import build_graph

graph = build_graph()
png = graph.get_graph().draw_mermaid_png()
with open("results/graph.png", "wb") as f:
    f.write(png)
print("图已保存到 results/graph.png")
