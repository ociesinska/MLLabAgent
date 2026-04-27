from langgraph.graph import END, START, StateGraph

from ml_lab_agent.api.agents.chat_graph.nodes import (
    compare_for_summary_node,
    compare_node,
    fallback_summary_node,
    parse_input_node,
    route_after_summary,
    route_after_validate,
    route_by_intent,
    show_best_run_node,
    show_node,
    summarize_compare_node,
    unknown_node,
    validate_compare_node,
)
from ml_lab_agent.api.agents.chat_graph.state import State

graph_builder = StateGraph(State)

graph_builder.add_node("parse_input_node", parse_input_node)
graph_builder.add_node("show_node", show_node)
graph_builder.add_node("show_best_run_node", show_best_run_node)
graph_builder.add_node("compare_node", compare_node)
graph_builder.add_node("compare_for_summary_node", compare_for_summary_node)
graph_builder.add_node("summarize_compare_node", summarize_compare_node)
graph_builder.add_node("fallback_summary_node", fallback_summary_node)
graph_builder.add_node("unknown_node", unknown_node)
graph_builder.add_node("validate_compare_node", validate_compare_node)

graph_builder.add_edge(START, "parse_input_node")
graph_builder.add_conditional_edges(
    "parse_input_node",
    route_by_intent,
    {
        "show_node": "show_node",
        "compare_path": "validate_compare_node",
        "summarize_path": "validate_compare_node",
        "unknown_node": "unknown_node",
        "show_best_run_node": "show_best_run_node",
    },
)

graph_builder.add_edge("show_node", END)
graph_builder.add_edge("unknown_node", END)
graph_builder.add_edge("show_best_run_node", END)

graph_builder.add_conditional_edges(
    "validate_compare_node",
    route_after_validate,
    path_map={"compare_node": "compare_node", "compare_for_summary_node": "compare_for_summary_node", "end": END},
)


graph_builder.add_conditional_edges(
    "summarize_compare_node",
    route_after_summary,
    {
        "fallback_summary_node": "fallback_summary_node",
        "end_after_summary": END,
    },
)

graph_builder.add_edge("compare_node", END)
graph_builder.add_edge("compare_for_summary_node", "summarize_compare_node")
graph_builder.add_edge("fallback_summary_node", END)


graph = graph_builder.compile()
