from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from ..graph.state import CustomFlowState
from ..graph.nodes import medgemma_decision, question_node, summary_node, edit_summary_node, advice_node
from ..graph.edges import route_decision, route_after_question

# ==================== 图构建 ====================
workflow = StateGraph(CustomFlowState)

workflow.add_node("medgemma_decision", medgemma_decision)
workflow.add_node("question_node", question_node)
workflow.add_node("summary_node", summary_node)
workflow.add_node("edit_summary_node", edit_summary_node)
workflow.add_node("advice_node", advice_node)

workflow.add_edge(START, "medgemma_decision")

workflow.add_conditional_edges(
    "medgemma_decision",
    route_decision,
    {
        "question_node": "question_node",
        "summary_node": "summary_node"
    }
)

workflow.add_conditional_edges(
    "question_node",
    route_after_question,
    {
        "__end__": END,
        "summary_node": "summary_node"
    }
)

workflow.add_edge("summary_node", "edit_summary_node")
workflow.add_edge("edit_summary_node", "advice_node")
workflow.add_edge("advice_node", END)

# ==================== 编译图 ====================
graph = workflow.compile(checkpointer=MemorySaver())
