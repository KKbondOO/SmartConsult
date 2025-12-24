from langgraph.graph import END
from ..graph.state import CustomFlowState
from ..config.settings import MAX_QUESTIONS

def route_decision(state: CustomFlowState):
    """
    从决策节点路由到问询节点或摘要节点
    """
    try:
        skip_to_advice = state.get("skip_to_advice", False)
        if skip_to_advice:
            print(f"🚀 检测到 skip_to_advice=True，直接跳转到摘要节点")
            return "summary_node"
        
        decision = state.get("decision_result", "QUESTION")
        if decision == "ADVICE":
            return "summary_node"
        else:
            return "question_node"
    except:
        return "question_node"

def route_after_question(state: CustomFlowState):
    """
    从问询节点路由：检查是否超过最大轮次或用户请求直接生成建议
    """
    question_count = state.get("question_count", 0)
    skip_to_advice = state.get("skip_to_advice", False)
    
    if skip_to_advice:
        print(f"用户请求直接生成建议，跳转到摘要节点")
        return "summary_node"
    
    if question_count >= MAX_QUESTIONS:
        print(f"已达到最大提问轮次 ({MAX_QUESTIONS})，强制进入摘要节点")
        return "summary_node"
    else:
        return "__end__"
