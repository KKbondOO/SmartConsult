from typing import Annotated, TypedDict
from langgraph.graph import add_messages

class CustomFlowState(TypedDict):
    messages: Annotated[list, add_messages]
    question_count: int  # 跟踪提问轮次
    skip_to_advice: bool  # 用户请求直接生成建议
    patient_summary: str  # 患者病情信息摘要
    decision_result: str  # 决策结果: "QUESTION" 或 "ADVICE",不放入messages
