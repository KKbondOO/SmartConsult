import asyncio
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt
from medgemma.gradio_chatbot.config.settings import MEDGEMMA_MODEL_CONFIG, QUESTIONER_MODEL_CONFIG
from medgemma.gradio_chatbot.config.prompts import DECISION_PROMPT, QUESTIONER_PROMPT, SUMMARY_PROMPT, ADVICE_SYSTEM_PROMPT
from medgemma.gradio_chatbot.graph.state import CustomFlowState
from medgemma.gradio_chatbot.utils.text_utils import clean_markdown
from medgemma.gradio_chatbot.tools.agent import get_agent

# Initialize models
medgemma_model = ChatOpenAI(**MEDGEMMA_MODEL_CONFIG)
questioner_model = ChatOpenAI(**QUESTIONER_MODEL_CONFIG)

# Node Definitions

async def medgemma_decision(state: CustomFlowState):
    """
    Decision Node: 分析对话历史，决定是继续提问还是给出建议
    """
    filtered_messages = []
    for msg in state["messages"]:
        if isinstance(msg, (HumanMessage, AIMessage)):
            filtered_messages.append(msg)
    
    if filtered_messages and isinstance(filtered_messages[-1], AIMessage):
        filtered_messages.append(HumanMessage(content="[继续分析]"))
    
    messages = filtered_messages + [SystemMessage(content=DECISION_PROMPT)]
    
    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        try:
            response = await medgemma_model.ainvoke(messages)
            content = response.content.strip().upper()
            
            if "ADVICE" in content:
                print(f"✅ 决策验证通过 (尝试 {attempt + 1}): ADVICE")
                return {"decision_result": "ADVICE"}
            elif "QUESTION" in content:
                print(f"✅ 决策验证通过 (尝试 {attempt + 1}): QUESTION")
                return {"decision_result": "QUESTION"}
            else:
                print(f"⚠️ 决策验证失败 (尝试 {attempt + 1}/{MAX_RETRIES}): {response.content}")
                if attempt < MAX_RETRIES - 1:
                    continue
                    
        except Exception as e:
            print(f"❌ 决策调用失败 (尝试 {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                continue
    
    print("⚠️ 所有重试失败，使用默认决策: QUESTION")
    return {"decision_result": "QUESTION"}

async def question_node(state: CustomFlowState):
    """
    Questioner Node: 提出一个关键问题或调用工具
    """
    ai_agent = await get_agent()
    messages = [SystemMessage(content=QUESTIONER_PROMPT)] + state["messages"]
    
    agent_input = {"messages": messages}
    agent_response = await ai_agent.ainvoke(agent_input)
    
    new_count = state.get("question_count", 0) + 1
    
    return {
        "messages": agent_response.get("messages", []),
        "question_count": new_count
    }

async def summary_node(state: CustomFlowState):
    """
    Summary Node: 生成患者病情信息摘要
    """
    filtered_messages = []
    for msg in state["messages"]:
        if isinstance(msg, (HumanMessage, AIMessage)):
            filtered_messages.append(msg)
    
    if filtered_messages and isinstance(filtered_messages[-1], AIMessage):
        filtered_messages.append(HumanMessage(content="[继续分析]"))
        
    messages = [SystemMessage(content=SUMMARY_PROMPT)] + filtered_messages
    
    response = await questioner_model.ainvoke(messages)
    return {
        "patient_summary": response.content
    }

async def edit_summary_node(state: CustomFlowState):
    """
    Edit Summary Node: 人工审核和编辑摘要
    """
    edited_summary = interrupt({
        "instruction": "请审核并编辑患者病情信息摘要。如果信息准确完整，可以直接继续；如果需要修改，请编辑后提交。",
        "summary": state["patient_summary"]
    })
    
    if edited_summary is None:
        edited_summary = state["patient_summary"]
    
    return {
        "patient_summary": edited_summary
    }

async def advice_node(state: CustomFlowState):
    """
    Advice Node: 基于患者病情摘要生成最终医疗建议
    """
    patient_summary = state.get("patient_summary", "")
    
    if patient_summary:
        clean_summary = clean_markdown(patient_summary)
        user_request = f"""【患者病情摘要】
{clean_summary}

请基于以上信息给出诊疗建议。"""
        
        messages = [
            SystemMessage(content=ADVICE_SYSTEM_PROMPT),
            HumanMessage(content=user_request)
        ]
    else:
        messages = [SystemMessage(content=ADVICE_SYSTEM_PROMPT)]
        filtered_messages = []
        for msg in state["messages"]:
            if isinstance(msg, (HumanMessage, AIMessage)):
                filtered_messages.append(msg)
        if filtered_messages and isinstance(filtered_messages[-1], AIMessage):
            filtered_messages.append(HumanMessage(content="[继续分析]"))
        messages = messages + filtered_messages

    response = await medgemma_model.ainvoke(messages)
    return {"messages": [response], "advice": response.content}
