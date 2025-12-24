from typing import Annotated, Callable
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import ToolRetryMiddleware
from langchain_openai import ChatOpenAI
from langgraph.graph import add_messages
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel
from ..config.settings import QUESTIONER_MODEL_CONFIG, FALLBACK_MODEL_CONFIG
from mcp_client import get_tools

# Models
model = ChatOpenAI(**QUESTIONER_MODEL_CONFIG)
fallback_model_instance = ChatOpenAI(**FALLBACK_MODEL_CONFIG)

class ContextSchema(BaseModel):
    user_name: str

class CustomState(AgentState):
    messages: Annotated[list, add_messages]

def wrap_model_call(state_schema=None):
    """Simple wrapper for model calls (placeholder for logic in mcp_agent.py)"""
    def decorator(fn):
        fn._state_schema = state_schema
        return fn
    return decorator

@wrap_model_call(state_schema=CustomState)
async def fallback_model(request, handler: Callable):
    """异步版本的 fallback model middleware"""
    try:
        return await handler(request)
    except Exception:
        pass
    request = request.override(model=fallback_model_instance)
    return await handler(request)

# Agent checkpointer
checkpointer = MemorySaver()

async def get_agent():
    tools = await get_tools()
    agent = create_agent(
        model,
        tools=tools,
        context_schema=ContextSchema,
        checkpointer=checkpointer,
        middleware=[
            fallback_model,
            ToolRetryMiddleware(
                max_retries=3,
                backoff_factor=2.0,
                initial_delay=1.0,
                on_failure='continue'
            ),
        ]
    )
    return agent
