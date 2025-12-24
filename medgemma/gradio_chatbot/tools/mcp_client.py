from langchain_mcp_adapters.client import MultiServerMCPClient
from ..config.settings import MCP_CONFIG

client = MultiServerMCPClient(MCP_CONFIG)

async def get_tools():
    tools = await client.get_tools()
    return tools
