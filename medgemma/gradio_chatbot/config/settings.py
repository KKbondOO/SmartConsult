import os
from pydantic import SecretStr
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# ==================== LangSmith Configuration ====================
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "consultation-flow")
os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

# ==================== API Keys ====================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY 未设置！请在 .env 文件中配置此环境变量。")

# ==================== Model Configurations ====================
QUESTIONER_MODEL_CONFIG = {
    "model": "z-ai/glm-4.5-air:free",
    "api_key": SecretStr(OPENROUTER_API_KEY),
    "base_url": "https://openrouter.ai/api/v1",
    "timeout": 60,
    "max_retries": 2,
}

MEDGEMMA_MODEL_CONFIG = {
    "api_key": SecretStr(os.getenv("MEDGEMMA_MODEL", "unsloth/medgemma-4b-it-bnb-4bit")),
    "base_url": os.getenv("MEDGEMMA_BASE_URL", "http://localhost:8000/v1"),
    "model": os.getenv("MEDGEMMA_MODEL", "unsloth/medgemma-4b-it-bnb-4bit"),
    "temperature": 0.9,
    "max_tokens": 2048,
}

FALLBACK_MODEL_CONFIG = {
    "model": "amazon/nova-2-lite-v1:free",
    "api_key": SecretStr(OPENROUTER_API_KEY),
    "base_url": "https://openrouter.ai/api/v1",
    "timeout": 60,
    "max_retries": 2,
}

# ==================== MCP Configuration ====================
MCP_MEDICAL_PATH = os.getenv(
    "MCP_MEDICAL_PATH", 
    r"C:\Users\27160\Desktop\softeng\medical-mcp\build\index.js"
)

MCP_CONFIG = {
    "medical_query": {
        "transport": "stdio",
        "command": "node",
        "args": [MCP_MEDICAL_PATH],
    },
}

# ==================== Kokoro TTS Paths ====================
KOKORO_VOICES_DIR = os.getenv(
    "KOKORO_VOICES_DIR",
    r"D:\hf_hub\kokoro\Kokoro-82M-v1.1-zh\voices"
)
KOKORO_MODEL_PATH = os.getenv(
    "KOKORO_MODEL_PATH",
    "D:/hf_hub/kokoro/Kokoro-82M-v1.1-zh/kokoro-v1_1-zh.pth"
)
KOKORO_CONFIG_PATH = os.getenv(
    "KOKORO_CONFIG_PATH",
    "D:/hf_hub/kokoro/Kokoro-82M-v1.1-zh/config.json"
)
KOKORO_REPO_ID = os.getenv(
    "KOKORO_REPO_ID",
    "hexgrad/Kokoro-82M-v1.1-zh"
)

# ==================== Session Configuration ====================
DEFAULT_SESSION_ID = "default_session"
MAX_QUESTIONS = 10
