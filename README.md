# 🏥 SmartConsult - AI 智能问诊助手

一个基于 LangGraph 和 Gradio 构建的智能医疗问诊系统，支持文字和语音交互，提供专业的医疗咨询服务。

## ✨ 项目特点

- 🤖 **智能问诊流程**：基于 LangGraph 构建的智能对话流，自动管理问诊流程
- 🎙️ **语音交互**：支持语音输入和语音输出，提供更自然的交互体验
- 💬 **流式响应**：实时流式生成回复，提升用户体验
- 🔄 **人机协同**：支持摘要审核和编辑，确保信息准确性
- 🛠️ **工具集成**：集成 MCP (Model Context Protocol) 工具，支持医疗知识库查询
- 📊 **可追踪性**：集成 LangSmith 进行全流程追踪和调试

## 🏗️ 项目结构

```
SmartConsult/
└── medgemma/
    └── gradio_chatbot/          # 主应用程序
        ├── app.py               # Gradio UI 主程序
        ├── config/              # 配置模块
        │   ├── settings.py      # 系统配置和 API 密钥
        │   └── prompts.py       # 各种提示词模板
        ├── graph/               # LangGraph 图结构
        │   ├── builder.py       # 图构建器
        │   ├── nodes.py         # 图节点定义
        │   ├── edges.py         # 图边和路由逻辑
        │   └── state.py         # 状态定义
        ├── tools/               # 工具模块
        │   ├── agent.py         # Agent 工具集成
        │   └── mcp_client.py    # MCP 客户端
        └── utils/               # 工具函数
            ├── asr.py           # 语音识别
            ├── tts.py           # 语音合成
            └── text_utils.py    # 文本处理工具
```

## 🚀 功能介绍

### 1. 智能问诊流程

系统采用多阶段问诊流程：

- **决策节点**：分析对话历史，判断是否需要继续询问或给出建议
- **提问节点**：根据现有信息生成关键问题
- **工具调用**：支持查询医疗知识库获取专业信息
- **摘要生成**：自动整理病情信息，生成结构化摘要
- **建议生成**：基于收集的信息提供医疗建议

### 2. 语音交互

- **语音输入**：使用 Whisper 模型进行语音识别 (ASR)
- **语音输出**：使用 Kokoro TTS 模型进行中文语音合成
- **流式合成**：支持边生成边播放的流式语音合成

### 3. 用户界面

- **对话历史**：完整的对话记录展示
- **摘要审核**：用户可以查看和编辑 AI 生成的病情摘要
- **快捷操作**：支持"直接生成建议"等快捷功能
- **清空对话**：一键清空对话历史，开始新的会话

## 🛠️ 技术栈

- **前端框架**：Gradio
- **AI 框架**：LangChain / LangGraph
- **语言模型**：
  - 提问模型：z-ai/glm-4.5-air (OpenRouter)
  - 医疗模型：MedGemma-4B (本地部署)
  - 备用模型：Amazon Nova 2 Lite
- **语音技术**：
  - ASR：OpenAI Whisper
  - TTS：Kokoro-82M-v1.1-zh
- **开发工具**：LangSmith (追踪和调试)

## 📋 前置要求

- Python 3.8+
- Node.js (用于 MCP 服务)
- CUDA 支持 (可选，用于加速语音模型)

## 🔧 安装步骤

### 1. 克隆项目

```bash
git clone https://github.com/your-username/SmartConsult.git
cd SmartConsult
```

### 2. 创建虚拟环境

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

> **注意**：确保安装了 `python-dotenv` 库，它用于加载环境变量：
> ```bash
> pip install python-dotenv
> ```

### 4. 配置环境变量

#### 4.1 创建 `.env` 文件

复制 `.env.example` 文件并重命名为 `.env`：

```bash
# Windows
copy .env.example .env

# Linux/Mac
cp .env.example .env
```

#### 4.2 配置 API 密钥和路径

在 `.env` 文件中填写你的配置信息：

```env
# ==================== API Keys ====================
# OpenRouter API Key (必填)
# 获取地址: https://openrouter.ai/keys
OPENROUTER_API_KEY=sk-or-v1-your_actual_openrouter_key_here

# LangSmith API Key (必填)
# 获取地址: https://smith.langchain.com/settings
LANGCHAIN_API_KEY=lsv2_pt_your_actual_langsmith_key_here

# ==================== LangSmith Configuration ====================
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=consultation-flow
LANGSMITH_ENDPOINT=https://api.smith.langchain.com

# ==================== Model Configuration ====================
# MedGemma 本地服务地址（如使用本地模型）
MEDGEMMA_BASE_URL=http://localhost:8000/v1
MEDGEMMA_MODEL=unsloth/medgemma-4b-it-bnb-4bit

# ==================== MCP Configuration ====================
# Medical MCP 服务路径（修改为你的实际路径）
MCP_MEDICAL_PATH=C:\path\to\medical-mcp\build\index.js

# ==================== Kokoro TTS Configuration ====================
# Kokoro 模型路径（修改为你的实际路径）
KOKORO_VOICES_DIR=D:\path\to\kokoro\voices
KOKORO_MODEL_PATH=D:/path/to/kokoro/kokoro-v1_1-zh.pth
KOKORO_CONFIG_PATH=D:/path/to/kokoro/config.json
KOKORO_REPO_ID=hexgrad/Kokoro-82M-v1.1-zh
```

**重要提示**：
- ⚠️ `.env` 文件包含敏感信息，**切勿**提交到 Git 仓库
- ✅ `.env` 已在 `.gitignore` 中排除
- 📝 仅 `.env.example` 模板会被提交到仓库



### 5. 部署 MedGemma 模型服务

项目使用本地部署的 MedGemma 模型，通过 Docker 中的 vLLM 进行加速推理。

#### 5.1 安装前置要求

- **Docker Desktop**（支持 NVIDIA GPU）
- **NVIDIA GPU** 和对应驱动（本项目在 RTX 4060 笔记本上测试通过）
- **足够的 GPU 内存**（建议至少 8GB）

#### 5.2 下载模型

从 Hugging Face 下载量化版本的 MedGemma 模型：

```bash
# 方式 1：使用 huggingface-cli 下载
huggingface-cli download unsloth/medgemma-4b-it-bnb-4bit --local-dir C:\Users\你的用户名\.cache\huggingface

# 方式 2：在 Python 中下载
# python -c "from transformers import AutoModel; AutoModel.from_pretrained('unsloth/medgemma-4b-it-bnb-4bit')"
```

> **注意**：模型会默认下载到 `C:\Users\你的用户名\.cache\huggingface` 目录

#### 5.3 启动 Docker 容器

使用以下命令启动 vLLM 服务：

```bash
docker run --runtime nvidia --gpus all \
  --name Medgemma-4b-it \
  --network bridge \
  -v C:\Users\你的用户名\.cache\huggingface:/root/.cache/huggingface \
  --env HUGGING_FACE_HUB_TOKEN=你的HF_TOKEN \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model unsloth/medgemma-4b-it-bnb-4bit \
  --tokenizer google/medgemma-4b-it \
  --gpu-memory-utilization=0.85 \
  --max-model-len 4096
```

**参数说明**：
- `--runtime nvidia --gpus all`：启用 NVIDIA GPU 支持
- `-v C:\Users\你的用户名\.cache\huggingface:/root/.cache/huggingface`：挂载本地模型缓存目录
- `--env HUGGING_FACE_HUB_TOKEN`：你的 Hugging Face Token（可选，用于下载模型）
- `-p 8000:8000`：映射端口 8000
- `--gpu-memory-utilization=0.85`：使用 85% 的 GPU 内存（根据你的显卡调整）
- `--max-model-len 4096`：最大上下文长度

> **💡 提示**：本配置在 **RTX 4060 笔记本**上测试通过，如果你的显卡显存不足，可以调低 `--gpu-memory-utilization` 参数。

#### 5.4 验证服务

检查服务是否正常运行：

```bash
# 查看容器状态
docker ps

# 测试 API
curl http://localhost:8000/v1/models
```

### 6. 部署 Medical MCP 服务

本项目使用 Medical MCP 服务提供医学知识库查询功能。

#### 6.1 克隆 MCP 项目

```bash
# 克隆开源项目
git clone https://github.com/JamesANZ/medical-mcp.git
cd medical-mcp
```

> **项目地址**：[https://github.com/JamesANZ/medical-mcp](https://github.com/JamesANZ/medical-mcp)

#### 6.2 安装依赖

```bash
# 安装 Node.js 依赖
npm install
```

#### 6.3 构建项目

```bash
# 编译 TypeScript 代码
npm run build
```

构建完成后，会在 `build/` 目录下生成 `index.js` 文件。

#### 6.4 配置 MCP 路径

在 `.env` 文件中配置 MCP 服务路径：

```env
# Windows 路径示例
MCP_MEDICAL_PATH=C:\path\to\medical-mcp\build\index.js

# 或使用绝对路径
MCP_MEDICAL_PATH=D:\Projects\medical-mcp\build\index.js
```

> **注意**：路径必须指向编译后的 `build/index.js` 文件

#### 6.5 验证 MCP 服务

MCP 服务通过 stdio 方式调用，无需单独启动。应用会在需要时自动调用。

你可以手动测试：

```bash
node build/index.js
```



## 🎯 使用方法

### 启动应用

```bash
cd medgemma/gradio_chatbot
python app.py
```

应用将在 `http://127.0.0.1:7860` 启动。

### 使用流程

1. **描述症状**：在文本框中输入症状或使用语音输入
2. **回答问题**：按照系统提示回答相关问题
3. **审核摘要**：系统会生成病情摘要，您可以查看和编辑
4. **获取建议**：确认摘要后，系统会提供医疗建议

### 快捷功能

- **💡 直接生成建议**：如果信息足够，可直接跳到建议阶段
- **🗑️ 清空对话**：开始新的问诊会话

## ⚙️ 配置说明

### 模型配置

在 `config/settings.py` 中修改：

```python
# 提问模型
QUESTIONER_MODEL_CONFIG = {
    "model": "z-ai/glm-4.5-air:free",
    "api_key": SecretStr(OPENROUTER_API_KEY),
    # ...
}

# MedGemma 模型
MEDGEMMA_MODEL_CONFIG = {
    "model": "unsloth/medgemma-4b-it-bnb-4bit",
    "base_url": "http://localhost:8000/v1",
    # ...
}
```

### 提示词配置

在 `config/prompts.py` 中自定义各阶段的提示词：

- `QUESTIONER_PROMPT`：提问阶段提示词
- `SUMMARY_PROMPT`：摘要生成提示词
- `ADVICE_SYSTEM_PROMPT`：建议生成提示词

## 📝 开发说明

### 代码结构

- **`app.py`**：主应用程序，包含 Gradio UI 逻辑
- **`graph/`**：LangGraph 工作流定义
  - `nodes.py`：定义各个处理节点
  - `edges.py`：定义节点间的路由逻辑
  - `builder.py`：组装完整的工作流图
- **`tools/`**：外部工具集成
- **`utils/`**：通用工具函数

### 扩展开发

1. **添加新节点**：在 `graph/nodes.py` 中定义新的处理节点
2. **修改路由**：在 `graph/edges.py` 中调整路由逻辑
3. **集成工具**：在 `tools/` 中添加新的工具集成

## 🔍 调试

项目集成了 LangSmith 追踪：

1. 访问 [LangSmith](https://smith.langchain.com/)
2. 查看项目 `consultation-flow`
3. 追踪每次运行的详细信息

## ⚠️ 注意事项

1. **医疗免责声明**：本系统提供的是初步建议，不能代替正式的医疗诊断。如有严重症状，请及时就医。
2. **API 密钥安全**：请勿将包含真实 API 密钥的配置文件提交到公共仓库
3. **模型资源**：语音模型需要较大内存，建议使用 GPU 加速

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目采用 MIT 许可证。

## 📧 联系方式

如有问题或建议，请提交 Issue 或联系项目维护者。

---

**开发时间**：2025-12

**最后更新**：2025-12-25
