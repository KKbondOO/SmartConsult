import json
import uuid
import gradio as gr
from typing import AsyncGenerator, Optional
from langchain_core.messages import AIMessageChunk, HumanMessage, AIMessage

from config.settings import DEFAULT_SESSION_ID
from config.prompts import WELCOME_MESSAGE
from utils.tts import text_to_speech
from utils.asr import speech_to_text
from graph.builder import graph

# 会话 ID
SESSION_ID = DEFAULT_SESSION_ID

# ==================== 核心功能函数 ====================

async def agent_stream_response(message: Optional[str], skip_to_advice: bool = False) -> AsyncGenerator[str, None]:
    """
    使用 consultation_flow graph 流式生成回复
    """
    if not message or not message.strip():
        return
    
    try:
        config = {"configurable": {"thread_id": SESSION_ID}}
        
        input_messages = {
            "messages": [HumanMessage(content=message)],
            "skip_to_advice": skip_to_advice
        }
        
        try:
            async for event in graph.astream(
                input_messages,
                config=config,
                stream_mode="updates"
            ):
                for node_name, node_output in event.items():
                    print(f"🔍 节点: {node_name}")
                    
                    if node_name == "__interrupt__":
                        state = await graph.aget_state(config)
                        if state.tasks and state.tasks[0].interrupts:
                            interrupt_data = state.tasks[0].interrupts[0]
                            print(f"🔔 检测到中断,返回审核数据")
                            yield "\n\n__INTERRUPT__\n" + json.dumps(interrupt_data.value, ensure_ascii=False)
                        continue
                    else:
                        if node_name == "medgemma_decision":
                            continue
                        
                        if "messages" in node_output:
                            messages = node_output["messages"]
                            if not isinstance(messages, list):
                                messages = [messages]
                            
                            if messages:
                                msg = messages[-1]
                                if isinstance(msg, (AIMessage, AIMessageChunk)):
                                    content = msg.content.strip() if msg.content else ""
                                    yield content
        except Exception as stream_error:
            print(f"⚠️ 流式输出错误: {stream_error}")
            raise
        
    except Exception as e:
        print(f"Agent 流式对话错误: {e}")
        yield f"抱歉，发生了错误：{str(e)}"

async def resume_with_edited_summary(edited_summary: str):
    """使用编辑后的摘要恢复执行"""
    from langgraph.types import Command
    try:
        config = {"configurable": {"thread_id": SESSION_ID}}
        result = await graph.ainvoke(
            Command(resume=edited_summary),
            config=config
        )
        return result
    except Exception as e:
        print(f"恢复执行错误: {e}")
        raise

async def get_current_question_count() -> int:
    """获取当前会话的提问轮次"""
    try:
        config = {"configurable": {"thread_id": SESSION_ID}}
        state = await graph.aget_state(config)
        return state.values.get("question_count", 0)
    except Exception as e:
        print(f"获取提问轮次错误: {e}")
        return 0

# ==================== Gradio UI Logic ====================

def add_user_message(history, message):
    if isinstance(message, dict):
        text = message.get("text") or ""
    else:
        text = message if message else ""
    
    if text.strip():
        history.append({"role": "user", "content": text})
    return history, ""

def check_interrupt_in_chunk(chunk):
    if "__INTERRUPT__" in chunk:
        parts = chunk.split("__INTERRUPT__")
        clean_chunk = parts[0]
        interrupt_data = None
        if len(parts) > 1:
            try:
                interrupt_data = json.loads(parts[1].strip())
            except:
                pass
        return True, interrupt_data, clean_chunk
    return False, None, chunk

async def process_text_input_stream(message, history, enable_tts, skip_to_advice=False):
    if isinstance(message, dict):
        user_text = message.get("text") or ""
    else:
        user_text = message if message else ""
    
    if not user_text.strip():
        yield history, None, gr.update(), gr.update(visible=False), ""
        return
    
    history.append({"role": "assistant", "content": ""})
    full_response = ""
    has_interrupt = False
    interrupt_data = None
    
    try:
        async for chunk in agent_stream_response(user_text, skip_to_advice=skip_to_advice):
            has_interrupt_in_chunk, interrupt_data_in_chunk, clean_chunk = check_interrupt_in_chunk(chunk)
            if has_interrupt_in_chunk:
                has_interrupt = True
                interrupt_data = interrupt_data_in_chunk
            
            full_response += clean_chunk
            history[-1]["content"] = full_response
            yield history, None, gr.update(), gr.update(visible=False), ""
        
        if has_interrupt and interrupt_data:
            summary_text = interrupt_data.get("summary", "")
            instruction = interrupt_data.get("instruction", "请审核病情摘要")
            history[-1]["content"] = full_response + f"\n\n📋 **{instruction}**"
            question_count = await get_current_question_count()
            button_visible = question_count >= 1 and not has_interrupt
            yield history, None, gr.update(visible=button_visible), gr.update(visible=True), summary_text
            return
        
        audio_path = text_to_speech(full_response) if enable_tts and full_response.strip() else None
        question_count = await get_current_question_count()
        button_visible = question_count >= 1
        yield history, audio_path, gr.update(visible=button_visible), gr.update(visible=False), ""
        
    except Exception as e:
        history[-1]["content"] = f"抱歉，发生了错误：{str(e)}"
        yield history, None, gr.update(), gr.update(visible=False), ""

def process_voice_to_text(audio, history):
    if audio is None: return history, ""
    text = speech_to_text(audio)
    if not text: return history, ""
    history.append({"role": "user", "content": text})
    return history, text

async def process_voice_response_stream(user_text, history, enable_tts, skip_to_advice=False):
    async for result in process_text_input_stream(user_text, history, enable_tts, skip_to_advice):
        yield result

async def generate_direct_advice(history, enable_tts):
    history.append({"role": "assistant", "content": ""})
    full_response = ""
    has_interrupt = False
    interrupt_data = None
    try:
        async for chunk in agent_stream_response("生成用户病况摘要", skip_to_advice=True):
            has_interrupt_in_chunk, interrupt_data_in_chunk, clean_chunk = check_interrupt_in_chunk(chunk)
            if has_interrupt_in_chunk:
                has_interrupt = True
                interrupt_data = interrupt_data_in_chunk
            full_response += clean_chunk
            history[-1]["content"] = full_response
            yield history, None, gr.update(visible=False), gr.update(visible=False), ""
        
        if has_interrupt and interrupt_data:
            summary_text = interrupt_data.get("summary", "")
            instruction = interrupt_data.get("instruction", "请审核病情摘要")
            history[-1]["content"] = full_response + f"\n\n📋 **{instruction}**"
            yield history, None, gr.update(visible=False), gr.update(visible=True), summary_text
            return
        
        audio_path = text_to_speech(full_response) if enable_tts and full_response.strip() else None
        yield history, audio_path, gr.update(visible=False), gr.update(visible=False), ""
    except Exception as e:
        history[-1]["content"] = f"抱歉，发生了错误：{str(e)}"
        yield history, None, gr.update(visible=False), gr.update(visible=False), ""

async def submit_summary_review(edited_summary, history, enable_tts):
    if not edited_summary or not edited_summary.strip():
        yield history, None, gr.update(visible=False), ""
        return
    history.append({"role": "assistant", "content": "正在基于您审核的摘要生成医疗建议..."})
    yield history, None, gr.update(visible=False), ""
    try:
        result = await resume_with_edited_summary(edited_summary)
        if result and "messages" in result:
            advice_content = result["messages"][-1].content
            history[-1]["content"] = advice_content
            audio_path = text_to_speech(advice_content) if enable_tts and advice_content.strip() else None
            yield history, audio_path, gr.update(visible=False), ""
        else:
            history[-1]["content"] = "建议生成完成"
            yield history, None, gr.update(visible=False), ""
    except Exception as e:
        history[-1]["content"] = f"抱歉，处理摘要时发生错误：{str(e)}"
        yield history, None, gr.update(visible=False), ""

def clear_conversation():
    global SESSION_ID
    SESSION_ID = str(uuid.uuid4())
    return WELCOME_MESSAGE, None, gr.update(visible=False), gr.update(visible=False), ""

# ==================== Gradio UI Layout ====================

custom_css = """
.title-text {
 text-align: center;
 font-size: 2em; 
 font-weight: bold; 
 background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
 -webkit-background-clip: text; 
 -webkit-text-fill-color: transparent; 
 margin-bottom: 0.5em; 
}
.subtitle-text { 
 text-align: center; 
 color: #666; 
 margin-bottom: 1em; 
}
.chatbot-container {
 border-radius: 15px !important; 
 box-shadow: 0 4px 20px rgba(0,0,0,0.1) !important; 
}
.primary-btn {
 background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; 
 border: none !important; 
 border-radius: 10px !important; 
 color: white !important;
}
.secondary-btn {
 background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important; 
 border: none !important; 
 border-radius: 10px !important; 
 color: white !important; 
}
.input-textbox textarea {
 border-radius: 10px !important; 
 border: 2px solid #e0e0e0 !important;
}
.markdown-preview {
 background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
 border-radius: 10px; 
 padding: 15px; 
 margin-top: 10px; 
 box-shadow: 0 2px 8px rgba(0,0,0,0.1); 
 max-height: 400px; 
 overflow-y: auto;
}
"""

with (gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo):
    gr.HTML("<div class='title-text'>🎙️ 智能语音助手</div><div class='subtitle-text'>支持文字输入和语音对话</div>")
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="对话记录", height=450, elem_classes=["chatbot-container"], type="messages", value=WELCOME_MESSAGE)
            user_message_state = gr.State(value="")
            text_input = gr.Textbox(label="💬 输入消息", placeholder="输入您想说的话，按回车发送...", lines=1, elem_classes=["input-textbox"], submit_btn=False)
            with gr.Row():
                gr.Column(scale=3)
                send_btn = gr.Button("发送 ✉️", variant="primary", elem_classes=["primary-btn"])
            with gr.Group(visible=False) as summary_review_group:
                summary_textbox = gr.Textbox(label="请审核并编辑病情摘要", lines=10, elem_classes=["input-textbox"])
                summary_preview = gr.Markdown(elem_classes=["markdown-preview"])
                summary_textbox.change(fn=lambda x: x, inputs=summary_textbox, outputs=summary_preview)
                with gr.Row():
                    submit_summary_btn = gr.Button("✅ 确认并生成建议", variant="primary", elem_classes=["primary-btn"])
                    cancel_summary_btn = gr.Button("❌ 取消", variant="secondary", elem_classes=["secondary-btn"])
            summary_state = gr.State(value="")
        with gr.Column(scale=1):
            audio_input = gr.Audio(sources=["microphone"], type="filepath", label="点击即可录音")
            audio_output = gr.Audio(label="机器人语音", autoplay=True)
            enable_tts = gr.Checkbox(label="🔈 启用语音回复", value=True)
            direct_advice_btn = gr.Button("💡 直接生成建议回复", variant="primary", elem_classes=["primary-btn"], visible=False)
            clear_btn = gr.Button("🗑️ 清空对话", variant="secondary", elem_classes=["secondary-btn"])

    def save_message_to_state(message):
        return message.get("text") if isinstance(message, dict) else (message or "")

    send_btn.click(
        fn=save_message_to_state,
        inputs=[text_input],
        outputs=[user_message_state]
    ).then(
        fn=add_user_message,
        inputs=[chatbot, text_input],
        outputs=[chatbot, text_input]
    ).then(
        fn=process_text_input_stream,
        inputs=[user_message_state, chatbot, enable_tts],
        outputs=[chatbot, audio_output, direct_advice_btn, summary_review_group, summary_state]
    ).then(fn=lambda s: [s,s], inputs=[summary_state], outputs=[summary_textbox, summary_preview])

    text_input.submit(
        fn=save_message_to_state,
        inputs=[text_input],
        outputs=[user_message_state]
    ).then(
        fn=add_user_message,
        inputs=[chatbot, text_input], outputs=[chatbot, text_input]
    ).then(
        fn=process_text_input_stream,
        inputs=[user_message_state, chatbot, enable_tts],
        outputs=[chatbot, audio_output, direct_advice_btn, summary_review_group, summary_state]
    ).then(fn=lambda s: [s, s], inputs=[summary_state], outputs=[summary_textbox, summary_preview])

    audio_input.stop_recording(
        fn=process_voice_to_text,
        inputs=[audio_input, chatbot],
        outputs=[chatbot, user_message_state]
    ).then(
        fn=process_voice_response_stream,
        inputs=[user_message_state, chatbot, enable_tts],
        outputs=[chatbot, audio_output, direct_advice_btn, summary_review_group, summary_state])

    direct_advice_btn.click(
        fn=generate_direct_advice,
        inputs=[chatbot, enable_tts],
        outputs=[chatbot, audio_output, direct_advice_btn, summary_review_group, summary_state]
    ).then(fn=lambda s: [s,s], inputs=[summary_state], outputs=[summary_textbox, summary_preview])

    submit_summary_btn.click(
        fn=submit_summary_review,
        inputs=[summary_textbox, chatbot, enable_tts],
        outputs=[chatbot, audio_output, summary_review_group, summary_state])

    cancel_summary_btn.click(
        fn=lambda: (gr.update(visible=False), ""),
        outputs=[summary_review_group, summary_state])

    clear_btn.click(
        fn=clear_conversation,
        outputs=[chatbot, audio_output, direct_advice_btn, summary_review_group, summary_state])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)
