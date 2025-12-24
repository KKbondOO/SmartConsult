from transformers import pipeline
from text_utils import convert_t2s

# ASR 模型 (Whisper)
print("正在加载 ASR 模型...")
asr_pipe = pipeline(
    "automatic-speech-recognition", 
    model="openai/whisper-small",
    generate_kwargs={"language": "zh", "task": "transcribe"}
)
print("ASR 模型加载完成！")

def speech_to_text(audio_path: str) -> str:
    """
    将语音转换为文本
    """
    if audio_path is None:
        return ""
    
    try:
        result = asr_pipe(audio_path)
        user_text = result["text"].strip()
        # 将繁体转换为简体
        user_text = convert_t2s(user_text)
        return user_text
    except Exception as e:
        print(f"ASR 错误: {e}")
        return ""
