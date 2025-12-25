import os
import torch
import numpy as np
import soundfile as sf
import tempfile
from kokoro import KPipeline, KModel
from medgemma.gradio_chatbot.config.settings import KOKORO_MODEL_PATH, KOKORO_CONFIG_PATH, KOKORO_REPO_ID, KOKORO_VOICES_DIR
# TTS 模型 (Kokoro)
print("正在加载 TTS 模型...")
voice_zf = "zf_xiaoxiao"
voice_zf_tensor = torch.load(
    os.path.join(KOKORO_VOICES_DIR, f"{voice_zf}.pt"), 
    weights_only=True
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tts_model = KModel(model=KOKORO_MODEL_PATH, config=KOKORO_CONFIG_PATH, repo_id=KOKORO_REPO_ID).to(device).eval()

en_pipeline = KPipeline(lang_code='a', repo_id=KOKORO_REPO_ID, model=tts_model)

def en_callable(text):
    return next(en_pipeline(text, voice=voice_zf_tensor)).phonemes

def speed_callable(len_ps):
    speed = 0.8
    if len_ps <= 83:
        speed = 1
    elif len_ps < 183:
        speed = 1 - (len_ps - 83) / 500
    return speed * 1.1

zh_pipeline = KPipeline(lang_code='z', repo_id=KOKORO_REPO_ID, model=tts_model, en_callable=en_callable)
print("TTS 模型加载完成！")

def text_to_speech(text: str) -> str | None:
    """
    将文本转换为语音
    返回音频文件路径
    """
    if not text or not text.strip():
        return None
    
    # 清理换行符，避免TTS处理问题
    text = text.replace('\n', ' ').replace('\r', ' ').strip()
    
    try:
        # 使用中文管道生成语音
        generator = zh_pipeline(
            text, 
            voice=voice_zf_tensor, 
            speed=speed_callable
        )
        # 收集所有音频片段并拼接
        audio_chunks = []
        for result in generator:
            audio_chunks.append(result.audio)
        
        if not audio_chunks:
            return None
        wav = np.concatenate(audio_chunks)
        
        # 保存到临时文件
        temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
        os.close(temp_fd)  # 关闭文件描述符
        sf.write(temp_path, wav, 24000)
        
        return temp_path
    except Exception as e:
        print(f"TTS 错误: {e}")
        return None
