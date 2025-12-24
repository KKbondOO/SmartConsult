import re
from opencc import OpenCC

# 繁简转换器
t2s = OpenCC('t2s')  # 繁体转简体

# 分句正则表达式（匹配中英文标点结尾的句子）
SENTENCE_END_PATTERN = re.compile(r'[。！？.!?]')

def clean_markdown(text: str) -> str:
    """移除 Markdown 格式标记"""
    # 移除粗体 **text**
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    # 移除斜体 *text*
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    # 移除标题标记 # ## ###
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    # 移除列表标记 - * +
    text = re.sub(r'^[\-\*\+]\s+', '', text, flags=re.MULTILINE)
    # 移除数字列表 1. 2. 3.
    text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
    return text.strip()

def extract_sentences(text: str) -> tuple[list[str], str]:
    """
    从文本中提取完整的句子
    返回: (完整句子列表, 剩余未完成的文本)
    """
    sentences = []
    last_end = 0
    
    for match in SENTENCE_END_PATTERN.finditer(text):
        end_pos = match.end()
        sentences.append(text[last_end:end_pos])
        last_end = end_pos
    
    remaining = text[last_end:]
    return sentences, remaining

def convert_t2s(text: str) -> str:
    """繁体转简体"""
    return t2s.convert(text)
