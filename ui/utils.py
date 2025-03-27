import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import gradio as gr
import numpy as np
import requests

# 导入配置
from config import Config

# 全局配置
g_config = Config()

# API配置
API_URL = "http://127.0.0.1:9880"  # 默认API地址

# 语言选项常量
LANGUAGE_OPTIONS = [
    ("中文", "中文"),
    ("英文", "英文"),
    ("日文", "日文"),
    ("粤语", "粤语"),
    ("韩文", "韩文"),
    ("中英混合", "中英混合"),
    ("日英混合", "日英混合"),
    ("粤英混合", "粤英混合"),
    ("韩英混合", "韩英混合"),
    ("多语种混合", "多语种混合"),
    ("多语种混合(粤语)", "多语种混合(粤语)"),
]

# 默认角色
g_default_role: str = "凡子霞"

# 确保必要的目录存在
os.makedirs("configs/roles", exist_ok=True)
os.makedirs("configs/ref_audio", exist_ok=True)
os.makedirs("output", exist_ok=True)

# 全局替换字典
g_word_replace_dict = {}


def load_word_replace_dict() -> Dict[str, str]:
    """加载字符替换字典
    
    从configs/word_replace.txt加载字符替换规则
    格式：替换前 替换后
    """
    replace_dict = {}
    replace_file = Path("configs/word_replace.txt")
    
    if not replace_file.exists():
        print(f"警告: 字符替换文件不存在: {replace_file}")
        return replace_dict
    
    try:
        with open(replace_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):  # 跳过空行和注释
                    continue
                
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    print(f"警告: 字符替换格式错误: {line}")
                    continue
                
                src, dst = parts
                # 英文不区分大小写，将英文部分转为小写作为键
                src_lower = ''.join([c.lower() if c.isascii() and c.isalpha() else c for c in src])
                replace_dict[src_lower] = dst
    except Exception as e:
        print(f"加载字符替换文件失败: {str(e)}")
    
    return replace_dict


def clean_text(text: str) -> str:
    """清理文本，进行字符替换
    
    Args:
        text: 原始文本
        
    Returns:
        替换后的文本
    """
    global g_word_replace_dict
    if not g_word_replace_dict:
        g_word_replace_dict = load_word_replace_dict()
        
    if not g_word_replace_dict:
        return text
    
    result = text
    
    # 对文本中的每个单词进行检查和替换
    for src, dst in g_word_replace_dict.items():
        # 创建一个模式，匹配源字符串，英文部分不区分大小写
        pattern = ''.join(['[' + c.upper() + c.lower() + ']' if c.isascii() and c.isalpha() else re.escape(c) for c in src])
        result = re.sub(pattern, dst, result)
    
    return result


def parse_line(line: str) -> tuple[str, str, str]:
    """解析文本行，返回(角色名, 情绪, 文本)"""
    line = line.strip()
    if not line:
        return "", "", ""

    # 匹配带情绪的格式 (角色|情绪)文本
    if line.startswith("(") and "|" in line.split(")")[0]:
        role_emotion, text = line.split(")", 1)
        role_name, emotion = role_emotion.strip("()").split("|")
        return role_name.strip(), emotion.strip(), text.strip()

    # 匹配不带情绪的格式 (角色)文本
    elif line.startswith("(") and ")" in line:
        role, text = line.split(")", 1)
        return role.strip("()").strip(), "", text.strip()

    return "", "", line.strip()


def load_word_replace_config():
    """加载并更新全局替换字典"""
    global g_word_replace_dict
    try:
        with open("configs/word_replace.txt", "r", encoding="utf-8") as f:
            content = f.read()
            # 更新全局字典
            g_word_replace_dict = load_word_replace_dict()
            return content
    except Exception as e:
        print(f"加载词语替换配置失败: {e}")
        return ""


def save_word_replace_config(text):
    """保存并更新全局替换字典"""
    global g_word_replace_dict
    try:
        with open("configs/word_replace.txt", "w", encoding="utf-8") as f:
            f.write(text)
        # 更新全局字典
        g_word_replace_dict = load_word_replace_dict()
        return "保存成功！"
    except Exception as e:
        print(f"保存词语替换配置失败: {e}")
        return f"保存失败: {e}"

# 初始化全局替换字典
g_word_replace_dict = load_word_replace_dict() 