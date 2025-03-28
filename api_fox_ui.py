import json
import os
import re
import time
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union, Dict

import gradio as gr
import librosa  # 添加这个导入
import numpy as np
import requests
import soundfile as sf
from numpy.typing import NDArray

# 导入配置
from config import Config

g_config = Config()

# API配置
API_URL = "http://127.0.0.1:9880"  # 默认API地址

# 修改语言选项常量
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

g_default_role: str = "凡子霞"

# 确保必要的目录存在
os.makedirs("configs/roles", exist_ok=True)
os.makedirs("configs/ref_audio", exist_ok=True)
os.makedirs("output", exist_ok=True)

def get_model_paths_from_role(role: str) -> Tuple[str, str]:
    """从角色配置文件中获取模型路径"""
    role_path = Path("configs/roles") / f"{role}.json"
    if not role_path.exists():
        raise ValueError(f"找不到角色配置文件: {role}")

    with open(role_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 获取模型路径，优先使用角色配置中的路径
    gpt_path = config.get("gpt_path")
    sovits_path = config.get("sovits_path")

    # 如果角色配置中没有指定，则使用全局配置
    if not gpt_path:
        gpt_path = g_config.gpt_path or g_config.pretrained_gpt_path
    if not sovits_path:
        sovits_path = g_config.sovits_path or g_config.pretrained_sovits_path

    # 确保路径是绝对路径
    gpt_path = os.path.abspath(gpt_path)
    sovits_path = os.path.abspath(sovits_path)

    # 检查模型文件是否存在
    if not os.path.exists(gpt_path):
        raise ValueError(f"找不到GPT模型: {gpt_path}")
    if not os.path.exists(sovits_path):
        raise ValueError(f"找不到SoVITS模型: {sovits_path}")

    return gpt_path, sovits_path


def init_models(role: str) -> Tuple[str, str]:
    """初始化模型"""
    # 从角色配置获取模型路径
    gpt_path, sovits_path = get_model_paths_from_role(role)
    print(f"正在设置模型 - 角色: {role}")
    print(f"- GPT模型: {gpt_path}")
    print(f"- SoVITS模型: {sovits_path}")

    # 设置模型
    try:
        response = requests.post(
            f"{API_URL}/set_model",
            json={
                "gpt_model_path": gpt_path,
                "sovits_model_path": sovits_path,
            },
            timeout=30,
        )
        if response.status_code != 200:
            error_msg = response.text
            try:
                error_data = response.json()
                if "message" in error_data:
                    error_msg = error_data["message"]
            except Exception:
                pass
            raise gr.Error(f"设置模型失败: {error_msg}")
    except requests.exceptions.RequestException as e:
        raise gr.Error(f"API请求失败: {str(e)}")

    return gpt_path, sovits_path


def list_roles() -> list[str]:
    """获取所有可用的角色列表"""
    roles_dir = Path("configs/roles")
    roles = []
    for file in roles_dir.glob("*.json"):
        roles.append(file.stem)
    roles = sorted(roles)
    if not roles:
        raise gr.Error("未找到任何角色配置文件，请确保 configs/roles 目录下有角色配置")
    return roles


def get_emotions(role: str = "") -> list[str]:
    """获取情绪列表，从角色配置文件中读取"""
    if not role:
        return []

    # 读取角色配置
    role_path = Path("configs/roles") / f"{role}.json"
    if not role_path.exists():
        return []

    with open(role_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if "emotions" not in config:
        return []

    # 直接返回配置文件中定义的情绪列表
    return list(config["emotions"].keys())


def get_role_config(role: str, emotion: str = "", text_lang: str = "中文") -> dict:
    """获取角色配置"""
    role_path = Path("configs/roles") / f"{role}.json"
    if not role_path.exists():
        raise ValueError(f"找不到角色配置文件: {role}")

    with open(role_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 创建配置的副本以避免修改原始配置
    result_config = config.copy()

    # 处理情绪配置
    if "emotions" not in config:
        raise ValueError(f"角色 {role} 配置缺少情绪配置")

    emotions = config["emotions"]
    # 如果没有指定情绪，使用第一个情绪作为默认值
    if not emotion:
        emotion = next(iter(emotions.keys()))
    elif emotion not in emotions:
        raise ValueError(f"角色 {role} 不支持情绪: {emotion}")

    # 获取情绪配置并更新结果配置
    emotion_config = emotions[emotion].copy()
    result_config.update(emotion_config)

    # 删除emotions字段，因为已经合并了具体的情绪配置
    if "emotions" in result_config:
        del result_config["emotions"]

    # 确保必要的字段存在
    result_config.setdefault("prompt_lang", "zh")
    result_config.setdefault("speed", 1.0)
    result_config.setdefault("top_k", 15)
    result_config.setdefault("top_p", 1.0)
    result_config.setdefault("temperature", 1.0)
    result_config.setdefault("sample_steps", 32)
    result_config.setdefault("if_sr", False)

    # 检查必要的配置
    if "ref_audio" not in result_config:
        raise ValueError(f"角色 {role} 的情绪 {emotion} 配置缺少参考音频路径")
    if not os.path.exists(result_config["ref_audio"]):
        raise ValueError(f"找不到参考音频文件: {result_config['ref_audio']}")

    # 删除从配置文件读取的语言设置
    if "text_lang" in result_config:
        del result_config["text_lang"]

    # 使用界面传入的语言设置
    result_config["text_lang"] = text_lang

    return result_config


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


# 全局替换字典
g_word_replace_dict = load_word_replace_dict()

def clean_text(text: str) -> str:
    """清理文本，进行字符替换
    
    Args:
        text: 原始文本
        
    Returns:
        替换后的文本
    """
    if not g_word_replace_dict:
        return text
    
    result = text
    
    # 对文本中的每个单词进行检查和替换
    for src, dst in g_word_replace_dict.items():
        # 创建一个模式，匹配源字符串，英文部分不区分大小写
        pattern = ''.join(['[' + c.upper() + c.lower() + ']' if c.isascii() and c.isalpha() else re.escape(c) for c in src])
        result = re.sub(pattern, dst, result)
      
    
    return result


def call_api(text: str, role_config: dict, role_name: str, cut_punc: str = "") -> bytes:
    """调用API进行推理

    Args:
        text: 要转换的文本
        role_config: 角色配置
        role_name: 角色名称
        cut_punc: 切分符号
    """
    # 清理文本，进行字符替换
    text = clean_text(text)
    
    # 检查必要的配置
    if "ref_audio" not in role_config:
        raise ValueError("角色配置缺少参考音频路径(ref_audio)")
    if not os.path.exists(role_config["ref_audio"]):
        raise ValueError(f"找不到参考音频文件: {role_config['ref_audio']}")

    # 构建请求参数
    params = {
        "refer_wav_path": role_config["ref_audio"],
        "prompt_text": role_config.get("prompt_text", ""),
        "prompt_language": role_config.get("prompt_lang", "zh"),
        "text": text,
        "text_language": role_config.get("text_lang", "zh"),
        "speed": role_config.get("speed", 1.0),
        "top_k": role_config.get("top_k", 15),
        "top_p": role_config.get("top_p", 1.0),
        "temperature": role_config.get("temperature", 1.0),
        "sample_steps": role_config.get("sample_steps", 32),
        "if_sr": role_config.get("if_sr", False),
        "cut_punc": cut_punc,
        "spk": role_name,  # 添加角色名称
    }

    if "aux_refs" in role_config:
        params["inp_refs"] = role_config["aux_refs"]

    try:
        print(params)
        response = requests.post(API_URL, json=params)
        if response.status_code != 200:
            error_msg = response.text
            try:
                error_data = response.json()
                if "message" in error_data:
                    error_msg = error_data["message"]
            except:
                pass
            raise RuntimeError(f"API调用失败: {error_msg}")
        return response.content
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"无法连接到API服务器({API_URL})，请确保API服务已启动")


def merge_audio_segments(
    audio_segments: List[bytes], target_sr: int = 32000
) -> Tuple[NDArray[np.float64], int]:
    """合并多个音频片段

    Args:
        audio_segments: 音频数据列表
        target_sr: 目标采样率，默认32000Hz
    """
    audio_arrays: List[NDArray[np.float64]] = []

    for audio_data in audio_segments:
        # 从字节数据读取音频
        with BytesIO(audio_data) as bio:
            # 先读取原始采样率
            audio_array, sr = sf.read(bio)

            # 如果采样率不一致，进行重采样
            if sr != target_sr:
                print(f"重采样: {sr} -> {target_sr}")
                # 将数据转换为float32以提高精度
                audio_array = audio_array.astype(np.float32)
                # 使用librosa进行重采样
                audio_array = librosa.resample(
                    audio_array, orig_sr=sr, target_sr=target_sr
                )

            audio_arrays.append(audio_array)

    if not audio_arrays:
        raise ValueError("没有有效的音频数据")

    # 在音频片段之间添加短暂的静音
    silence_duration = 0.3  # 秒
    silence_samples = int(target_sr * silence_duration)
    silence = np.zeros(silence_samples, dtype=np.float64)

    # 合并所有音频片段，中间加入静音
    merged: List[NDArray[np.float64]] = []
    for i, audio in enumerate(audio_arrays):
        merged.append(audio)
        if i < len(audio_arrays) - 1:  # 最后一个片段后不加静音
            merged.append(silence)

    return np.concatenate(merged), target_sr


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


def preprocess_text(text_content: str, default_role: str, default_emotion: str) -> str:
    """预处理文本，提取双引号或「」内容作为对白，其他作为叙述
    
    Args:
        text_content: 原始文本内容
        default_role: 默认角色
        default_emotion: 默认情绪
        
    Returns:
        处理后的文本
    """
    if not default_role:
        raise gr.Error("请先选择默认角色")
    
    # 获取角色配置
    role_path = Path("configs/roles") / f"{default_role}.json"
    if not role_path.exists():
        raise gr.Error(f"找不到角色配置文件: {default_role}")
    
    with open(role_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # 获取情绪列表
    if "emotions" not in config:
        raise gr.Error(f"角色 {default_role} 配置缺少情绪配置")
    
    emotions = list(config["emotions"].keys())
    
    # 如果没有指定默认情绪，使用第一个情绪
    if not default_emotion:
        default_emotion = emotions[0]
    
    # 处理文本
    lines = text_content.strip().split("\n")
    processed_lines = []
    
    for line in lines:
        if not line.strip():
            processed_lines.append("")
            continue
        
        # 先处理双引号包围的内容
        if '"' in line:
            # 查找所有双引号包围的内容
            dialogue_parts = re.findall(r'"([^"]*)"', line)
            
            if dialogue_parts:
                # 处理包含对白的行
                remaining = line
                for part in dialogue_parts:
                    # 分割成对白前、对白、对白后三部分
                    before, remaining = remaining.split(f'"{part}"', 1)
                    
                    # 处理对白前的叙述部分(如果有)，不添加角色标记
                    if before.strip():
                        processed_lines.append(f"{before.strip()}")
                    
                    # 处理对白部分，添加角色和情绪
                    processed_lines.append(f"({default_role}|{default_emotion})\"{part}\"")
                
                # 处理最后一个对白后的叙述部分(如果有)，不添加角色标记
                if remaining.strip():
                    processed_lines.append(f"{remaining.strip()}")
                
                continue  # 已处理完这一行，继续下一行
        
        # 处理「」包围的内容
        if '「' in line and '」' in line:
            # 查找所有「」包围的内容
            dialogue_parts = re.findall(r'「([^」]*)」', line)
            
            if dialogue_parts:
                # 处理包含对白的行
                remaining = line
                for part in dialogue_parts:
                    # 分割成对白前、对白、对白后三部分
                    before, remaining = remaining.split(f'「{part}」', 1)
                    
                    # 处理对白前的叙述部分(如果有)，不添加角色标记
                    if before.strip():
                        processed_lines.append(f"{before.strip()}")
                    
                    # 处理对白部分，添加角色和情绪，转换为双引号格式
                    processed_lines.append(f"({default_role}|{default_emotion})\"{part}\"")
                
                # 处理最后一个对白后的叙述部分(如果有)，不添加角色标记
                if remaining.strip():
                    processed_lines.append(f"{remaining.strip()}")
                
                continue  # 已处理完这一行，继续下一行

         # 处理"包围的内容
        if '"' in line and '"' in line:
            # 查找所有"包围的内容
            dialogue_parts = re.findall(r'"([^"]*)"', line)
            
            if dialogue_parts:
                # 处理包含对白的行
                remaining = line
                for part in dialogue_parts:
                    # 分割成对白前、对白、对白后三部分
                    before, remaining = remaining.split(f'"{part}"', 1)
                    
                    # 处理对白前的叙述部分(如果有)，不添加角色标记
                    if before.strip():
                        processed_lines.append(f"{before.strip()}")
                    
                    # 处理对白部分，添加角色和情绪，转换为双引号格式
                    processed_lines.append(f"({default_role}|{default_emotion})\"{part}\"")
                
                # 处理最后一个对白后的叙述部分(如果有)，不添加角色标记
                if remaining.strip():
                    processed_lines.append(f"{remaining.strip()}")
                
                continue  # 已处理完这一行，继续下一行
        
        # 没有对白，整行作为叙述，不添加任何标记
        processed_lines.append(f"{line.strip()}")
    
    return "\n".join(processed_lines)

def process_text(
    text: str,
    role: str,
    emotion: str = "",
    text_lang: str = "中文",
    cut_punc: str = "",
    output_dir: str = "output",
) -> str:
    """处理单条文本"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, f"output_{role}_{emotion}_{int(time.time())}.wav"
    )

    try:
        # 确保使用正确的模型
        init_models(role)

        role_config = get_role_config(role, emotion, text_lang)
        audio_data = call_api(text, role_config, role, cut_punc=cut_punc)

        with open(output_path, "wb") as f:
            f.write(audio_data)

        return output_path
    except Exception as e:
        raise gr.Error(f"处理失败: {str(e)}")


def process_text_content(
    text_content: str,
    force_role: str = "",
    default_role: str = "",
    force_emotion: str = "",
    default_emotion: str = "",
    text_lang: str = "中文",
    cut_punc: str = "",
    disable_parsing: bool = False,
    output_dir: str = "output",
) -> str:
    """处理文本内容"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"output_{int(time.time())}.wav")

    try:
        # 如果禁用解析，直接使用默认角色和情绪处理整个文本
        if disable_parsing:
            if not default_role:
                raise gr.Error("禁用解析时必须设置默认角色")

            # 如果没有指定情绪，使用角色配置中的第一个情绪
            if not default_emotion or default_emotion == "无":
                role_path = Path("configs/roles") / f"{default_role}.json"
                with open(role_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                if "emotions" in config:
                    default_emotion = next(iter(config["emotions"].keys()))
                else:
                    raise gr.Error(f"角色 {default_role} 配置缺少情绪配置")

            print(f"禁用解析模式 - 角色: {default_role} 情绪: {default_emotion}")
            
            # 处理文本，移除所有角色和情绪标记
            processed_text = ""
            lines = text_content.strip().split("\n")
            for line in lines:
                # 移除角色和情绪标记
                if line.startswith("(") and ")" in line:
                    _, text = line.split(")", 1)
                    processed_text += text.strip() + "\n"
                else:
                    processed_text += line.strip() + "\n"
            
            # 初始化默认角色的模型
            init_models(default_role)
            # 获取配置并调用API
            role_config = get_role_config(default_role, default_emotion, text_lang)
            audio_data = call_api(
                processed_text.strip(), role_config, default_role, cut_punc=cut_punc
            )

            # 保存音频
            with open(output_path, "wb") as f:
                f.write(audio_data)
            return output_path

        # 原有的处理逻辑
        lines = text_content.strip().split("\n")
        audio_segments = []
        current_role = None

        for i, line in enumerate(lines):
            role_name, emotion, text = parse_line(line)

            # 使用强制角色或原始解析的角色或默认角色
            if force_role and force_role != "" and force_role != "无":
                role_name = force_role
            elif not role_name:
                role_name = default_role

            if not role_name:
                raise gr.Error("未指定角色且未设置默认角色")

            # 使用强制情绪或原始解析的情绪或默认情绪
            if force_emotion and force_emotion != "" and force_emotion != "无":
                emotion = force_emotion
            elif not emotion and default_emotion and default_emotion != "无":
                emotion = default_emotion

            # 如果角色改变，需要重新初始化模型
            if role_name != current_role:
                print(f"切换到角色: {role_name}")
                init_models(role_name)
                current_role = role_name

            if not text:  # 空行
                continue

            try:
                print(f"当前角色: {role_name} 当前情绪: {emotion} 当前文本: {text}")
                # 获取配置并调用API
                role_config = get_role_config(role_name, emotion, text_lang)
                audio_data = call_api(text, role_config, role_name, cut_punc=cut_punc)
                audio_segments.append(audio_data)
            except Exception as e:
                print(f"处理文本失败: {text}, 错误: {str(e)}")
                continue

        if not audio_segments:
            raise gr.Error("没有可处理的文本")

        # 合并音频
        merged_audio, sample_rate = merge_audio_segments(audio_segments)

        # 保存合并后的音频
        sf.write(output_path, merged_audio, sample_rate)

        return output_path
    except Exception as e:
        raise gr.Error(f"处理失败: {str(e)}")


def process_file(
    file: gr.components.File,
    force_role: str = "",
    default_role: str = "",
    force_emotion: str = "",
    default_emotion: str = "",
    text_lang: str = "中文",
    cut_punc: str = "",
    disable_parsing: bool = False,
    output_dir: str = "output",
) -> str:
    """处理文本文件"""
    if not file or not hasattr(file, "name"):
        raise gr.Error("请选择文件")

    with open(file.name, "r", encoding="utf-8") as f:
        text_content = f.read()

    return process_text_content(
        text_content,
        force_role,
        default_role,
        force_emotion,
        default_emotion,
        text_lang,
        cut_punc,
        disable_parsing,
        output_dir,
    )


def update_force_emotions(role: str) -> gr.update:
    """更新强制情绪选项"""
    # 如果选择了"无"角色，返回空的情绪列表
    if not role or role == "无":
        empty_choices = [("", "无")]
        return gr.update(choices=empty_choices, value="")

    emotions = get_emotions(role)
    force_choices = [("", "无")] + [(e, e) for e in emotions]
    return gr.update(choices=force_choices, value="")


def update_default_emotions(role: str) -> gr.update:
    """更新默认情绪选项"""
    # 如果选择了"无"角色，返回空的情绪列表
    if not role or role == "无":
        return gr.update(choices=[], value=None)

    emotions = get_emotions(role)
    if not emotions:
        return gr.update(choices=[], value=None)

    # 直接使用情绪列表，不添加"无"选项
    default_choices = [(e, e) for e in emotions]
    return gr.update(choices=default_choices, value=emotions[0])  # 默认选择第一个情绪


def get_model_lists():
    """获取GPT和SoVITS模型列表"""
    # GPT模型列表
    gpt_models = []
    gpt_roots = ["GPT_weights", "GPT_weights_v2", "GPT_weights_v3"]
    for root in gpt_roots:
        if os.path.exists(root):
            for name in os.listdir(root):
                if name.endswith(".ckpt"):
                    gpt_models.append(f"{root}/{name}")
    
    # SoVITS模型列表
    sovits_models = []
    sovits_roots = ["SoVITS_weights", "SoVITS_weights_v2", "SoVITS_weights_v3"]
    for root in sovits_roots:
        if os.path.exists(root):
            for name in os.listdir(root):
                if name.endswith(".pth"):
                    sovits_models.append(f"{root}/{name}")
    
    return sorted(gpt_models), sorted(sovits_models)


def get_role_name_from_file_path(file_path):
    """从文件路径中提取角色名称"""
    # 提取文件名，不包含扩展名
    file_name = os.path.basename(file_path)
    file_name = os.path.splitext(file_name)[0]
    return file_name


def save_role_config(
    role_name: str,
    gpt_model: str,
    sovits_model: str,
    ref_audio: str,
    prompt_text: str,
    prompt_lang: str,
    text_lang: str,
    speed: float,
    ref_free: bool,
    if_sr: bool,
    top_k: int,
    top_p: float,
    temperature: float,
    sample_steps: int,
    pause_second: float,
    description: str = "",
    aux_refs: List[str] = None
) -> str:
    """保存角色配置到JSON文件
    
    Args:
        role_name: 角色名称
        gpt_model: GPT模型路径
        sovits_model: SoVITS模型路径
        ref_audio: 参考音频路径
        prompt_text: 参考音频文本
        prompt_lang: 参考音频语言
        text_lang: 默认文本语言
        speed: 语速
        ref_free: 是否开启参考自由模式
        if_sr: 是否使用超分辨率
        top_k: top_k参数
        top_p: top_p参数
        temperature: 温度参数
        sample_steps: 采样步数
        pause_second: 停顿秒数
        description: 角色描述
        aux_refs: 辅助参考音频列表
    
    Returns:
        成功或错误消息
    """
    if not role_name:
        return "请输入角色名称"
    
    if not ref_audio:
        return "请上传参考音频"
        
    if not gpt_model or not sovits_model:
        return "请选择GPT模型和SoVITS模型"
    
    # 确定版本
    version = "v2"  # 默认v2
    if "GPT_weights_v3" in gpt_model or "SoVITS_weights_v3" in sovits_model:
        version = "v3"
    elif "GPT_weights_v2" in gpt_model or "SoVITS_weights_v2" in sovits_model:
        version = "v2"
    elif "GPT_weights/" in gpt_model or "SoVITS_weights/" in sovits_model:
        version = "v1"
    
    # 创建角色参考音频目录
    ref_audio_dir = Path("configs/ref_audio") / role_name
    ref_audio_dir.mkdir(parents=True, exist_ok=True)
    
    # 提取参考音频文件名作为情绪名称和目标文件名
    orig_audio_filename = os.path.basename(ref_audio)
    emotion_name = os.path.splitext(orig_audio_filename)[0]
    if len(emotion_name) > 10:  # 如果名称太长，截取一部分
        short_emotion_name = emotion_name[:10]
    else:
        short_emotion_name = emotion_name
    
    # 复制参考音频文件到角色目录
    target_ref_audio = str(ref_audio_dir / orig_audio_filename)
    try:
        import shutil
        shutil.copy2(ref_audio, target_ref_audio)
        print(f"复制参考音频: {ref_audio} -> {target_ref_audio}")
    except Exception as e:
        return f"复制参考音频失败: {str(e)}"
    
    # 处理辅助参考音频
    aux_refs_copied = []
    if aux_refs and isinstance(aux_refs, (list, tuple)) and len(aux_refs) > 0:
        # 处理多个辅助参考音频文件
        for aux_ref in aux_refs:
            if aux_ref and os.path.exists(aux_ref):
                aux_filename = os.path.basename(aux_ref)
                target_aux_ref = str(ref_audio_dir / aux_filename)
                try:
                    shutil.copy2(aux_ref, target_aux_ref)
                    aux_refs_copied.append(target_aux_ref)
                    print(f"复制辅助参考音频: {aux_ref} -> {target_aux_ref}")
                except Exception as e:
                    print(f"复制辅助参考音频失败: {str(e)}")
    elif aux_refs and os.path.exists(aux_refs):  # 处理单个文件情况
        # 处理单个辅助参考音频
        aux_filename = os.path.basename(aux_refs)
        target_aux_ref = str(ref_audio_dir / aux_filename)
        try:
            shutil.copy2(aux_refs, target_aux_ref)
            aux_refs_copied.append(target_aux_ref)
            print(f"复制辅助参考音频: {aux_refs} -> {target_aux_ref}")
        except Exception as e:
            print(f"复制辅助参考音频失败: {str(e)}")
    
    # 构建配置数据
    config = {
        "version": version,
        "emotions": {
            short_emotion_name: {
                "ref_audio": target_ref_audio,
                "prompt_text": prompt_text
            }
        },
        "text_lang": text_lang,
        "prompt_lang": prompt_lang,
        "gpt_path": gpt_model,
        "sovits_path": sovits_model,
        "speed": speed,
        "ref_free": ref_free,
        "if_sr": if_sr,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "sample_steps": sample_steps,
        "pause_second": pause_second,
        "description": description
    }
    
    # 添加辅助参考音频
    if aux_refs_copied:
        config["emotions"][short_emotion_name]["aux_refs"] = aux_refs_copied
    
    # 保存配置文件
    config_dir = Path("configs/roles")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = config_dir / f"{role_name}.json"
    
    # 如果文件已存在，读取已有配置并合并情绪
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                existing_config = json.load(f)
            
            # 合并情绪配置
            if "emotions" in existing_config:
                # 保留现有的情绪配置
                for emotion, emotion_config in existing_config["emotions"].items():
                    if emotion != short_emotion_name:  # 避免覆盖新添加的情绪
                        config["emotions"][emotion] = emotion_config
            
            # 保留版本设置
            if "version" in existing_config:
                config["version"] = existing_config["version"]
                
        except Exception as e:
            return f"读取现有配置失败: {str(e)}"
    
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
        return f"角色 {role_name} 配置保存成功"
    except Exception as e:
        return f"保存配置失败: {str(e)}"


def delete_role_config(role_name: str) -> str:
    """删除角色配置文件
    
    Args:
        role_name: 角色名称
        
    Returns:
        成功或错误消息
    """
    config_path = Path("configs/roles") / f"{role_name}.json"
    if not config_path.exists():
        return f"角色 {role_name} 配置文件不存在"
    
    try:
        os.remove(config_path)
        return f"角色 {role_name} 配置已删除"
    except Exception as e:
        return f"删除配置失败: {str(e)}"


def test_role_synthesis(
    text: str,
    gpt_model: str,
    sovits_model: str,
    ref_audio: str,
    prompt_text: str,
    prompt_lang: str,
    text_lang: str,
    speed: float,
    ref_free: bool,
    if_sr: bool,
    top_k: int,
    top_p: float,
    temperature: float,
    sample_steps: int,
    cut_punc: str,
    role_name: str = "临时角色",
    aux_refs: List[str] = None
) -> str:
    """测试角色合成
    
    Args:
        text: 目标文本
        gpt_model: GPT模型路径
        sovits_model: SoVITS模型路径
        ref_audio: 参考音频路径
        prompt_text: 参考音频文本
        prompt_lang: 参考音频语言
        text_lang: 默认文本语言
        speed: 语速
        ref_free: 是否开启参考自由模式
        if_sr: 是否使用超分辨率
        top_k: top_k参数
        top_p: top_p参数
        temperature: 温度参数
        sample_steps: 采样步数
        cut_punc: 切分符号
        role_name: 角色名称
        aux_refs: 辅助参考音频列表
    
    Returns:
        合成的音频文件路径
    """
    if not text:
        raise gr.Error("请输入要合成的文本")
    
    if not ref_audio:
        raise gr.Error("请上传参考音频")
        
    if not gpt_model or not sovits_model:
        raise gr.Error("请选择GPT模型和SoVITS模型")
    
    # 检查参考音频是否存在
    if not os.path.exists(ref_audio):
        # 尝试查找可能的位置
        filename = os.path.basename(ref_audio)
        possible_locations = [
            os.path.join("configs/ref_audio", role_name, filename),
            os.path.join("configs/refsounds", filename)
        ]
        
        found = False
        for location in possible_locations:
            if os.path.exists(location):
                ref_audio = location
                found = True
                print(f"找到参考音频: {ref_audio}")
                break
                
        if not found:
            raise gr.Error(f"参考音频文件不存在: {ref_audio}")
    
    # 检查辅助参考音频
    valid_aux_refs = []
    if aux_refs:
        # 确保aux_refs是列表
        if not isinstance(aux_refs, list):
            # 单个文件情况，转为列表处理
            aux_refs = [aux_refs]
            
        for aux_ref in aux_refs:
            if aux_ref and os.path.exists(aux_ref):
                valid_aux_refs.append(aux_ref)
            else:
                # 尝试查找可能的位置
                filename = os.path.basename(aux_ref)
                possible_locations = [
                    os.path.join("configs/ref_audio", role_name, filename),
                    os.path.join("configs/refsounds", filename)
                ]
                
                found = False
                for location in possible_locations:
                    if os.path.exists(location):
                        valid_aux_refs.append(location)
                        found = True
                        print(f"找到辅助参考音频: {location}")
                        break
                        
                if not found and aux_ref:
                    print(f"警告: 辅助参考音频文件不存在: {aux_ref}")
    
    # 检查模型文件是否存在
    if not os.path.exists(gpt_model):
        raise gr.Error(f"GPT模型文件不存在: {gpt_model}")
    
    if not os.path.exists(sovits_model):
        raise gr.Error(f"SoVITS模型文件不存在: {sovits_model}")
    
    os.makedirs("output", exist_ok=True)
    output_path = os.path.join("output", f"output_{role_name}_{int(time.time())}.wav")
    
    try:
        # 设置模型
        try:
            response = requests.post(
                f"{API_URL}/set_model",
                json={
                    "gpt_model_path": gpt_model,
                    "sovits_model_path": sovits_model,
                },
                timeout=30,
            )
            if response.status_code != 200:
                error_msg = response.text
                try:
                    error_data = response.json()
                    if "message" in error_data:
                        error_msg = error_data["message"]
                except:
                    pass
                raise gr.Error(f"设置模型失败: {error_msg}")
        except requests.exceptions.RequestException as e:
            raise gr.Error(f"API请求失败: {str(e)}")
        
        # 准备API请求参数
        params = {
            "refer_wav_path": ref_audio,
            "prompt_text": prompt_text,
            "prompt_language": prompt_lang,
            "text": text,
            "text_language": text_lang,
            "speed": speed,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "sample_steps": sample_steps,
            "if_sr": if_sr,
            "cut_punc": cut_punc,
            "spk": role_name,
        }
        
        if valid_aux_refs:
            params["inp_refs"] = valid_aux_refs
            
        # 调用API
        response = requests.post(API_URL, json=params)
        if response.status_code != 200:
            error_msg = response.text
            try:
                error_data = response.json()
                if "message" in error_data:
                    error_msg = error_data["message"]
            except:
                pass
            raise RuntimeError(f"API调用失败: {error_msg}")
            
        # 保存音频
        with open(output_path, "wb") as f:
            f.write(response.content)
            
        return output_path
    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        print(f"合成异常: {trace}")
        raise gr.Error(f"合成失败: {str(e)}")


def create_ui():
    global g_default_role  # 添加这一行，声明使用全局变量
    
    with gr.Blocks(title="GPT-SoVITS API推理") as app:
        gr.Markdown("# GPT-SoVITS 文本转语音 (API版)")

        # 获取默认角色
        roles = list_roles()

        g_default_role = roles[0] if g_default_role not in roles else g_default_role

        # 初始化默认角色的模型
        gpt_path, sovits_path = init_models(g_default_role)

        with gr.Tab("文本文件"):
            with gr.Column():
                with gr.Row():
                    with gr.Column(scale=2):
                        text_content = gr.Textbox(
                            label="输入多行文本",
                            placeholder="每行一句话，支持以下格式：\n(角色)文本内容\n(角色|情绪)文本内容\n直接输入文本",
                            lines=8,
                        )
                    with gr.Column(scale=1):
                        file_input = gr.File(label="或上传文本文件", file_types=[".txt"])
                file_output = gr.Audio(label="输出音频")

            with gr.Row():
                roles = list_roles()  # 获取角色列表

                # 第一列：默认角色及其情绪
                with gr.Column():
                    default_role = gr.Dropdown(
                        label="默认角色（必选）",
                        choices=[(role, role) for role in roles],
                        value=g_default_role,
                        type="value",
                    )
                    initial_emotions = get_emotions(g_default_role)
                    default_emotion = gr.Dropdown(
                        label="默认情绪",  # 移除"可选"说明
                        choices=[(e, e) for e in initial_emotions],  # 移除"无"选项
                        value=initial_emotions[0]
                        if initial_emotions
                        else None,  # 默认选择第一个情绪
                        type="value",
                    )

                # 第二列：强制角色及其情绪
                with gr.Column():
                    force_role = gr.Dropdown(
                        label="强制使用角色（可选）",
                        choices=[("", "无")] + [(role, role) for role in roles],
                        value="",
                        type="value",
                    )
                    force_emotion = gr.Dropdown(
                        label="强制使用情绪（可选）",
                        choices=[("", "无")] + [(e, e) for e in initial_emotions],
                        value="",
                        type="value",
                    )

                # 第三列：语言选择
                with gr.Column():
                    text_lang = gr.Dropdown(
                        label="文本语言",
                        choices=LANGUAGE_OPTIONS,
                        value="中文",
                        type="value",
                    )

                # 第四列：切分符号
                with gr.Column():
                    cut_punc_input = gr.Textbox(
                        label="切分符号（可选）",
                        placeholder="例如：,.。，",
                        value="。！？：.!?:",
                    )
                    disable_parsing = gr.Checkbox(
                        label="禁用角色情绪解析",
                        value=True,
                        info="开启后将使用默认角色和情绪处理整个文本，不进行逐行解析",
                    )

            with gr.Row():            
                process_text_btn = gr.Button("处理文本", variant="primary")
                process_file_btn = gr.Button("处理文件", variant="secondary")
                preprocess_text_btn = gr.Button("预处理文本", variant="secondary")
                refresh_roles_btn = gr.Button("刷新角色列表", variant="secondary")  # 新增刷新按钮
       
            process_text_btn.click(
                process_text_content,
                inputs=[
                    text_content,
                    force_role,
                    default_role,
                    force_emotion,
                    default_emotion,
                    text_lang,
                    cut_punc_input,
                    disable_parsing,
                ],
                outputs=file_output,
            )

            process_file_btn.click(
                process_file,
                inputs=[
                    file_input,
                    force_role,
                    default_role,
                    force_emotion,
                    default_emotion,
                    text_lang,
                    cut_punc_input,
                    disable_parsing,
                ],
                outputs=file_output,
            )

            preprocess_text_btn.click(
                preprocess_text,
                inputs=[text_content, default_role, default_emotion],
                outputs=text_content,
            )

            # 添加刷新角色列表按钮的事件
            refresh_roles_btn.click(
                fn=lambda: gr.update(choices=[(role, role) for role in list_roles()]),
                inputs=[],
                outputs=[default_role]
            )

            # 更新角色切换事件
            default_role.change(
                update_default_emotions,
                inputs=[default_role],
                outputs=[default_emotion],
            )
            force_role.change(
                update_force_emotions,
                inputs=[force_role],
                outputs=[force_emotion],
            )

        with gr.Tab("单条文本"):
            with gr.Row():
                text_input = gr.Textbox(
                    label="输入文本", placeholder="请输入要转换的文本", lines=3
                )
                audio_output = gr.Audio(label="输出音频")

            with gr.Row():
                roles = list_roles()
                role = gr.Dropdown(
                    label="选择角色",
                    choices=[(role, role) for role in roles],
                    value=g_default_role,
                )
                # 初始化情绪列表
                initial_emotions = get_emotions(g_default_role)
                emotion = gr.Dropdown(
                    label="选择情绪",
                    choices=[(e, e) for e in initial_emotions],
                    value=initial_emotions[0] if initial_emotions else None,
                )
                text_lang = gr.Dropdown(
                    label="文本语言",
                    choices=LANGUAGE_OPTIONS,
                    value="中文",
                    type="value",
                )

                # 添加切分符号输入
                cut_punc_input = gr.Textbox(
                    label="切分符号（可选）",
                    placeholder="例如：,.。，…",
                    value="",
                )

            convert_btn = gr.Button("转换")
            convert_btn.click(
                process_text,
                inputs=[text_input, role, emotion, text_lang, cut_punc_input],
                outputs=audio_output,
            )

            # 修改单行文本模式的角色切换事件
            role.change(
                update_default_emotions,
                inputs=[role],
                outputs=[emotion],
            )

        with gr.Tab("角色管理合成"):
            gpt_models, sovits_models = get_model_lists()
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 模型选择")
                    gpt_model = gr.Dropdown(
                        label="GPT模型",
                        choices=gpt_models,
                        value=gpt_models[0] if gpt_models else None,
                        interactive=True
                    )
                    sovits_model = gr.Dropdown(
                        label="SoVITS模型",
                        choices=sovits_models,
                        value=sovits_models[0] if sovits_models else None,
                        interactive=True
                    )
                    refresh_models_btn = gr.Button("刷新模型列表")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 参考音频")
                    ref_audio = gr.Audio(
                        label="参考音频",
                        type="filepath",
                        interactive=True
                    )
                    prompt_text = gr.Textbox(
                        label="参考音频文本",
                        placeholder="参考音频对应的文本内容",
                        lines=2
                    )
                    aux_refs = gr.File(
                        label="辅助参考音频(可选)",
                        file_count="multiple", 
                        file_types=["audio"],
                        type="filepath",
                        interactive=True
                    )
                    
                with gr.Column(scale=1):
                    gr.Markdown("### 语言设置")
                    prompt_lang = gr.Dropdown(
                        label="参考音频语言",
                        choices=LANGUAGE_OPTIONS,
                        value="中文",
                        type="value"
                    )
                    text_lang = gr.Dropdown(
                        label="目标文本语言",
                        choices=LANGUAGE_OPTIONS,
                        value="中文",
                        type="value"
                    )
            
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 目标文本")
                    target_text = gr.Textbox(
                        label="目标文本",
                        placeholder="输入要合成的文本内容",
                        lines=4
                    )
                    cut_punc = gr.Textbox(
                        label="切分符号（可选）",
                        placeholder="例如：,.。，",
                        value="。！？：.!?:"
                    )
                    
                with gr.Column(scale=1):
                    gr.Markdown("### 参数配置")
                    with gr.Row():
                        with gr.Column(scale=1):
                            speed = gr.Slider(
                                label="语速",
                                minimum=0.5,
                                maximum=2.0,
                                value=1.0,
                                step=0.1
                            )
                            ref_free = gr.Checkbox(
                                label="参考自由模式",
                                value=False
                            )
                            if_sr = gr.Checkbox(
                                label="使用超分辨率",
                                value=False
                            )
                        
                        with gr.Column(scale=1):
                            top_k = gr.Slider(
                                label="Top-K",
                                minimum=1,
                                maximum=30,
                                value=15,
                                step=1
                            )
                            top_p = gr.Slider(
                                label="Top-P",
                                minimum=0.1,
                                maximum=1.0,
                                value=1.0,
                                step=0.05
                            )
                            temperature = gr.Slider(
                                label="温度",
                                minimum=0.1,
                                maximum=2.0,
                                value=1.0,
                                step=0.1
                            )
                            sample_steps = gr.Slider(
                                label="采样步数",
                                minimum=16,
                                maximum=64,
                                value=32,
                                step=8
                            )
                            pause_second = gr.Slider(
                                label="停顿秒数",
                                minimum=0.1,
                                maximum=1.0,
                                value=0.3,
                                step=0.1
                            )
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 角色信息")
                    role_name = gr.Textbox(
                        label="角色名称",
                        placeholder="输入要保存的角色名称",
                        value=""
                    )
                    description = gr.Textbox(
                        label="角色描述(可选)",
                        placeholder="输入角色描述，如：性别，声音特点等",
                        lines=2
                    )
                    
                with gr.Column(scale=1):
                    gr.Markdown("### 操作")
                    create_btn = gr.Button("新建/更新角色", variant="primary")
                    role_list = gr.Dropdown(
                        label="现有角色列表",
                        choices=[(role, role) for role in roles],
                        value=g_default_role if g_default_role in roles else None,
                        type="value"
                    )
                    with gr.Row():
                        load_role_btn = gr.Button("加载角色")
                        delete_role_btn = gr.Button("删除角色", variant="stop")
                    refresh_role_list_btn = gr.Button("刷新角色列表", variant="secondary")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 合成结果")
                    synthesis_btn = gr.Button("合成测试", variant="primary")
                    synthesis_output = gr.Audio(label="合成结果")
                    status_text = gr.Markdown("")
            
            # 刷新模型列表
            refresh_models_btn.click(
                fn=lambda: (
                    gr.update(choices=get_model_lists()[0]),
                    gr.update(choices=get_model_lists()[1])
                ),
                inputs=[],
                outputs=[gpt_model, sovits_model]
            )
            
            # 从文件名自动提取参考文本
            def extract_text_from_filename(file_path):
                if not file_path:
                    return ""
                file_name = os.path.basename(file_path)
                name_without_ext = os.path.splitext(file_name)[0]
                return name_without_ext
            
            ref_audio.change(
                fn=extract_text_from_filename,
                inputs=[ref_audio],
                outputs=[prompt_text]
            )
            
            # 辅助参考音频处理函数
            def process_aux_refs(aux_refs_list):
                """处理辅助参考音频列表，提取文件路径"""
                if not aux_refs_list:
                    return []
                
                processed_refs = []
                if isinstance(aux_refs_list, list):
                    for item in aux_refs_list:
                        if isinstance(item, dict) and "name" in item:
                            processed_refs.append(item["name"])
                        elif isinstance(item, str):
                            processed_refs.append(item)
                elif isinstance(aux_refs_list, dict) and "name" in aux_refs_list:
                    processed_refs.append(aux_refs_list["name"])
                elif isinstance(aux_refs_list, str):
                    processed_refs.append(aux_refs_list)
                
                return processed_refs
            
            # 保存角色配置
            create_btn.click(
                fn=lambda role_name, gpt_model, sovits_model, ref_audio, prompt_text, prompt_lang, text_lang, 
                       speed, ref_free, if_sr, top_k, top_p, temperature, sample_steps, pause_second, 
                       description, aux_refs: save_role_config(
                    role_name, gpt_model, sovits_model, ref_audio, prompt_text, prompt_lang, text_lang,
                    speed, ref_free, if_sr, top_k, top_p, temperature, sample_steps, pause_second, 
                    description, process_aux_refs(aux_refs)
                ),
                inputs=[
                    role_name, gpt_model, sovits_model, 
                    ref_audio, prompt_text, prompt_lang, text_lang,
                    speed, ref_free, if_sr, top_k, top_p, 
                    temperature, sample_steps, pause_second, description, aux_refs
                ],
                outputs=[status_text]
            ).then(
                fn=lambda: (
                    gr.update(choices=[(role, role) for role in list_roles()])
                ),
                inputs=[],
                outputs=[role_list]
            )
            
            synthesis_btn.click(
                fn=lambda target_text, gpt_model, sovits_model, ref_audio, prompt_text, 
                        prompt_lang, text_lang, speed, ref_free, if_sr, top_k, top_p,
                        temperature, sample_steps, cut_punc, role_name, aux_refs: test_role_synthesis(
                    target_text, gpt_model, sovits_model, ref_audio, prompt_text, 
                    prompt_lang, text_lang, speed, ref_free, if_sr, top_k, top_p,
                    temperature, sample_steps, cut_punc, role_name, process_aux_refs(aux_refs)
                ),
                inputs=[
                    target_text, gpt_model, sovits_model,
                    ref_audio, prompt_text, prompt_lang, text_lang,
                    speed, ref_free, if_sr, top_k, top_p,
                    temperature, sample_steps, cut_punc, role_name, aux_refs
                ],
                outputs=[synthesis_output]
            )

            # 加载角色配置
            load_role_btn.click(
                fn=lambda role: load_and_process_role_config(role, process_aux_refs),
                inputs=[role_list],
                outputs=[
                    gpt_model, sovits_model, ref_audio, prompt_text, aux_refs,
                    prompt_lang, text_lang, speed, ref_free, if_sr,
                    top_k, top_p, temperature, sample_steps, pause_second,
                    description, status_text
                ]
            )
            
            # 删除角色配置
            delete_role_btn.click(
                fn=delete_role_config,
                inputs=[role_list],
                outputs=[status_text]
            ).then(
                fn=lambda: (
                    gr.update(choices=[(role, role) for role in list_roles()])
                ),
                inputs=[],
                outputs=[role_list]
            )
            
            # 刷新角色列表
            refresh_role_list_btn.click(
                fn=lambda: (
                    gr.update(choices=[(role, role) for role in list_roles()])
                ),
                inputs=[],
                outputs=[role_list]
            )

        with gr.Tab("配置"):
            with gr.Tab("词语替换"):
                word_replace_text = gr.TextArea(
                    label="词语替换配置",
                    value=load_word_replace_config(),
                    lines=20,
                    max_lines=50
                )
                
                with gr.Row():
                    save_word_replace_btn = gr.Button("保存词语替换配置")
                    refresh_word_replace_btn = gr.Button("刷新")
                
                save_word_replace_btn.click(
                    fn=save_word_replace_config,
                    inputs=[word_replace_text],
                    outputs=[gr.Markdown("保存成功！")]
                )
                
                refresh_word_replace_btn.click(
                    fn=load_word_replace_config,
                    inputs=[],
                    outputs=[word_replace_text]
                )

        gr.Markdown("""
        ## 使用说明
        1. **单条文本**：直接输入文本，选择角色和情绪，点击转换
        2. **文本文件**：可以选择以下两种方式之一：
           - 上传文本文件
           - 直接在文本框中输入多行文本
           支持以下格式：
           - `(角色)文本内容`
           - `(角色|情绪)文本内容`
           - 直接输入文本（需要设置默认角色）
        3. **强制角色**：忽略文本中的角色标记，全部使用指定角色
        4. **默认角色**：当文本没有指定角色时使用的角色
        5. **预处理文本**：将双引号内的文本作为对白（使用默认情绪），其他文本作为叙述
        """)

    return app

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

def load_and_process_role_config(role, process_aux_refs_func=None):
    """加载角色配置并处理辅助参考音频
    
    Args:
        role: 角色名称
        process_aux_refs_func: 处理辅助参考音频的函数，默认为None
        
    Returns:
        加载的角色配置列表
    """
    if not role:
        return [gr.update()] * 16 + ["请选择角色"]
    
    try:
        role_path = Path("configs/roles") / f"{role}.json"
        if not role_path.exists():
            return [gr.update()] * 16 + [f"找不到角色配置文件: {role}"]
        
        with open(role_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # 获取第一个情绪配置
        if "emotions" not in config or not config["emotions"]:
            return [gr.update()] * 16 + [f"角色 {role} 没有情绪配置"]
        
        first_emotion = next(iter(config["emotions"].values()))
        
        # 提取配置参数
        gpt_path = config.get("gpt_path", "")
        sovits_path = config.get("sovits_path", "")
        ref_audio = first_emotion.get("ref_audio", "")
        prompt_text = first_emotion.get("prompt_text", "")
        aux_refs = first_emotion.get("aux_refs", [])
        prompt_lang = config.get("prompt_lang", "中文")
        text_lang = config.get("text_lang", "中文")
        speed = config.get("speed", 1.0)
        ref_free = config.get("ref_free", False)
        if_sr = config.get("if_sr", False)
        top_k = config.get("top_k", 15)
        top_p = config.get("top_p", 1.0)
        temperature = config.get("temperature", 1.0)
        sample_steps = config.get("sample_steps", 32)
        pause_second = config.get("pause_second", 0.3)
        description = config.get("description", "")
        
        # 检查文件是否存在
        warning_msg = ""
        
        # 检查参考音频
        if not os.path.exists(ref_audio):
            # 尝试在旧目录中查找文件
            old_ref_audio = ref_audio
            filename = os.path.basename(ref_audio)
            ref_audio_alternatives = [
                os.path.join("configs/ref_audio", role, filename),
                os.path.join("configs/refsounds", filename)
            ]
            
            ref_audio = None
            for alt_path in ref_audio_alternatives:
                if os.path.exists(alt_path):
                    ref_audio = alt_path
                    warning_msg += f"已重定向参考音频: {old_ref_audio} → {ref_audio}\n"
                    break
            
            if not ref_audio:
                warning_msg += f"警告: 参考音频文件不存在: {old_ref_audio}\n"
                ref_audio = old_ref_audio  # 保留原路径，即使不存在
        
        # 检查GPT模型
        if not os.path.exists(gpt_path):
            warning_msg += f"警告: GPT模型文件不存在: {gpt_path}\n"
        
        # 检查SoVITS模型
        if not os.path.exists(sovits_path):
            warning_msg += f"警告: SoVITS模型文件不存在: {sovits_path}\n"
        
        # 检查辅助参考音频
        aux_refs_to_use = []
        if aux_refs:
            # 处理辅助参考音频列表
            if isinstance(aux_refs, list):
                for aux_ref in aux_refs:
                    if os.path.exists(aux_ref):
                        aux_refs_to_use.append(aux_ref)
                    else:
                        # 尝试重定向
                        old_aux_ref = aux_ref
                        filename = os.path.basename(aux_ref)
                        aux_ref_alternatives = [
                            os.path.join("configs/ref_audio", role, filename),
                            os.path.join("configs/refsounds", filename)
                        ]
                        
                        found = False
                        for alt_path in aux_ref_alternatives:
                            if os.path.exists(alt_path):
                                aux_refs_to_use.append(alt_path)
                                found = True
                                warning_msg += f"已重定向辅助参考音频: {old_aux_ref} → {alt_path}\n"
                                break
                        
                        if not found:
                            warning_msg += f"警告: 辅助参考音频文件不存在: {old_aux_ref}\n"
            elif aux_refs and isinstance(aux_refs, str):
                # 处理单个辅助参考音频
                if os.path.exists(aux_refs):
                    aux_refs_to_use.append(aux_refs)
                else:
                    # 尝试重定向
                    old_aux_ref = aux_refs
                    filename = os.path.basename(aux_refs)
                    aux_ref_alternatives = [
                        os.path.join("configs/ref_audio", role, filename),
                        os.path.join("configs/refsounds", filename)
                    ]
                    
                    found = False
                    for alt_path in aux_ref_alternatives:
                        if os.path.exists(alt_path):
                            aux_refs_to_use.append(alt_path)
                            found = True
                            warning_msg += f"已重定向辅助参考音频: {old_aux_ref} → {alt_path}\n"
                            break
                    
                    if not found:
                        warning_msg += f"警告: 辅助参考音频文件不存在: {old_aux_ref}\n"
        
        # 构建状态消息
        status = f"角色 {role} 配置已加载"
        if warning_msg:
            status = warning_msg + status
        
        return [
            gr.update(value=gpt_path),
            gr.update(value=sovits_path),
            gr.update(value=ref_audio),
            gr.update(value=prompt_text),
            gr.update(value=aux_refs_to_use),
            gr.update(value=prompt_lang),
            gr.update(value=text_lang),
            gr.update(value=speed),
            gr.update(value=ref_free),
            gr.update(value=if_sr),
            gr.update(value=top_k),
            gr.update(value=top_p),
            gr.update(value=temperature),
            gr.update(value=sample_steps),
            gr.update(value=pause_second),
            gr.update(value=description),
            status
        ]
    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        print(f"加载配置异常: {trace}")
        return [gr.update()] * 16 + [f"加载配置失败: {str(e)}"]

if __name__ == "__main__":
    app = create_ui()
    app.launch(server_name="0.0.0.0", server_port=7878, inbrowser=True)
