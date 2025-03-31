import os
import time
from io import BytesIO
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import requests
import librosa
import soundfile as sf
from numpy.typing import NDArray

from ui.utils import API_URL, clean_text


def call_api(text: str, role_config: Dict[str, Any], role_name: str, cut_punc: str = "") -> bytes:
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


def get_formatted_filename(role_name: str, text: str, emotion: str = "") -> str:
    """生成格式化的文件名：角色名_时间戳_文本内容前20个字符
    
    Args:
        role_name: 角色名称
        text: 文本内容
        emotion: 情绪（可选）
    
    Returns:
        格式化的文件名
    """
    # 提取前20个字符，去除可能导致文件名问题的字符
    if text:
        short_text = text[:20].strip()
        # 替换不适合作为文件名的字符
        for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
            short_text = short_text.replace(char, '_')
    else:
        short_text = "无文本"
    
    # 生成文件名
    emotion_part = f"_{emotion}" if emotion else ""
    return f"{role_name}{emotion_part}_{int(time.time())}_{short_text}"


def process_text(
    text: str,
    role: str,
    emotion: str = "",
    text_lang: str = "中文",
    cut_punc: str = "",
    output_dir: str = "output",
) -> str:
    """处理单条文本"""
    from ui.models import init_models
    from ui.roles import get_role_config
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用新的文件名格式
    filename = get_formatted_filename(role, text, emotion)
    output_path = os.path.join(output_dir, f"{filename}.wav")

    try:
        # 确保使用正确的模型
        init_models(role)

        role_config = get_role_config(role, emotion, text_lang)
        audio_data = call_api(text, role_config, role, cut_punc=cut_punc)

        with open(output_path, "wb") as f:
            f.write(audio_data)

        return output_path
    except Exception as e:
        import gradio as gr
        raise gr.Error(f"处理失败: {str(e)}")


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
    aux_refs: List[str] = None,
    emotion: str = None
) -> str:
    """测试角色合成"""
    import gradio as gr
    
    if not text:
        raise gr.Error("请输入要合成的文本")
    
    if not ref_audio:
        raise gr.Error("请上传参考音频")
        
    if not gpt_model or not sovits_model:
        raise gr.Error("请选择GPT模型和SoVITS模型")
    
    # 检查参考音频是否存在
    if not os.path.exists(ref_audio):
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
           
    if not valid_aux_refs:
        print(f"没有找到辅助参考音频: {aux_refs}")
    
    # 检查模型文件是否存在
    if not os.path.exists(gpt_model):
        raise gr.Error(f"GPT模型文件不存在: {gpt_model}")
    
    if not os.path.exists(sovits_model):
        raise gr.Error(f"SoVITS模型文件不存在: {sovits_model}")
    
    os.makedirs("output", exist_ok=True)
    
    # 使用新的文件名格式
    filename = get_formatted_filename(role_name, text, emotion)
    output_path = os.path.join("output", f"{filename}.wav")
    
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
        
        # 构建角色配置字典，与call_api兼容
        role_config = {
            "ref_audio": ref_audio,
            "prompt_text": prompt_text,
            "prompt_lang": prompt_lang,
            "text_lang": text_lang,
            "speed": speed,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "sample_steps": sample_steps,
            "if_sr": if_sr
        }
        
        # 添加辅助参考音频
        if valid_aux_refs:
            role_config["aux_refs"] = valid_aux_refs
        
        # 添加情绪信息
        if emotion:
            role_config["emotion"] = emotion
            
        # 调用现有的API函数，避免代码重复
        audio_data = call_api(text, role_config, role_name, cut_punc=cut_punc)
            
        # 保存音频
        with open(output_path, "wb") as f:
            f.write(audio_data)
            
        return output_path
    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        print(f"合成异常: {trace}")
        raise gr.Error(f"合成失败: {str(e)}") 