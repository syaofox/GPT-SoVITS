import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import gradio as gr


def list_roles() -> List[str]:
    """获取所有可用的角色列表"""
    roles_dir = Path("configs/roles")
    roles = []
    
    # 确保目录存在
    if not roles_dir.exists():
        roles_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有子目录作为角色
    for item in roles_dir.iterdir():
        if item.is_dir():
            # 检查config.json是否存在
            if (item / "config.json").exists():
                roles.append(item.name)
    
    roles = sorted(roles)
    if not roles:
        raise gr.Error("未找到任何角色配置文件，请确保 configs/roles 目录下有角色子目录，且包含config.json文件")
    return roles


def get_emotions(role: str = "") -> List[str]:
    """获取情绪列表，从角色配置文件中读取"""
    if not role:
        return []

    # 读取角色配置
    role_path = Path("configs/roles") / role / "config.json"
    if not role_path.exists():
        return []

    with open(role_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if "emotions" not in config:
        return []

    # 直接返回配置文件中定义的情绪列表
    return list(config["emotions"].keys())


def get_role_config(role: str, emotion: str = "", text_lang: str = "中文") -> Dict[str, Any]:
    """获取角色配置"""
    role_dir = Path("configs/roles") / role
    role_path = role_dir / "config.json"
    
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

    # 处理参考音频路径 - 如果是相对路径，则相对于角色目录
    if "ref_audio" in result_config:
        ref_audio = result_config["ref_audio"]
        ref_path = Path(ref_audio)
        
        # 如果是相对路径或只是文件名，则视为相对于角色目录
        if not ref_path.is_absolute() and not ref_path.exists():
            role_ref_path = role_dir / ref_audio
            if role_ref_path.exists():
                result_config["ref_audio"] = str(role_ref_path)
            
    # 处理辅助参考音频路径
    if "aux_refs" in result_config and isinstance(result_config["aux_refs"], list):
        resolved_aux_refs = []
        for aux_ref in result_config["aux_refs"]:
            aux_path = Path(aux_ref)
            # 如果是相对路径或只是文件名，则视为相对于角色目录
            if not aux_path.is_absolute() and not aux_path.exists():
                role_aux_path = role_dir / aux_ref
                if role_aux_path.exists():
                    resolved_aux_refs.append(str(role_aux_path))
                else:
                    resolved_aux_refs.append(aux_ref)  # 保持原始路径
            else:
                resolved_aux_refs.append(aux_ref)
        result_config["aux_refs"] = resolved_aux_refs

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
    """保存角色配置到角色目录的config.json文件"""
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
    
    # 创建角色目录
    role_dir = Path("configs/roles") / role_name
    role_dir.mkdir(parents=True, exist_ok=True)
    
    # 提取参考音频文件名作为情绪名称和目标文件名
    orig_audio_filename = os.path.basename(ref_audio)
    emotion_name = os.path.splitext(orig_audio_filename)[0]
    if len(emotion_name) > 10:  # 如果名称太长，截取一部分
        short_emotion_name = emotion_name[:10]
    else:
        short_emotion_name = emotion_name
    
    # 复制参考音频文件到角色目录
    target_ref_audio = str(role_dir / orig_audio_filename)
    try:
        if not os.path.exists(target_ref_audio):
            shutil.copy2(ref_audio, target_ref_audio)
            print(f"复制参考音频: {ref_audio} -> {target_ref_audio}")
        else:
            print(f"参考音频已存在: {target_ref_audio}")
    except Exception as e:
        return f"复制参考音频失败: {str(e)}"
    
    # 使用相对路径保存到配置文件中
    rel_ref_audio = orig_audio_filename
    
    # 处理辅助参考音频
    aux_refs_copied = []
    rel_aux_refs = []
    if aux_refs:
        # 处理aux_refs为字符串的情况（来自文本框，每行一个路径）
        if isinstance(aux_refs, str):
            for line in aux_refs.splitlines():
                line = line.strip()
                if line and os.path.exists(line):
                    aux_filename = os.path.basename(line)
                    target_aux_ref = str(role_dir / aux_filename)
                    try:
                        if not os.path.exists(target_aux_ref):
                            shutil.copy2(line, target_aux_ref)                            
                            print(f"复制辅助参考音频: {line} -> {target_aux_ref}")
                        else:
                            print(f"辅助参考音频已存在: {target_aux_ref}")
                        # 保存相对路径
                        rel_aux_refs.append(aux_filename)
                        # 保存绝对路径用于写入配置
                        aux_refs_copied.append(target_aux_ref)
                    except Exception as e:
                        print(f"复制辅助参考音频失败: {str(e)}")
        # 兼容列表形式（旧格式或从process_aux_refs函数处理后的结果）
        elif isinstance(aux_refs, (list, tuple)) and len(aux_refs) > 0:
            # 处理多个辅助参考音频文件
            for aux_ref in aux_refs:
                if aux_ref and os.path.exists(aux_ref):
                    aux_filename = os.path.basename(aux_ref)
                    target_aux_ref = str(role_dir / aux_filename)
                    try:
                        if not os.path.exists(target_aux_ref):
                            shutil.copy2(aux_ref, target_aux_ref)
                            print(f"复制辅助参考音频: {aux_ref} -> {target_aux_ref}")
                        else:
                            print(f"辅助参考音频已存在: {target_aux_ref}")
                        # 保存相对路径
                        rel_aux_refs.append(aux_filename)
                        # 保存绝对路径用于写入配置
                        aux_refs_copied.append(target_aux_ref)
                    except Exception as e:
                        print(f"复制辅助参考音频失败: {str(e)}")
        elif aux_refs and os.path.exists(aux_refs):  # 处理单个文件情况
            # 处理单个辅助参考音频
            aux_filename = os.path.basename(aux_refs)
            target_aux_ref = str(role_dir / aux_filename)
            try:
                if not os.path.exists(target_aux_ref):
                    shutil.copy2(aux_refs, target_aux_ref)
                    print(f"复制辅助参考音频: {aux_refs} -> {target_aux_ref}")
                else:
                    print(f"辅助参考音频已存在: {target_aux_ref}")
                # 保存相对路径
                rel_aux_refs.append(aux_filename)
                # 保存绝对路径用于写入配置
                aux_refs_copied.append(target_aux_ref)
            except Exception as e:
                print(f"复制辅助参考音频失败: {str(e)}")
    
    # 构建配置数据
    config = {
        "version": version,
        "emotions": {
            short_emotion_name: {
                "ref_audio": rel_ref_audio,  # 使用相对路径
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
    if rel_aux_refs:
        config["emotions"][short_emotion_name]["aux_refs"] = rel_aux_refs
    
    # 配置文件路径
    config_path = role_dir / "config.json"
    
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
    """删除角色配置及相关文件"""
    role_dir = Path("configs/roles") / role_name
    if not role_dir.exists():
        return f"角色 {role_name} 目录不存在"
    
    try:
        # 删除整个角色目录及其中的所有文件
        shutil.rmtree(role_dir)
        return f"角色 {role_name} 配置及相关文件已删除"
    except Exception as e:
        return f"删除角色失败: {str(e)}"


def load_and_process_role_config(role, emotion=None, process_aux_refs_func=None):
    """加载角色配置并处理辅助参考音频"""
    if not role:
        return [gr.update()] * 16 + ["请选择角色"]
    
    try:
        role_dir = Path("configs/roles") / role
        role_path = role_dir / "config.json"
        
        if not role_path.exists():
            # 尝试使用旧格式路径作为后备
            old_role_path = Path("configs/roles") / f"{role}.json"
            if old_role_path.exists():
                role_path = old_role_path
            else:
                return [gr.update()] * 16 + [f"找不到角色配置文件: {role}"]
        
        with open(role_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # 检查角色是否有情绪配置
        if "emotions" not in config or not config["emotions"]:
            return [gr.update()] * 16 + [f"角色 {role} 没有情绪配置"]
        
        emotions = config["emotions"]
        
        # 如果没有指定情绪，或者指定的情绪不存在，使用第一个情绪
        if not emotion or emotion not in emotions:
            emotion_key = next(iter(emotions.keys()))
        else:
            emotion_key = emotion
            
        emotion_config = emotions[emotion_key]
        
        # 提取配置参数
        gpt_path = config.get("gpt_path", "")
        sovits_path = config.get("sovits_path", "")
        ref_audio = emotion_config.get("ref_audio", "")
        prompt_text = emotion_config.get("prompt_text", "")
        aux_refs = emotion_config.get("aux_refs", [])
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
        
        # 处理参考音频路径
        if not os.path.isabs(ref_audio) and not os.path.exists(ref_audio):
            # 尝试将相对路径转换为相对于角色目录的绝对路径
            role_ref_path = role_dir / ref_audio
            if role_ref_path.exists():
                ref_audio = str(role_ref_path)
        
        # 处理辅助参考音频路径
        if isinstance(aux_refs, list):
            resolved_aux_refs = []
            for aux_ref in aux_refs:
                # 如果是相对路径，尝试转换为相对于角色目录的绝对路径
                if not os.path.isabs(aux_ref) and not os.path.exists(aux_ref):
                    role_aux_path = role_dir / aux_ref
                    if role_aux_path.exists():
                        resolved_aux_refs.append(str(role_aux_path))
                    else:
                        resolved_aux_refs.append(aux_ref)
                else:
                    resolved_aux_refs.append(aux_ref)
            aux_refs = resolved_aux_refs
        
        # 检查文件是否存在
        warning_msg = ""
        
        # 检查参考音频
        if not os.path.exists(ref_audio):
            warning_msg += f"警告: 参考音频文件不存在: {ref_audio}\n"            
        
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
                    
                        
                       
            elif aux_refs and isinstance(aux_refs, str):
                # 处理单个辅助参考音频
                if os.path.exists(aux_refs):
                    aux_refs_to_use.append(aux_refs)
                
        # 构建状态消息
        status = f"角色 {role} 的情绪 {emotion_key} 配置已加载"
        if warning_msg:
            status = warning_msg + status
        
        # 将辅助参考音频列表直接转换为文本格式（每行一个路径）
        # 这样就不需要依赖后续的.then处理
        aux_refs_text = "\n".join(aux_refs_to_use) if aux_refs_to_use else ""
        
        return [
            gr.update(value=gpt_path),
            gr.update(value=sovits_path),
            gr.update(value=ref_audio),
            gr.update(value=prompt_text),
            gr.update(value=aux_refs_text),  # 直接返回文本格式
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


def get_role_name_from_file_path(file_path):
    """从文件路径中提取角色名称"""
    # 提取文件名，不包含扩展名
    file_name = os.path.basename(file_path)
    file_name = os.path.splitext(file_name)[0]
    return file_name 