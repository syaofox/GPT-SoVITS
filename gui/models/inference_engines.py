"""
推理引擎模块

提供基础推理引擎和多角色推理引擎实现
"""

import os
import re
import uuid
import json
from typing import Dict, List, Tuple, Callable, Optional, Any, Union
import soundfile as sf
import numpy as np
from pathlib import Path

# 导入推理模块
from gpt_sovits_inference import GPTSoVITSInference


class RoleTextParser:
    """角色文本解析工具"""
    
    @staticmethod
    def parse_multi_role_text(text: str) -> List[Dict]:
        """
        解析多角色文本
        
        参数:
            text: 包含多角色标记的文本
            
        返回:
            解析后的角色和文本列表，格式为 [{"role": "角色名", "emotion": "情绪", "text": "文本内容"}, ...]
        """
        # 正则表达式匹配 <角色名|情绪> 格式
        pattern = r'<([^|>]+)\|([^>]+)>(.*?)(?=<[^|>]+\|[^>]+>|$)'
        
        # 查找所有匹配项
        matches = re.findall(pattern, text, re.DOTALL)
        
        # 如果没有匹配到角色定义，则认为是单角色文本
        if not matches:
            return [{"role": None, "emotion": None, "text": text.strip()}]
        
        result = []
        for role, emotion, content in matches:
            if content.strip():  # 只添加非空内容
                result.append({
                    "role": role.strip(),
                    "emotion": emotion.strip(),
                    "text": content.strip()
                })
        
        return result


class RoleConfigManager:
    """角色配置管理器"""
    
    @staticmethod
    def get_role_config(role_name: str, emotion_name: str) -> Dict:
        """
        获取角色配置
        
        参数:
            role_name: 角色名称
            emotion_name: 情绪名称
            
        返回:
            角色配置字典
        """
        # 角色配置目录
        roles_dir = Path("gui/roles")
        
        # 构建角色配置文件路径
        role_dir = roles_dir / role_name
        config_path = role_dir / "config.json"
        
        # 检查配置文件是否存在
        if not config_path.exists():
            # 尝试寻找包含该角色名称的目录
            alternative_dirs = [d for d in roles_dir.glob("*") if d.is_dir() and role_name in d.name]
            
            # 如果找到了可能的匹配目录，使用第一个
            if alternative_dirs:
                role_dir = alternative_dirs[0]
                config_path = role_dir / "config.json"
                print(f"获取角色配置: {role_dir.name}/{emotion_name}")
            else:
                # 未找到匹配的角色目录
                print(f"警告: 未找到角色 '{role_name}' 的配置目录")
                return {}
        else:
            print(f"获取角色配置: {role_name}/{emotion_name}")
        
        try:
            # 读取配置文件
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                
                # 获取指定情绪的配置
                if "emotions" in config_data and emotion_name in config_data["emotions"]:
                    emotion_config = config_data["emotions"][emotion_name]
                    
                    # 确保ref_audio路径是正确的
                    if "ref_audio" in emotion_config and not os.path.isabs(emotion_config["ref_audio"]):
                        # 如果是相对路径，转换为相对于角色目录的完整路径
                        rel_path = emotion_config["ref_audio"]
                        if not os.path.exists(rel_path):
                            # 尝试解析为相对于角色目录的路径
                            full_path = str(role_dir / rel_path)
                            if os.path.exists(full_path):
                                emotion_config["ref_audio"] = full_path
                    
                    # 打印配置中的模型信息
                    print(f"配置中的模型: gpt={emotion_config.get('gpt_path', '未指定')}, sovits={emotion_config.get('sovits_path', '未指定')}")
                    
                    return emotion_config
                
                # 如果找不到指定情绪，尝试使用任意一个可用的情绪配置
                elif "emotions" in config_data and config_data["emotions"]:
                    # 获取第一个可用的情绪配置
                    first_emotion = next(iter(config_data["emotions"].values()))
                    print(f"警告: 未找到情绪 '{emotion_name}' 的配置，使用默认情绪配置")
                    
                    # 确保ref_audio路径是正确的
                    if "ref_audio" in first_emotion and not os.path.isabs(first_emotion["ref_audio"]):
                        # 如果是相对路径，转换为相对于角色目录的完整路径
                        rel_path = first_emotion["ref_audio"]
                        if not os.path.exists(rel_path):
                            # 尝试解析为相对于角色目录的路径
                            full_path = str(role_dir / rel_path)
                            if os.path.exists(full_path):
                                first_emotion["ref_audio"] = full_path
                    
                    # 打印配置中的模型信息
                    print(f"配置中的模型: gpt={first_emotion.get('gpt_path', '未指定')}, sovits={first_emotion.get('sovits_path', '未指定')}")
                    
                    return first_emotion
        except Exception as e:
            print(f"读取角色配置失败: {e}")
        
        return {}


class BaseInferenceEngine:
    """推理引擎基类"""
    
    def __init__(self, output_dir: str = "output"):
        """
        初始化推理引擎基类
        
        参数:
            output_dir: 输出音频文件目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate(self, config: Dict, progress_callback: Callable = None) -> Tuple[bool, str]:
        """
        生成语音
        
        参数:
            config: 推理配置
            progress_callback: 进度回调函数
            
        返回:
            (成功标志, 结果路径或错误信息)
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def sanitize_filename(self, text):
        """处理文件名以确保合法性"""
        # 确保输入是字符串
        if not isinstance(text, str):
            text = str(text)
            
        # 替换非法字符为下划线
        illegal_chars = r'[\\/*?:"<>|]'
        sanitized = re.sub(illegal_chars, '_', text)
        # 移除空格和换行符
        sanitized = sanitized.replace(' ', '_').replace('\n', '_').replace('\r', '_')
        # 替换连续的下划线为单个下划线
        sanitized = re.sub(r'_+', '_', sanitized)
        # 限制长度
        return sanitized[:30] if len(sanitized) > 30 else sanitized


class SingleRoleInferenceEngine(BaseInferenceEngine):
    """单角色推理引擎"""
    
    def __init__(self, output_dir: str = "output"):
        """初始化单角色推理引擎"""
        super().__init__(output_dir)
        self.inference_engine = None
        self.current_gpt_path = ""
        self.current_sovits_path = ""
        self.should_stop = False
    
    def set_stop_flag(self, should_stop: bool):
        """设置停止标志"""
        self.should_stop = should_stop
    
    def get_inference_engine(self, gpt_path: str, sovits_path: str) -> GPTSoVITSInference:
        """
        获取推理引擎实例
        
        参数:
            gpt_path: GPT模型路径
            sovits_path: SoVITS模型路径
            
        返回:
            GPTSoVITSInference实例
        """
        # 检查当前引擎是否可以复用
        if (self.inference_engine is not None and 
            gpt_path == self.current_gpt_path and 
            sovits_path == self.current_sovits_path):
            return self.inference_engine
        
        # 创建新的推理引擎
        try:
            engine = GPTSoVITSInference(
                gpt_path=gpt_path,
                sovits_path=sovits_path
            )
            self.inference_engine = engine
            self.current_gpt_path = gpt_path
            self.current_sovits_path = sovits_path
            return engine
        except Exception as e:
            raise RuntimeError(f"初始化推理引擎失败: {str(e)}")
    
    def progress_wrapper(self, callback, prefix=""):
        """
        进度回调包装函数
        
        参数:
            callback: 原始回调函数
            prefix: 进度信息前缀
            
        返回:
            包装后的回调函数
        """
        def wrapped_callback(current_segment, total_segments=None):
            # 检查是否应该停止
            if self.should_stop:
                return True
            
            if callback:
                # 确保类型是数字
                if current_segment is not None:
                    current_segment = int(current_segment) if isinstance(current_segment, (int, float)) else 0
                
                if total_segments is not None:
                    total_segments = int(total_segments) if isinstance(total_segments, (int, float)) else 0
                
                # 添加前缀信息
                if prefix and current_segment > 0:
                    if total_segments:
                        callback(f"{prefix} {current_segment}/{total_segments}")
                    else:
                        callback(f"{prefix} {current_segment}")
                # 对于处理分段的初始通知，直接调用原始回调
                return callback(current_segment, total_segments)
            return False
        
        return wrapped_callback
    
    def generate(self, config: Dict, progress_callback: Callable = None) -> Tuple[bool, str]:
        """
        生成语音
        
        参数:
            config: 推理配置
            progress_callback: 进度回调函数
            
        返回:
            (成功标志, 结果路径或错误信息)
        """
        gpt_path = config.get("gpt_path")
        sovits_path = config.get("sovits_path")
        
        if not gpt_path or not sovits_path:
            return False, "模型路径未指定"
        
        # 重置停止标志
        self.should_stop = False
        
        try:
            # 获取或初始化推理引擎
            engine = self.get_inference_engine(gpt_path, sovits_path)
            
            # 准备进度回调包装
            wrapped_callback = None
            if progress_callback:
                progress_callback("准备生成语音...")
                wrapped_callback = lambda current, total=None: self.progress_wrapper(
                    lambda msg, _=None: progress_callback(str(msg)) if msg is not None else None,
                    ""
                )(current, total)
            
            # 生成语音
            sample_rate, audio_data = engine.generate_speech(
                ref_wav_path=config.get("ref_audio", ""),
                prompt_text=config.get("prompt_text", ""),
                prompt_language=config.get("prompt_lang", "中文"),
                text=config.get("text", ""),
                text_language=config.get("text_lang", "中文"),
                how_to_cut=config.get("how_to_cut", "按中文句号。切"),
                top_k=int(config.get("top_k", 20)),
                top_p=float(config.get("top_p", 0.6)),
                temperature=float(config.get("temperature", 0.6)),
                ref_free=bool(config.get("ref_free", False)),
                speed=float(config.get("speed", 1.0)),
                if_freeze=bool(config.get("if_freeze", False)),
                inp_refs=config.get("aux_refs", []),
                sample_steps=int(config.get("sample_steps", 8)),
                if_sr=bool(config.get("if_sr", False)),
                pause_second=float(config.get("pause_second", 0.3)),
                progress_callback=wrapped_callback
            )
            
            # 检查是否在处理过程中被中断
            if self.should_stop:
                return False, "推理已停止"
            
            # 生成文件名
            timestamp = uuid.uuid4().hex[:8]
            role_name = self.sanitize_filename(config.get("role_name", "未知角色"))
            emotion_name = self.sanitize_filename(config.get("emotion_name", "未知情绪"))
            text_prefix = self.sanitize_filename(config.get("text", "")[:20])
            filename = f"{role_name}_{emotion_name}_{text_prefix}_{timestamp}.wav"
            output_path = str(self.output_dir / filename)
            
            # 保存音频文件
            sf.write(output_path, audio_data, sample_rate)
            
            return True, output_path
            
        except Exception as e:
            return False, f"生成语音失败: {str(e)}"


class MultiRoleInferenceEngine(BaseInferenceEngine):
    """多角色推理引擎"""
    
    def __init__(self, output_dir: str = "output"):
        """初始化多角色推理引擎"""
        super().__init__(output_dir)
        self.single_engine = SingleRoleInferenceEngine(output_dir)
        self.text_parser = RoleTextParser()
        self.role_config_manager = RoleConfigManager()
        self.should_stop = False
        self.role_segments = []
        self.total_segments = 0
        self.current_segment = 0
        self.current_role_index = 0
    
    def set_stop_flag(self, should_stop: bool):
        """设置停止标志"""
        self.should_stop = should_stop
        self.single_engine.set_stop_flag(should_stop)
    
    def progress_wrapper(self, callback):
        """
        进度回调包装函数
        
        参数:
            callback: 原始回调函数
            
        返回:
            包装后的回调函数
        """
        def wrapped_callback(current_segment, total_segments=None):
            # 检查是否应该停止
            if self.should_stop:
                return True
            
            # 确保类型是数字
            if current_segment is not None:
                current_segment = int(current_segment) if isinstance(current_segment, (int, float)) else 0
            else:
                current_segment = 0
                
            if total_segments is not None:
                total_segments = int(total_segments) if isinstance(total_segments, (int, float)) else 0
            
            # 更新当前段信息
            if current_segment > 0:
                self.current_segment = current_segment
            
            # 更新总段数信息
            if total_segments is not None:
                self.total_segments = total_segments
            
            # 计算总进度
            role_index = self.current_role_index
            role_count = len(self.role_segments)
            
            if role_count <= 0:
                # 防止除零错误
                progress_msg = "准备中..."
            elif current_segment <= 0:
                # 初始阶段
                progress_msg = f"处理角色 {role_index + 1}/{role_count}: {self.role_segments[role_index].get('role', '未知角色')}"
            else:
                # 计算当前角色的进度百分比
                progress_msg = (
                    f"处理角色 {role_index + 1}/{role_count}: "
                    f"{self.role_segments[role_index].get('role', '未知角色')} "
                    f"({current_segment}/{total_segments})"
                )
            
            # 调用原始回调
            if callback:
                callback(progress_msg)
            
            return False
        
        return wrapped_callback
    
    def generate(self, config: Dict, progress_callback: Callable = None) -> Tuple[bool, str]:
        """
        生成多角色语音
        
        参数:
            config: 推理配置
            progress_callback: 进度回调函数
            
        返回:
            (成功标志, 结果路径或错误信息)
        """
        # 重置状态
        self.should_stop = False
        self.single_engine.set_stop_flag(False)
        self.current_role_index = 0
        self.current_segment = 0
        self.total_segments = 0
        
        # 解析多角色文本
        text = config.get("text", "")
        self.role_segments = self.text_parser.parse_multi_role_text(text)
        
        # 检查是否有有效的角色段
        if not self.role_segments:
            return False, "没有找到有效的文本内容"
        
        # 如果只有一个角色且没有指定角色名，则使用单角色引擎
        if len(self.role_segments) == 1 and self.role_segments[0]["role"] is None:
            return self.single_engine.generate(config, progress_callback)
        
        # 多角色处理
        if progress_callback:
            progress_callback(f"解析到{len(self.role_segments)}个角色，开始处理")
        
        # 存储每个角色的合成结果
        role_outputs = []
        
        # 逐个处理每个角色的文本
        for i, role_data in enumerate(self.role_segments):
            # 更新当前角色索引
            self.current_role_index = i
            
            if self.should_stop:
                return False, "推理已停止"
            
            role_name = role_data.get("role")
            emotion_name = role_data.get("emotion")
            role_text = role_data.get("text")
            
            # 如果是未指定角色名，使用配置中的角色名
            if role_name is None:
                role_name = config.get("role_name", "未知角色")
            
            # 如果是未指定情绪，使用配置中的情绪名
            if emotion_name is None:
                emotion_name = config.get("emotion_name", "未知情绪")
            
            # 获取角色特定的配置（从角色配置文件中读取）
            role_specific_config = self.role_config_manager.get_role_config(role_name, emotion_name)
            
            # 创建该角色的配置（先使用角色特定配置，然后使用原始配置作为备份）
            role_config = role_specific_config.copy() if role_specific_config else {}
            
            # 如果角色特定配置中缺少某些必要参数，从原始配置中补充
            for key, value in config.items():
                if key not in role_config:
                    role_config[key] = value
            
            # 确保必要的角色信息存在
            role_config["role_name"] = role_name
            role_config["emotion_name"] = emotion_name
            role_config["text"] = role_text
            
            # 添加进度说明，包括当前使用的模型信息
            if progress_callback:
                gpt_path = role_config.get("gpt_path", "未知")
                sovits_path = role_config.get("sovits_path", "未知")
                ref_audio = role_config.get("ref_audio", "无")
                progress_callback(
                    f"开始处理角色 {role_name} (情绪: {emotion_name})\n"
                    f"使用模型: {gpt_path} / {sovits_path}\n"
                    f"参考音频: {ref_audio}"
                )
            
            # 确保数值类型正确
            for key in ["top_k", "sample_steps"]:
                if key in role_config:
                    role_config[key] = int(role_config[key])
            
            for key in ["top_p", "temperature", "speed", "pause_second"]:
                if key in role_config:
                    role_config[key] = float(role_config[key])
            
            for key in ["ref_free", "if_freeze", "if_sr"]:
                if key in role_config:
                    role_config[key] = bool(role_config[key])
            
            # 包装进度回调
            wrapped_callback = None
            if progress_callback:
                wrapped_callback = self.progress_wrapper(progress_callback)
            
            # 使用单角色引擎合成该角色的语音
            success, result = self.single_engine.generate(role_config, wrapped_callback)
            
            if not success:
                return False, f"角色 '{role_name}' 推理失败: {result}"
            
            # 保存成功的结果路径
            role_outputs.append(result)
        
        # 合并所有角色的音频
        if len(role_outputs) == 1:
            # 只有一个角色，直接返回结果
            return True, role_outputs[0]
        
        # 否则，需要合并多个音频文件
        try:
            if progress_callback:
                progress_callback("正在合并多角色音频...")
            
            # 读取所有音频文件
            audio_segments = []
            sample_rate = None
            
            for audio_path in role_outputs:
                data, sr = sf.read(audio_path)
                if sample_rate is None:
                    sample_rate = sr
                elif sr != sample_rate:
                    # 如果采样率不一致，则需要重采样（简化处理，实际应用可能需要更复杂的处理）
                    return False, "角色音频采样率不一致，无法合并"
                
                audio_segments.append(data)
            
            # 合并音频（简单的拼接，可能需要添加间隔静音）
            merged_audio = np.concatenate(audio_segments)
            
            # 生成合并后的文件名
            timestamp = uuid.uuid4().hex[:8]
            filename = f"多角色_{timestamp}.wav"
            output_path = str(self.output_dir / filename)
            
            # 保存合并后的音频
            sf.write(output_path, merged_audio, sample_rate)
            
            return True, output_path
            
        except Exception as e:
            return False, f"合并多角色音频失败: {str(e)}" 