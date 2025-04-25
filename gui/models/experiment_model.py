"""
实验模型

管理语音合成实验的配置和数据处理逻辑
"""

import os
from pathlib import Path
from typing import Dict, List, Optional


class ExperimentModel:
    """实验模型类，用于处理实验标签页的数据逻辑"""
    
    def __init__(self):
        """初始化实验模型"""
        # 模型路径字典
        self.gpt_model_paths = {}
        self.sovits_model_paths = {}
    
    def load_gpt_models(self, model_dict: Dict[str, str]) -> None:
        """
        加载GPT模型列表
        
        参数:
            model_dict: 模型名称到路径的映射字典
        """
        self.gpt_model_paths = model_dict
        
    def load_sovits_models(self, model_dict: Dict[str, str]) -> None:
        """
        加载SoVITS模型列表
        
        参数:
            model_dict: 模型名称到路径的映射字典
        """
        self.sovits_model_paths = model_dict
    
    def get_gpt_model_paths(self) -> Dict[str, str]:
        """获取GPT模型路径字典"""
        return self.gpt_model_paths
        
    def get_sovits_model_paths(self) -> Dict[str, str]:
        """获取SoVITS模型路径字典"""
        return self.sovits_model_paths
    
    def get_inference_config(self, view_data: Dict, is_save_role: bool = False) -> Optional[Dict]:
        """
        根据视图数据生成推理配置
        
        参数:
            view_data: 从视图层收集的数据
            is_save_role: 是否为保存角色而获取配置
            
        返回:
            推理配置字典，若配置无效则返回None
        """
        config = {}
        
        # 获取基本配置
        if not is_save_role:
            text = view_data.get("text", "").strip()
            if not text:
                return None
        
            config["text"] = text
            
        config["text_lang"] = view_data.get("text_lang", "")
        config["how_to_cut"] = view_data.get("how_to_cut", "")
        
        # 获取角色名和情绪名
        config["role_name"] = view_data.get("role_name", "").strip()
        config["emotion_name"] = view_data.get("emotion_name", "").strip()
        
        # 获取模型选择
        gpt_model_name = view_data.get("gpt_model_name", "")
        sovits_model_name = view_data.get("sovits_model_name", "")
        
        if not gpt_model_name or not sovits_model_name:
            return None
            
        # 转换为实际路径
        if gpt_model_name in self.gpt_model_paths:
            config["gpt_path"] = self.gpt_model_paths[gpt_model_name]
        else:
            return None
            
        if sovits_model_name in self.sovits_model_paths:
            config["sovits_path"] = self.sovits_model_paths[sovits_model_name]
        else:
            return None
        
        # 获取参考音频
        ref_audio_path = view_data.get("ref_audio", "")
        ref_free = view_data.get("ref_free", False)
        
        if not ref_audio_path and not ref_free:
            return None
            
        config["ref_audio"] = ref_audio_path
        config["prompt_text"] = view_data.get("prompt_text", "").strip()
        config["prompt_lang"] = view_data.get("prompt_lang", "")
        
        # 高级设置
        config["speed"] = view_data.get("speed", 1.0)
        config["top_k"] = view_data.get("top_k", 15)
        config["top_p"] = view_data.get("top_p", 1.0)
        config["temperature"] = view_data.get("temperature", 1.0)
        config["sample_steps"] = view_data.get("sample_steps", 8)
        config["pause_second"] = view_data.get("pause_second", 0.3)
        
        # 选项
        config["ref_free"] = ref_free
        config["if_sr"] = view_data.get("if_sr", False)
        
        # 辅助参考音频
        aux_refs = view_data.get("aux_refs", [])
        if aux_refs:
            config["aux_refs"] = aux_refs
            
        return config
    
    def convert_to_relative_path(self, absolute_path: str, project_root: str) -> str:
        """
        将绝对路径转换为相对项目根目录的路径
        
        参数:
            absolute_path: 绝对路径
            project_root: 项目根目录
            
        返回:
            相对路径，若无法转换则返回原路径
        """
        if not absolute_path:
            return absolute_path
            
        try:
            # 转换为相对路径
            relative_path = os.path.relpath(absolute_path, project_root)
            # 确保使用正斜杠（Windows兼容性）
            relative_path = relative_path.replace('\\', '/')
            return relative_path
        except ValueError:  # 如果路径在不同驱动器上
            return absolute_path  # 保留原始路径 