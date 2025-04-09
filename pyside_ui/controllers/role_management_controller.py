#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
角色管理控制器
"""

import os
import platform
import subprocess
from pathlib import Path

from ui.utils import LANGUAGE_OPTIONS, clean_file_path
from ui.models import get_model_lists
from ui.roles import (
    list_roles, save_role_config, delete_role_config, 
    load_and_process_role_config, get_emotions
)
from ui.api_client import test_role_synthesis


class RoleManagementController:
    def __init__(self):
        """初始化控制器"""
        pass
    
    def get_roles(self):
        """获取角色列表"""
        return list_roles()
    
    def get_model_lists(self):
        """获取模型列表"""
        return get_model_lists()
    
    def get_language_options(self):
        """获取语言选项"""
        return LANGUAGE_OPTIONS
    
    def get_emotions(self, role):
        """获取角色的情绪列表"""
        return get_emotions(role)
    
    def extract_text_from_filename(self, file_path):
        """从文件名自动提取参考文本"""
        if not file_path:
            return ""
        # 清理文件路径，去除可能的引号
        file_path = clean_file_path(file_path)
        file_name = os.path.basename(file_path)
        name_without_ext = os.path.splitext(file_name)[0]
        return name_without_ext
    
    def process_aux_refs(self, aux_refs_text):
        """处理辅助参考音频列表，提取文件路径"""
        if not aux_refs_text:
            return []
        
        processed_refs = []
        
        # 按行分割，每行作为一个文件路径
        for line in aux_refs_text.splitlines():
            line = line.strip()
            if line:  # 跳过空行
                processed_refs.append(clean_file_path(line))
        
        return processed_refs
    
    def save_role_config(self, role_name, gpt_model, sovits_model, ref_audio, prompt_text,
                         prompt_lang, text_lang, speed, ref_free, if_sr, top_k, top_p, 
                         temperature, sample_steps, pause_second, description, aux_refs):
        """保存角色配置"""
        # 处理辅助参考音频
        processed_aux_refs = self.process_aux_refs(aux_refs)
        
        # 调用原始的保存函数
        return save_role_config(
            role_name, gpt_model, sovits_model, clean_file_path(ref_audio), prompt_text, prompt_lang, text_lang,
            speed, ref_free, if_sr, top_k, top_p, temperature, sample_steps, pause_second, 
            description, processed_aux_refs
        )
    
    def load_role_config(self, role, emotion=None):
        """加载角色配置"""
        # 调用原始的加载函数
        result = load_and_process_role_config(role, emotion, self.process_aux_refs)
        
        # 将结果转换为字典并确保所有值都是适当的基本类型
        config_dict = {}
        
        # 处理GPT模型路径
        if isinstance(result[0], dict) and "value" in result[0]:
            config_dict["gpt_model"] = str(result[0]["value"]) if result[0]["value"] else ""
        else:
            config_dict["gpt_model"] = str(result[0]) if result[0] else ""
        
        # 处理SoVITS模型路径
        if isinstance(result[1], dict) and "value" in result[1]:
            config_dict["sovits_model"] = str(result[1]["value"]) if result[1]["value"] else ""
        else:
            config_dict["sovits_model"] = str(result[1]) if result[1] else ""
        
        # 处理参考音频路径
        if isinstance(result[2], dict) and "value" in result[2]:
            config_dict["ref_audio"] = str(result[2]["value"]) if result[2]["value"] else ""
        else:
            config_dict["ref_audio"] = str(result[2]) if result[2] else ""
        
        # 处理提示文本
        if isinstance(result[3], dict) and "value" in result[3]:
            config_dict["prompt_text"] = str(result[3]["value"]) if result[3]["value"] else ""
        else:
            config_dict["prompt_text"] = str(result[3]) if result[3] else ""
        
        # 处理辅助参考音频
        aux_refs = result[4]
        if isinstance(aux_refs, dict) and "value" in aux_refs:
            if isinstance(aux_refs["value"], list):
                config_dict["aux_refs"] = "\n".join(str(item) for item in aux_refs["value"])
            else:
                config_dict["aux_refs"] = str(aux_refs["value"]) if aux_refs["value"] else ""
        elif isinstance(aux_refs, list):
            config_dict["aux_refs"] = "\n".join(str(item) for item in aux_refs)
        else:
            config_dict["aux_refs"] = str(aux_refs) if aux_refs else ""
        
        # 处理提示语言
        if isinstance(result[5], dict) and "value" in result[5]:
            config_dict["prompt_lang"] = str(result[5]["value"]) if result[5]["value"] else "中文"
        else:
            config_dict["prompt_lang"] = str(result[5]) if result[5] else "中文"
        
        # 处理文本语言
        if isinstance(result[6], dict) and "value" in result[6]:
            config_dict["text_lang"] = str(result[6]["value"]) if result[6]["value"] else "中文"
        else:
            config_dict["text_lang"] = str(result[6]) if result[6] else "中文"
        
        # 处理速度
        if isinstance(result[7], dict) and "value" in result[7]:
            config_dict["speed"] = float(result[7]["value"]) if isinstance(result[7]["value"], (int, float, str)) else 1.0
        else:
            config_dict["speed"] = float(result[7]) if isinstance(result[7], (int, float, str)) else 1.0
        
        # 处理ref_free
        if isinstance(result[8], dict) and "value" in result[8]:
            config_dict["ref_free"] = bool(result[8]["value"]) if isinstance(result[8]["value"], (bool, int)) else False
        else:
            config_dict["ref_free"] = bool(result[8]) if isinstance(result[8], (bool, int)) else False
        
        # 处理if_sr
        if isinstance(result[9], dict) and "value" in result[9]:
            config_dict["if_sr"] = bool(result[9]["value"]) if isinstance(result[9]["value"], (bool, int)) else False
        else:
            config_dict["if_sr"] = bool(result[9]) if isinstance(result[9], (bool, int)) else False
        
        # 处理top_k
        if isinstance(result[10], dict) and "value" in result[10]:
            config_dict["top_k"] = int(float(result[10]["value"])) if isinstance(result[10]["value"], (int, float, str)) else 15
        else:
            config_dict["top_k"] = int(float(result[10])) if isinstance(result[10], (int, float, str)) else 15
        
        # 处理top_p
        if isinstance(result[11], dict) and "value" in result[11]:
            config_dict["top_p"] = float(result[11]["value"]) if isinstance(result[11]["value"], (int, float, str)) else 1.0
        else:
            config_dict["top_p"] = float(result[11]) if isinstance(result[11], (int, float, str)) else 1.0
        
        # 处理temperature
        if isinstance(result[12], dict) and "value" in result[12]:
            config_dict["temperature"] = float(result[12]["value"]) if isinstance(result[12]["value"], (int, float, str)) else 1.0
        else:
            config_dict["temperature"] = float(result[12]) if isinstance(result[12], (int, float, str)) else 1.0
        
        # 处理sample_steps
        if isinstance(result[13], dict) and "value" in result[13]:
            config_dict["sample_steps"] = int(float(result[13]["value"])) if isinstance(result[13]["value"], (int, float, str)) else 32
        else:
            config_dict["sample_steps"] = int(float(result[13])) if isinstance(result[13], (int, float, str)) else 32
        
        # 处理pause_second
        if isinstance(result[14], dict) and "value" in result[14]:
            config_dict["pause_second"] = float(result[14]["value"]) if isinstance(result[14]["value"], (int, float, str)) else 0.3
        else:
            config_dict["pause_second"] = float(result[14]) if isinstance(result[14], (int, float, str)) else 0.3
        
        # 处理description
        if isinstance(result[15], dict) and "value" in result[15]:
            config_dict["description"] = str(result[15]["value"]) if result[15]["value"] else ""
        else:
            config_dict["description"] = str(result[15]) if result[15] else ""
            
        return config_dict
    
    def delete_role_config(self, role):
        """删除角色配置"""
        return delete_role_config(role)
    
    def test_role_synthesis(self, target_text, gpt_model, sovits_model, ref_audio,
                            prompt_text, prompt_lang, text_lang, speed, ref_free, if_sr,
                            top_k, top_p, temperature, sample_steps, cut_punc, role_name,
                            aux_refs, emotion="", pause_second=0.3):
        """测试角色合成"""
        # 处理辅助参考音频
        processed_aux_refs = self.process_aux_refs(aux_refs)
        
        # 调用原始的合成函数
        return test_role_synthesis(
            target_text, gpt_model, sovits_model, clean_file_path(ref_audio), prompt_text,
            prompt_lang, text_lang, speed, ref_free, if_sr, top_k, top_p,
            temperature, sample_steps, cut_punc, role_name, processed_aux_refs,
            emotion=emotion, pause_second=pause_second
        )
    
    def play_audio(self, audio_path):
        """播放音频"""
        try:
            system = platform.system()
            if system == "Windows":
                os.startfile(audio_path)
            elif system == "Darwin":  # macOS
                subprocess.call(["open", audio_path])
            else:  # Linux
                subprocess.call(["xdg-open", audio_path])
        except Exception as e:
            raise Exception(f"播放音频失败: {str(e)}")