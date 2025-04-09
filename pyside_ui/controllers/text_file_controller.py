#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文本文件控制器
"""

import os
import time
import platform
import subprocess
from pathlib import Path

from ui.utils import LANGUAGE_OPTIONS, g_default_role
from ui.roles import list_roles, get_emotions
from ui.text_processing import preprocess_text
from ui.api_client import process_text, merge_audio_segments
from ui.models import init_models


class TextFileController:
    def __init__(self):
        """初始化控制器"""
        # 获取角色列表
        self.roles = list_roles()
        
        # 确保默认角色在角色列表中，否则使用第一个角色
        self.default_role = g_default_role
        if self.default_role not in self.roles and self.roles:
            self.default_role = self.roles[0]
    
    def get_roles(self):
        """获取角色列表"""
        return self.roles
    
    def get_language_options(self):
        """获取语言选项"""
        return LANGUAGE_OPTIONS
    
    def get_emotions(self, role):
        """获取角色的情绪列表"""
        return get_emotions(role)
    
    def refresh_roles(self):
        """刷新角色列表"""
        self.roles = list_roles()
        return self.roles
    
    def preprocess_text(self, content, default_role, default_emotion):
        """预处理文本"""
        return preprocess_text(content, default_role, default_emotion)
    
    def get_tone_wrap_markers(self):
        """从shortcuts.conf获取前后包围标记
        
        Returns:
            tuple: (前缀, 后缀)
        """
        try:
            with open("configs/shortcuts.conf", "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split('|')
                    if len(parts) >= 3:
                        return parts[1].strip(), parts[2].strip()
        except Exception:
            pass
        return "<tone as=\"\">", "</tone>"
        
    def get_formatted_filename(self, role_name, text_content):
        """生成格式化的文件名：角色名_时间戳_文本内容前20个字符
        
        Args:
            role_name: 角色名称
            text_content: 文本内容
        
        Returns:
            格式化的文件名
        """
        # 获取第一个非空行的前20个字符
        first_text = ""
        for line in text_content.strip().split("\n"):
            line = line.strip()
            if line:
                # 如果有角色标记，提取实际文本内容
                if line.startswith("(") and ")" in line:
                    _, text = line.split(")", 1)
                    first_text = text.strip()
                else:
                    first_text = line
                break
        
        # 提取前20个字符，去除可能导致文件名问题的字符
        if first_text:
            short_text = first_text[:20].strip()
            # 替换不适合作为文件名的字符
            for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
                short_text = short_text.replace(char, '_')
        else:
            short_text = "无文本"
            
        # 生成文件名
        return f"{role_name}_{int(time.time())}_{short_text}"
    
    def process_text_content(
        self,
        text_content,
        force_role="",
        default_role="",
        force_emotion="",
        default_emotion="",
        text_lang="中文",
        cut_punc="",
        process_mode="逐行处理",
        output_dir="output",
    ):
        """处理文本内容生成音频
        
        Args:
            text_content: 文本内容
            force_role: 强制使用的角色
            default_role: 默认角色
            force_emotion: 强制使用的情绪
            default_emotion: 默认情绪
            text_lang: 文本语言
            cut_punc: 切分符号
            process_mode: 处理模式
            output_dir: 输出目录
        
        Returns:
            输出音频文件路径
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成输出文件名
        output_filename = self.get_formatted_filename(default_role, text_content)
        output_path = os.path.join(output_dir, f"{output_filename}.wav")
        
        # 根据处理模式处理文本
        if process_mode == "逐行处理":
            # 逐行处理
            audio_segments = []
            
            for line in text_content.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                
                # 解析行内容
                role, emotion, text = self.parse_line(line)
                
                # 使用强制角色或解析出的角色
                current_role = force_role if force_role else (role if role else default_role)
                
                # 使用强制情绪或解析出的情绪
                current_emotion = force_emotion if force_emotion else (emotion if emotion else default_emotion)
                
                # 处理文本生成音频
                segment_path = process_text(
                    text=text,
                    role=current_role,
                    emotion=current_emotion,
                    text_lang=text_lang,
                    cut_punc=cut_punc,
                    output_dir=output_dir
                )
                
                audio_segments.append(segment_path)
            
            # 合并音频片段
            if audio_segments:
                # 读取所有音频文件
                audio_data_list = []
                for segment_path in audio_segments:
                    with open(segment_path, 'rb') as f:
                        audio_data_list.append(f.read())
                
                # 合并音频
                merged_audio, sr = merge_audio_segments(audio_data_list)
                
                # 保存合并后的音频
                import soundfile as sf
                sf.write(output_path, merged_audio, sr)
                
                return output_path
            else:
                raise ValueError("没有有效的文本内容")
        
        elif process_mode == "全文本处理":
            # 全文本处理
            return process_text(
                text=text_content,
                role=force_role if force_role else default_role,
                emotion=force_emotion if force_emotion else default_emotion,
                text_lang=text_lang,
                cut_punc=cut_punc,
                output_dir=output_dir
            )
        
        elif process_mode == "分段处理":
            # 分段处理
            # 按标点符号分段
            segments = []
            current_segment = ""
            
            for char in text_content:
                current_segment += char
                
                # 如果遇到切分符号且当前段落不为空，则添加到段落列表
                if char in cut_punc and current_segment.strip():
                    segments.append(current_segment.strip())
                    current_segment = ""
            
            # 添加最后一个段落
            if current_segment.strip():
                segments.append(current_segment.strip())
            
            # 如果分段后为空，则按固定长度分段
            if not segments:
                # 按500字符分段
                text = text_content.strip()
                while text:
                    segments.append(text[:500])
                    text = text[500:]
            
            # 处理每个段落
            audio_segments = []
            
            for i, segment in enumerate(segments):
                # 处理文本生成音频
                segment_path = process_text(
                    text=segment,
                    role=force_role if force_role else default_role,
                    emotion=force_emotion if force_emotion else default_emotion,
                    text_lang=text_lang,
                    cut_punc=cut_punc,
                    output_dir=output_dir
                )
                
                audio_segments.append(segment_path)
            
            # 合并音频片段
            if audio_segments:
                # 读取所有音频文件
                audio_data_list = []
                for segment_path in audio_segments:
                    with open(segment_path, 'rb') as f:
                        audio_data_list.append(f.read())
                
                # 合并音频
                merged_audio, sr = merge_audio_segments(audio_data_list)
                
                # 保存合并后的音频
                import soundfile as sf
                sf.write(output_path, merged_audio, sr)
                
                return output_path
            else:
                raise ValueError("没有有效的文本内容")
        
        else:
            raise ValueError(f"不支持的处理模式: {process_mode}")
    
    def parse_line(self, line):
        """解析行内容，提取角色、情绪和文本
        
        Args:
            line: 行内容
        
        Returns:
            (角色, 情绪, 文本)
        """
        from ui.utils import parse_line
        return parse_line(line)
    
    def play_audio(self, audio_path):
        """播放音频
        
        Args:
            audio_path: 音频文件路径
        """
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