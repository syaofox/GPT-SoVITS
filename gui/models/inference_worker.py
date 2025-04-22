"""
推理工作线程模型

在独立线程中执行语音合成推理任务
"""

import uuid
import re
from datetime import datetime
from typing import Dict, Any
from pathlib import Path
import soundfile as sf

from PySide6.QtCore import QObject, Signal

# 导入推理模块
from gpt_sovits_inference import GPTSoVITSInference


class InferenceWorker(QObject):
    """推理工作线程"""
    
    finished = Signal(bool, str)  # 成功标志，结果路径或错误信息
    progress = Signal(str)  # 进度信息
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = {}
        self.engine = None
        self.output_dir = "output"
        self.gpt_path = ""
        self.sovits_path = ""
        self.device = None
        self.half = True
        self.text_segments = []
        self.total_segments = 0
        self.current_segment = 0
    
    def set_config(self, config: Dict, engine, output_dir: str):
        """设置推理配置和引擎（如果已有）"""
        self.config = config
        self.engine = engine
        self.output_dir = output_dir
        
        # 保存模型路径以便需要时初始化
        self.gpt_path = config.get("gpt_path", "")
        self.sovits_path = config.get("sovits_path", "")
    
    def initialize_engine(self):
        """初始化推理引擎"""
        self.progress.emit("正在初始化模型...")
        
        try:
            self.engine = GPTSoVITSInference(
                gpt_path=self.gpt_path,
                sovits_path=self.sovits_path,
                device=self.device,
                half=self.half
            )
            return True
        except Exception as e:
            error_msg = f"初始化推理引擎失败: {str(e)}"
            print(error_msg)
            return False
    
    def sanitize_filename(self, text):
        """处理文件名以确保合法性"""
        # 替换非法字符为下划线
        # 对Windows和UNIX都有效的文件名处理
        illegal_chars = r'[\\/*?:"<>|]'
        sanitized = re.sub(illegal_chars, '_', text)
        # 移除空格和换行符
        sanitized = sanitized.replace(' ', '_').replace('\n', '_').replace('\r', '_')
        # 替换连续的下划线为单个下划线
        sanitized = re.sub(r'_+', '_', sanitized)
        # 限制长度
        return sanitized[:30] if len(sanitized) > 30 else sanitized
    
    def progress_callback(self, current_segment, total_segments=None):
        """进度回调函数，直接使用从推理引擎获取的进度信息"""
        # 更新总段数（如果提供）
        if total_segments is not None:
            self.total_segments = total_segments
            # 如果是初始通知（current_segment=0），显示分段信息
            if current_segment == 0:
                self.progress.emit(f"文本已分割为 {self.total_segments} 个片段")
                return
        
        # 确保数值有效
        if current_segment <= 0:
            current_segment = 1
            
        # 确保total_segments有效
        if self.total_segments <= 0:
            self.total_segments = max(1, current_segment)
        
        # 确保current不会超过total
        current_segment = min(current_segment, self.total_segments)
        
        # 更新当前段索引
        self.current_segment = current_segment
        
        # 发送进度信息
        self.progress.emit(f"正在合成: {current_segment}/{self.total_segments}")
    
    def run(self):
        """执行推理任务"""
        # 检查是否需要初始化引擎
        if not self.engine and (self.gpt_path and self.sovits_path):
            success = self.initialize_engine()
            if not success:
                self.finished.emit(False, "模型初始化失败")
                return
        
        if not self.engine:
            self.finished.emit(False, "推理引擎未初始化")
            return
            
        try:
            # 获取配置参数
            text = self.config.get("text", "")
            how_to_cut = self.config.get("how_to_cut", "凑四句一切")
            
            # 重置进度信息
            self.total_segments = 0
            self.current_segment = 0
            
            # 显示开始消息
            self.progress.emit("准备生成语音...")
            
            # 生成语音（GPTSoVITSInference会计算文本分段并通过回调提供进度）
            sample_rate, audio_data = self.engine.generate_speech(
                ref_wav_path=self.config.get("ref_audio", ""),
                prompt_text=self.config.get("prompt_text", ""),
                prompt_language=self.config.get("prompt_lang", "中文"),
                text=text,
                text_language=self.config.get("text_lang", "中文"),
                how_to_cut=how_to_cut,
                top_k=self.config.get("top_k", 20),
                top_p=self.config.get("top_p", 0.6),
                temperature=self.config.get("temperature", 0.6),
                ref_free=self.config.get("ref_free", False),
                speed=self.config.get("speed", 1.0),
                inp_refs=self.config.get("aux_refs", []),
                sample_steps=self.config.get("sample_steps", 8),
                if_sr=self.config.get("if_sr", False),
                pause_second=self.config.get("pause_second", 0.3),
                progress_callback=self.progress_callback
            )
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 获取角色名、情绪名和文本
            role_name = self.sanitize_filename(self.config.get("role_name", "未知角色"))
            emotion_name = self.sanitize_filename(self.config.get("emotion_name", "未知情绪"))
            
            # 提取文本的前10个字符并确保合法性
            text_prefix = self.sanitize_filename(text[:10])
            if not text_prefix:
                text_prefix = "无文本"
            
            # 组合文件名: 时间戳_角色_情绪_文本前10字
            filename = f"{timestamp}_{role_name}_{emotion_name}_{text_prefix}.wav"
            
            output_path = Path(self.output_dir) / filename
            
            # 保存音频
            self.progress.emit("正在保存音频...")
            sf.write(output_path, audio_data, sample_rate)
            
            self.finished.emit(True, str(output_path))
        except Exception as e:
            error_msg = f"生成语音失败: {str(e)}"
            print(error_msg)
            self.finished.emit(False, error_msg) 