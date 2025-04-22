"""
推理工作线程模型

在独立线程中执行语音合成推理任务
"""

import uuid
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
            self.progress.emit("正在生成语音...")
            
            # 生成语音
            sample_rate, audio_data = self.engine.generate_speech(
                ref_wav_path=self.config.get("ref_audio", ""),
                prompt_text=self.config.get("prompt_text", ""),
                prompt_language=self.config.get("prompt_lang", "中文"),
                text=self.config.get("text", ""),
                text_language=self.config.get("text_lang", "中文"),
                how_to_cut=self.config.get("how_to_cut", "凑四句一切"),
                top_k=self.config.get("top_k", 20),
                top_p=self.config.get("top_p", 0.6),
                temperature=self.config.get("temperature", 0.6),
                ref_free=self.config.get("ref_free", False),
                speed=self.config.get("speed", 1.0),
                inp_refs=self.config.get("aux_refs", []),
                sample_steps=self.config.get("sample_steps", 8),
                if_sr=self.config.get("if_sr", False),
                pause_second=self.config.get("pause_second", 0.3),
            )
            
            # 生成唯一文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"{timestamp}_{unique_id}.wav"
            output_path = Path(self.output_dir) / filename
            
            # 保存音频
            self.progress.emit("正在保存音频...")
            sf.write(output_path, audio_data, sample_rate)
            
            self.finished.emit(True, str(output_path))
        except Exception as e:
            error_msg = f"生成语音失败: {str(e)}"
            print(error_msg)
            self.finished.emit(False, error_msg) 