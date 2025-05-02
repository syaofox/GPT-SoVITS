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
from gui.models.inference_engines import (
    SingleRoleInferenceEngine, 
    MultiRoleInferenceEngine
)


class InferenceWorker(QObject):
    """推理工作线程"""
    
    finished = Signal(bool, str)  # 成功标志，结果路径或错误信息
    progress = Signal(str)  # 进度信息
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = {}
        self.engine_type = "single"  # 默认为单角色引擎
        self.output_dir = "output"
        self.should_stop = False  # 添加停止标志
        
        # 推理引擎
        self.single_engine = None
        self.multi_engine = None
    
    def set_config(self, config: Dict, engine_type: str, output_dir: str):
        """
        设置推理配置
        
        参数:
            config: 推理配置
            engine_type: 引擎类型，"single"或"multi"
            output_dir: 输出目录
        """
        # 复制配置并确保参数类型正确
        self.config = self._sanitize_config(config)
        self.engine_type = engine_type
        self.output_dir = str(output_dir)
        
        # 重置停止标志
        self.should_stop = False
    
    def _sanitize_config(self, config: Dict) -> Dict:
        """确保配置参数类型正确"""
        result = config.copy()
        
        # 确保数值类型正确
        for key in ["top_k", "sample_steps"]:
            if key in result:
                try:
                    result[key] = int(result[key])
                except (ValueError, TypeError):
                    result[key] = 20 if key == "top_k" else 8
        
        for key in ["top_p", "temperature", "speed", "pause_second"]:
            if key in result:
                try:
                    result[key] = float(result[key])
                except (ValueError, TypeError):
                    default_values = {"top_p": 0.6, "temperature": 0.6, "speed": 1.0, "pause_second": 0.3}
                    result[key] = default_values.get(key, 0.0)
        
        for key in ["ref_free", "if_freeze", "if_sr"]:
            if key in result:
                result[key] = bool(result[key])
        
        return result
    
    def stop(self):
        """设置停止标志，请求推理停止"""
        self.should_stop = True
        self.progress.emit("准备停止推理...")
        
        # 如果引擎已创建，也向其发送停止信号
        if self.single_engine:
            self.single_engine.set_stop_flag(True)
        if self.multi_engine:
            self.multi_engine.set_stop_flag(True)
    
    def setup_engines(self):
        """初始化推理引擎"""
        self.progress.emit("正在初始化推理引擎...")
        
        try:
            # 根据引擎类型初始化相应的引擎
            if self.engine_type == "single":
                self.single_engine = SingleRoleInferenceEngine(self.output_dir)
            else:  # multi
                self.multi_engine = MultiRoleInferenceEngine(self.output_dir)
            
            return True
        except Exception as e:
            error_msg = f"初始化推理引擎失败: {str(e)}"
            self.progress.emit(error_msg)
            return False
    
    def run(self):
        """执行推理任务"""
        # 初始化引擎
        if not self.setup_engines():
            self.finished.emit(False, "推理引擎初始化失败")
            return
            
        try:
            # 设置超时监控
            import threading
            import time
            
            # 标志变量，指示是否已完成或发生错误
            is_completed = False
            error_message = None
            
            # 定义监控函数
            def monitor_progress():
                # 检查卡死情况的时间间隔（秒）
                check_interval = 30
                # 最大无响应时间（秒）
                max_no_response_time = 300  # 多角色合成可能需要更长时间
                
                last_progress_time = time.time()
                
                while not is_completed and not self.should_stop:
                    time.sleep(check_interval)
                    
                    # 如果长时间没有进度更新，可能是卡住了
                    current_time = time.time()
                    if current_time - last_progress_time > max_no_response_time:
                        # 如果超过最大无响应时间，请求停止
                        self.should_stop = True
                        self.progress.emit("推理过程超时，已请求停止")
                        # 不直接结束线程，让推理过程自行检测停止标志并处理
                    
                    # 每次检查将时间重置，避免误判（后续可改进为更精确的检测）
                    last_progress_time = current_time
            
            # 启动监控线程
            monitor_thread = threading.Thread(target=monitor_progress)
            monitor_thread.daemon = True
            monitor_thread.start()
            
            # 选择合适的引擎执行推理
            if self.engine_type == "single":
                engine = self.single_engine
            else:
                engine = self.multi_engine
            
            # 确保进度回调正确处理字符串
            def safe_progress_callback(msg):
                self.progress.emit(str(msg))
            
            # 执行推理
            success, result = engine.generate(
                self.config, 
                safe_progress_callback
            )
            
            # 设置完成标志
            is_completed = True
            
            # 发送结果
            self.finished.emit(success, result)
                
        except Exception as e:
            error_message = f"推理过程发生错误: {str(e)}"
            self.progress.emit(error_message)
            self.finished.emit(False, error_message) 