"""
推理模型

管理语音合成推理和历史记录
"""

import os
import uuid
import gc
import torch
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from datetime import datetime

# 导入多线程支持
from PySide6.QtCore import QThread, Slot, Signal

# 导入推理模块
from gpt_sovits_inference import GPTSoVITSInference
from gui.models.inference_worker import InferenceWorker

# 导入推理引擎
from gui.models.inference_engines import (
    BaseInferenceEngine, 
    SingleRoleInferenceEngine, 
    MultiRoleInferenceEngine, 
    RoleTextParser
)


class InferenceModel:
    """推理模型类，用于处理语音合成请求"""
    
    def __init__(self, output_dir: str = "output"):
        """
        初始化推理模型
        
        参数:
            output_dir: 输出音频文件目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 推理引擎实例
        self.single_engine = SingleRoleInferenceEngine(str(self.output_dir))
        self.multi_engine = MultiRoleInferenceEngine(str(self.output_dir))
        
        # 历史记录列表 - 仅在内存中保存
        self._history = []
        
        # 工作线程相关
        self.worker = None
        self.thread = None
        
        # 文本解析器
        self.text_parser = RoleTextParser()
    
    def _load_history(self) -> List[Dict]:
        """加载历史记录 - 已废弃，仅返回空列表"""
        return []
    
    def save_history(self):
        """保存历史记录 - 已废弃，不执行任何操作"""
        pass
    
    def stop_inference(self) -> bool:
        """
        停止当前推理任务
        
        返回:
            bool: 是否成功发送停止信号（True表示有正在运行的任务被请求停止）
        """
        if self.worker and self.thread and self.thread.isRunning():
            # 告诉工作线程停止处理
            self.worker.stop()
            return True
        else:
            return False
    
    def generate_speech_async(self, config: Dict, on_finished: Callable, on_progress=None) -> bool:
        """
        异步生成语音
        
        参数:
            config: 推理配置
            on_finished: 完成回调函数
            on_progress: 进度回调函数
            
        返回:
            是否成功启动推理
        """
        gpt_path = config.get("gpt_path")
        sovits_path = config.get("sovits_path")
        
        if not gpt_path or not sovits_path:
            return False
        
        # 停止现有的工作线程（如果有）
        if self.thread and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()
        
        # 创建新的工作线程
        self.thread = QThread()
        self.worker = InferenceWorker()
        self.worker.moveToThread(self.thread)
        
        # 设置推理配置和引擎
        text = config.get("text", "")
        
        # 检查是否是多角色文本
        role_segments = self.text_parser.parse_multi_role_text(text)
        is_multi_role = len(role_segments) > 1 or (len(role_segments) == 1 and role_segments[0]["role"] is not None)
        
        # 设置使用的引擎类型
        engine_type = "multi" if is_multi_role else "single"
        
        # 设置工作线程配置
        self.worker.set_config(config, engine_type, str(self.output_dir))
        
        # 连接信号
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._on_worker_finished)
        if on_progress:
            self.worker.progress.connect(on_progress)
        self.worker.finished.connect(lambda: self.thread.quit())
        self.thread.finished.connect(lambda: self._cleanup_thread())
        
        # 保存回调函数
        self._on_finished_callback = on_finished
        
        # 启动线程
        self.thread.start()
        
        return True
    
    def _on_worker_finished(self, success: bool, result: str):
        """工作线程完成回调"""
        if success:
            # 添加到内存中的历史记录
            self._add_to_history(result)
        
        # 调用外部回调
        if self._on_finished_callback:
            self._on_finished_callback(success, result)
    
    def _cleanup_thread(self):
        """清理线程资源"""
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
        if self.thread:
            self.thread.deleteLater()
            self.thread = None
    
    def generate_speech(self, config: Dict) -> Tuple[bool, str]:
        """
        同步生成语音（为兼容现有代码保留）
        
        参数:
            config: 推理配置
            
        返回:
            (成功标志，结果路径或错误信息)
        """
        text = config.get("text", "")
        
        # 解析文本，检测是否为多角色文本
        role_segments = self.text_parser.parse_multi_role_text(text)
        is_multi_role = len(role_segments) > 1 or (len(role_segments) == 1 and role_segments[0]["role"] is not None)
        
        # 选择合适的引擎
        engine = self.multi_engine if is_multi_role else self.single_engine
        
        # 生成语音
        success, result = engine.generate(config)
        
        # 如果成功，添加到历史记录
        if success:
            self._add_to_history(result)
            
        return success, result
    
    def _add_to_history(self, audio_path: str):
        """
        添加记录到内存中的历史记录
        
        参数:
            audio_path: 生成的音频文件路径
        """
        if not audio_path:
            return
            
        # 提取文件名
        filename = os.path.basename(audio_path)
        
        # 创建历史记录
        record = {
            "id": str(uuid.uuid4()),
            "path": audio_path,
            "filename": filename,
            "timestamp": datetime.now().isoformat(),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 添加到内存中的历史记录
        self._history.append(record)
        
        # 限制历史记录数量（保留最新的100条）
        if len(self._history) > 100:
            self._history = self._history[-100:]
    
    @Slot(result=list)
    def get_history(self) -> List[Dict]:
        """获取历史记录"""
        return self._history 