"""
推理模型

管理语音合成推理和历史记录
"""

import os
import json
import uuid
import gc
import torch
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from datetime import datetime

# 导入多线程支持
from PySide6.QtCore import QThread

# 导入推理模块
from gpt_sovits_inference import GPTSoVITSInference
from gui.models.inference_worker import InferenceWorker


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
        
        # 历史记录文件
        self.history_file = Path("gui") / "history.json"
        
        self.inference_engine: Optional[GPTSoVITSInference] = None
        self.history: List[Dict] = []
        self.current_gpt_path: str = ""
        self.current_sovits_path: str = ""
        
        # 工作线程相关
        self.worker = None
        self.thread = None
        
        # 加载历史记录
        self.load_history()
    
    def reset_engine(self):
        """重置推理引擎"""
        if self.inference_engine is not None:
            del self.inference_engine
            self.inference_engine = None
            self.current_gpt_path = ""
            self.current_sovits_path = ""
            # 强制垃圾回收以释放GPU内存
            gc.collect()
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
    
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
        
        # 检查模型路径是否变化
        if (self.inference_engine is not None and 
            (gpt_path != self.current_gpt_path or sovits_path != self.current_sovits_path)):
            # 如果模型路径变化，重置引擎
            self.reset_engine()
        
        # 创建新的工作线程
        self.thread = QThread()
        self.worker = InferenceWorker()
        self.worker.moveToThread(self.thread)
        
        # 设置推理配置和引擎（如果需要重新初始化，则引擎为None）
        self.worker.set_config(config, self.inference_engine, str(self.output_dir))
        
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
            # 如果生成成功，更新当前引擎引用（如果模型是在工作线程中初始化的）
            if self.worker.engine and not self.inference_engine:
                self.inference_engine = self.worker.engine
                self.current_gpt_path = self.worker.gpt_path
                self.current_sovits_path = self.worker.sovits_path
            
            # 添加到历史记录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            history_entry = {
                "timestamp": timestamp,
                "path": result,
                "config": {k: v for k, v in self.worker.config.items() if k != "aux_refs"},
                "text": self.worker.config.get("text", "")
            }
            self.history.append(history_entry)
            
            # 保存历史记录
            self.save_history()
        
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
        同步生成语音（保留以兼容现有代码）
        
        参数:
            config: 推理配置
            
        返回:
            (是否成功, 输出文件路径)
        """
        # 为保持一致性，同步方法也改为使用异步实现
        future_result = {"success": False, "result": "未完成"}
        
        def on_finished(success, result):
            future_result["success"] = success
            future_result["result"] = result
            
        # 启动异步任务
        started = self.generate_speech_async(config, on_finished)
        if not started:
            return False, "启动任务失败"
            
        # 等待异步任务完成（这会阻塞UI，但由于这是同步方法，这是预期行为）
        while self.thread and self.thread.isRunning():
            QThread.msleep(100)  # 等待100毫秒
            QThread.yieldCurrentThread()  # 让出当前线程以处理事件
            
        # 返回结果
        return future_result["success"], future_result["result"]
    
    def get_history(self) -> List[Dict]:
        """获取历史记录"""
        return self.history
    
    def clear_history(self):
        """清空历史记录"""
        self.history = []
        self.save_history()
    
    def save_history(self):
        """保存历史记录"""
        try:
            # 将历史记录转换为JSON
            history_data = []
            for entry in self.history:
                # 创建一个新的条目副本用于保存
                entry_copy = entry.copy()
                
                # 解决JSON序列化问题
                entry_copy.pop("config", None)  # 暂时不保存配置
                
                history_data.append(entry_copy)
                
            # 保存到文件
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(history_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存历史记录失败: {str(e)}")
    
    def load_history(self):
        """加载历史记录"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, "r", encoding="utf-8") as f:
                    self.history = json.load(f)
                    
                # 过滤出有效的记录
                valid_history = []
                for entry in self.history:
                    path = entry.get("path", "")
                    if path and os.path.exists(path):
                        valid_history.append(entry)
                        
                self.history = valid_history
        except Exception as e:
            print(f"加载历史记录失败: {str(e)}")
            self.history = [] 