"""
推理控制器

处理语音合成推理和历史记录管理
"""

import atexit
from typing import Dict, List, Callable
from PySide6.QtCore import Slot, Signal

from gui.controllers.base_controller import BaseController
from gui.models.inference_model import InferenceModel


class InferenceController(BaseController):
    """推理控制器类"""
    
    inference_started = Signal()
    inference_completed = Signal(str)
    inference_failed = Signal(str)
    history_updated = Signal()
    progress_updated = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.inference_model = InferenceModel()
        
        # 注册程序退出时保存历史记录
        atexit.register(self.save_history)
    
    def __del__(self):
        """析构函数，确保保存历史记录"""
        self.save_history()
        # 取消注册退出函数，避免重复调用
        try:
            atexit.unregister(self.save_history)
        except:
            pass
    
    @Slot("QVariantMap")
    def generate_speech_async(self, config: Dict):
        """异步生成语音"""
        if not config.get("text"):
            self.error_occurred.emit("文本不能为空")
            return
            
        if not config.get("ref_audio"):
            self.error_occurred.emit("参考音频不能为空")
            return
            
        self.inference_started.emit()
        
        # 设置回调函数
        def on_finished(success, result):
            if success:
                self.inference_completed.emit(result)
                self.history_updated.emit()
            else:
                self.inference_failed.emit(result)
        
        # 启动异步推理
        success = self.inference_model.generate_speech_async(
            config, 
            on_finished,
            self.progress_updated.emit
        )
        
        if not success:
            self.inference_failed.emit("启动推理任务失败")
    
    @Slot("QVariantMap", result=str)
    def generate_speech(self, config: Dict) -> str:
        """同步生成语音（保留以兼容现有代码）"""
        if not config.get("text"):
            self.error_occurred.emit("文本不能为空")
            return ""
            
        if not config.get("ref_audio"):
            self.error_occurred.emit("参考音频不能为空")
            return ""
            
        self.inference_started.emit()
        
        success, result = self.inference_model.generate_speech(config)
        
        if success:
            self.inference_completed.emit(result)
            self.history_updated.emit()
            return result
        else:
            self.inference_failed.emit(result)
            return ""
    
    @Slot(result=list)
    def get_history(self) -> List[Dict]:
        """获取历史记录"""
        return self.inference_model.get_history()
    
    @Slot()
    def clear_history(self):
        """清空历史记录"""
        self.inference_model.clear_history()
        self.history_updated.emit()
    
    @Slot()
    def save_history(self):
        """保存历史记录"""
        self.inference_model.save_history() 