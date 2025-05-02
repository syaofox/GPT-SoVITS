"""
推理控制器

处理语音合成推理和历史记录管理
"""

from typing import Dict, List, Callable
from PySide6.QtCore import Slot, Signal

from gui.controllers.base_controller import BaseController
from gui.models.inference_model import InferenceModel


class InferenceController(BaseController):
    """推理控制器类"""
    
    inference_started = Signal()
    inference_completed = Signal(str)
    inference_failed = Signal(str)
    inference_stopped = Signal()  # 新增推理停止信号
    history_updated = Signal()
    progress_updated = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.inference_model = InferenceModel()
    
    @Slot()
    def stop_inference(self):
        """停止推理"""
        if self.inference_model.stop_inference():
            self.progress_updated.emit("正在停止推理...")
            # 实际的停止完成信号会在工作线程结束时触发
        else:
            # 如果没有正在运行的推理任务，直接发送停止信号
            self.inference_stopped.emit()
    
    @Slot("QVariantMap")
    def generate_speech_async(self, config: Dict):
        """异步生成语音"""
        if not config.get("text"):
            self.error_occurred.emit("文本不能为空")
            return
            
        if not config.get("ref_free") and not config.get("ref_audio"):
            self.error_occurred.emit("参考音频不能为空")
            return
            
        # 确保角色和情感信息存在
        if not config.get("role_name"):
            config["role_name"] = "未知角色"
        if not config.get("emotion_name"):
            config["emotion_name"] = "未知情绪"
            
        self.inference_started.emit()
        
        # 设置回调函数
        def on_finished(success, result):
            if success:
                self.inference_completed.emit(result)
                self.history_updated.emit()
            else:
                if result == "推理已停止":
                    # 如果是主动停止，触发停止信号
                    self.inference_stopped.emit()
                elif "超时" in result:
                    # 如果是超时导致的停止，触发特定的超时信号
                    self.inference_failed.emit(f"合成过程超时: {result}")
                    # 不强制释放GPU内存
                else:
                    # 否则触发失败信号
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
            
        if not config.get("ref_free") and not config.get("ref_audio"):
            self.error_occurred.emit("参考音频不能为空")
            return ""
            
        # 确保角色和情感信息存在
        if not config.get("role_name"):
            config["role_name"] = "未知角色"
        if not config.get("emotion_name"):
            config["emotion_name"] = "未知情绪"
            
        self.inference_started.emit()
        
        success, result = self.inference_model.generate_speech(config)
        
        if success:
            self.inference_completed.emit(result)
            self.history_updated.emit()
            return result
        else:
            if result == "推理已停止":
                # 如果是主动停止，触发停止信号
                self.inference_stopped.emit()
            else:
                # 否则触发失败信号
                self.inference_failed.emit(result)
            return ""
    
    @Slot(result=list)
    def get_history(self) -> List[Dict]:
        """获取历史记录"""
        return self.inference_model.get_history()
    
    def _force_cleanup(self):
        """强制清理资源，特别是在推理卡死情况下"""
        # 在实际情况下此函数不会被调用
        # 但保留函数定义以避免代码错误
        pass 