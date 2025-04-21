"""
GUI控制器层

处理用户交互和业务逻辑调用
"""

from typing import Dict, List, Tuple, Optional, Any, Union
from PySide6.QtCore import QObject, Signal, Slot, Property

from gui.models import RoleModel, InferenceModel


class BaseController(QObject):
    """基础控制器类"""
    
    error_occurred = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)


class RoleController(BaseController):
    """角色控制器类"""
    
    roles_changed = Signal()
    role_saved = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.role_model = RoleModel()
    
    @Slot(result=list)
    def get_role_names(self) -> List[str]:
        """获取所有角色名称"""
        return list(self.role_model.roles.keys())
    
    @Slot(str, result=list)
    def get_emotion_names(self, role_name: str) -> List[str]:
        """获取角色的所有情感名称"""
        emotions = self.role_model.get_role_emotions(role_name)
        return list(emotions.keys())
    
    @Slot(str, str, result="QVariantMap")
    def get_emotion_config(self, role_name: str, emotion_name: str) -> Dict:
        """获取情感配置"""
        return self.role_model.get_emotion_config(role_name, emotion_name)
    
    @Slot()
    def refresh_roles(self):
        """刷新角色列表"""
        self.role_model.load_roles()
        self.roles_changed.emit()
    
    @Slot(str, "QVariantMap", result=bool)
    def save_role_config(self, role_name: str, config: Dict) -> bool:
        """保存角色配置"""
        if not role_name:
            self.error_occurred.emit("角色名不能为空")
            return False
            
        result = self.role_model.save_role_config(role_name, config)
        if result:
            self.role_saved.emit(role_name)
            self.roles_changed.emit()
        else:
            self.error_occurred.emit(f"保存角色配置失败: {role_name}")
        
        return result


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