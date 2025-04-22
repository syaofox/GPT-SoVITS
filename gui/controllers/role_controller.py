"""
角色控制器

管理角色和情感的创建、读取和更新
"""

from typing import Dict, List
from PySide6.QtCore import Slot, Signal

from gui.controllers.base_controller import BaseController
from gui.models.role_model import RoleModel


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