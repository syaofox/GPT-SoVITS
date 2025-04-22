"""
GUI控制器包

控制器负责处理用户交互和业务逻辑调用
"""

from gui.controllers.base_controller import BaseController
from gui.controllers.role_controller import RoleController
from gui.controllers.inference_controller import InferenceController

__all__ = ['BaseController', 'RoleController', 'InferenceController'] 