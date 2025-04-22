"""
基础控制器

所有控制器的基类
"""

from PySide6.QtCore import QObject, Signal


class BaseController(QObject):
    """基础控制器类"""
    
    error_occurred = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent) 