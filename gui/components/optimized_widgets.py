"""
性能优化控件模块

提供对PySide6控件的性能优化版本
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTextEdit, QMessageBox
from PySide6.QtGui import QTextCursor, QKeyEvent


class OptimizedTextEdit(QTextEdit):
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptRichText(False)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setLineWrapMode(QTextEdit.WidgetWidth)