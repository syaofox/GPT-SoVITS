"""
性能优化控件模块

提供对PySide6控件的性能优化版本
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTextEdit, QMessageBox
from PySide6.QtGui import QTextCursor


class OptimizedTextEdit(QTextEdit):
    """性能优化的文本编辑框，处理大量文本时更流畅"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptRichText(False)
        self.document().setMaximumBlockCount(5000)  # 限制最大块数量
        
    def insertFromMimeData(self, source):
        """优化粘贴操作"""
        if source.hasText():
            # 获取纯文本
            text = source.text()
            
            # 如果文本太长，截断并提示用户
            if len(text) > 50000:  # 设置一个合理的最大长度
                text = text[:50000]
                QMessageBox.warning(self, "文本过长", "粘贴的文本过长，已截断至前50000个字符")
            
            # 使用优化的方式插入文本
            cursor = self.textCursor()
            cursor.beginEditBlock()
            cursor.insertText(text)
            cursor.endEditBlock()
        else:
            super().insertFromMimeData(source) 