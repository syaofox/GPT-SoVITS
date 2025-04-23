"""
可拖拽控件模块

提供支持拖放功能的控件
"""

import os
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLineEdit, QListWidget, QListWidgetItem
from PySide6.QtGui import QDragEnterEvent, QDropEvent


class DraggableLineEdit(QLineEdit):
    """支持拖放功能的LineEdit，可用于接收音频文件路径"""
    
    def __init__(self, parent=None, accepted_extensions=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        # 默认接受的文件扩展名
        self.accepted_extensions = accepted_extensions or ['.wav', '.mp3', '.flac', '.ogg']
        
    def dragEnterEvent(self, event: QDragEnterEvent):
        """拖动进入事件"""
        if event.mimeData().hasUrls():
            # 检查是否为可接受的文件
            urls = event.mimeData().urls()
            if urls and len(urls) == 1:
                file_path = urls[0].toLocalFile()
                ext = os.path.splitext(file_path)[1].lower()
                if ext in self.accepted_extensions:
                    event.acceptProposedAction()
                    return
        event.ignore()
    
    def dragMoveEvent(self, event):
        """拖动移动事件"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()
    
    def dropEvent(self, event: QDropEvent):
        """放置事件"""
        urls = event.mimeData().urls()
        if urls and len(urls) == 1:
            file_path = urls[0].toLocalFile()
            # 检查文件扩展名
            ext = os.path.splitext(file_path)[1].lower()
            if ext in self.accepted_extensions:
                self.setText(file_path)
                # 发送文本改变信号，触发相关处理
                self.textChanged.emit(file_path)
                event.acceptProposedAction()
                return
        event.ignore()


class DraggableListWidget(QListWidget):
    """支持拖放功能的ListWidget，可用于接收多个音频文件路径"""
    
    # 定义一个自定义信号，用于通知文件添加
    files_dropped = Signal(list)
    
    def __init__(self, parent=None, accepted_extensions=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        # 默认接受的文件扩展名
        self.accepted_extensions = accepted_extensions or ['.wav', '.mp3', '.flac', '.ogg']
        
    def dragEnterEvent(self, event: QDragEnterEvent):
        """拖动进入事件"""
        if event.mimeData().hasUrls():
            # 检查是否至少有一个可接受的文件
            valid_files = self._filter_valid_files(event.mimeData().urls())
            if valid_files:
                event.acceptProposedAction()
                return
        event.ignore()
    
    def dragMoveEvent(self, event):
        """拖动移动事件"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()
    
    def dropEvent(self, event: QDropEvent):
        """放置事件"""
        if event.mimeData().hasUrls():
            # 获取所有有效文件路径
            valid_files = self._filter_valid_files(event.mimeData().urls())
            if valid_files:
                # 发送文件列表信号
                self.files_dropped.emit(valid_files)
                event.acceptProposedAction()
                return
        event.ignore()
    
    def _filter_valid_files(self, urls):
        """过滤有效的音频文件URL"""
        valid_files = []
        for url in urls:
            file_path = url.toLocalFile()
            ext = os.path.splitext(file_path)[1].lower()
            if ext in self.accepted_extensions:
                valid_files.append(file_path)
        return valid_files 