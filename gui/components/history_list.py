"""
历史记录列表组件

显示和管理音频合成历史记录
"""

import os
import subprocess
from typing import Dict, List
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QListView, QMessageBox
from PySide6.QtGui import QStandardItemModel, QStandardItem


class HistoryList(QWidget):
    """历史记录列表组件"""
    
    audio_selected = Signal(str)
    history_cleared = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.layout = QVBoxLayout(self)
        
        # 标题
        title_label = QLabel("历史记录")
        self.layout.addWidget(title_label)
        
        # 历史列表
        self.history_model = QStandardItemModel()
        self.history_view = QListView()
        self.history_view.setModel(self.history_model)
        self.history_view.clicked.connect(self.on_item_clicked)
        self.history_view.doubleClicked.connect(self.on_item_double_clicked)
        
        # 设置为不可编辑
        self.history_view.setEditTriggers(QListView.NoEditTriggers)
        
        self.layout.addWidget(self.history_view)
        
        # 清空按钮
        self.clear_button = QPushButton("清空历史")
        self.clear_button.clicked.connect(self.clear_history)
        self.layout.addWidget(self.clear_button)
    
    def update_history(self, history: List[Dict]):
        """更新历史记录"""
        self.history_model.clear()
        
        for entry in reversed(history):  # 最新的在最前面
            timestamp = entry.get("timestamp", "")
            text = entry.get("text", "")
            display_text = f"{timestamp}: {text[:30]}..." if len(text) > 30 else f"{timestamp}: {text}"
            
            item = QStandardItem(display_text)
            item.setData(entry.get("path", ""), Qt.UserRole)
            self.history_model.appendRow(item)
    
    def on_item_clicked(self, index):
        """历史项点击回调"""
        if not index.isValid():
            return
            
        item = self.history_model.itemFromIndex(index)
        path = item.data(Qt.UserRole)
        if path and os.path.exists(path):
            self.audio_selected.emit(path)
    
    def on_item_double_clicked(self, index):
        """历史项双击回调 - 使用系统播放器打开"""
        if not index.isValid():
            return
            
        item = self.history_model.itemFromIndex(index)
        path = item.data(Qt.UserRole)
        if path and os.path.exists(path):
            try:
                # 使用系统默认程序打开音频文件
                if os.name == 'nt':  # Windows
                    os.startfile(path)
                elif os.name == 'posix':  # macOS/Linux
                    # 对于macOS，用open命令；对于Linux，用xdg-open命令
                    if os.uname().sysname == 'Darwin':  # macOS
                        subprocess.Popen(['open', path])
                    else:  # Linux
                        subprocess.Popen(['xdg-open', path])
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "打开失败",
                    f"无法使用系统播放器打开文件: {str(e)}"
                )
    
    def clear_history(self):
        """清空历史记录"""
        reply = QMessageBox.question(
            self,
            "清空历史", 
            "确定要清空所有历史记录吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.history_model.clear()
            self.history_cleared.emit() 