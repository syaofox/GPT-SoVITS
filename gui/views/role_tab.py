"""
角色推理标签页

提供基于保存角色配置的语音生成功能
"""

import os
from typing import Dict
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, 
    QGroupBox, QLabel, QTextEdit, QComboBox, QPushButton, 
    QMessageBox, QSplitter
)
from PySide6.QtGui import QTextCursor

from gui.components.audio_player import AudioPlayer
from gui.components.history_list import HistoryList
from gui.components.optimized_widgets import OptimizedTextEdit


class RoleTab(QWidget):
    """角色选项卡，用于使用已保存的角色进行推理"""
    
    # 添加生成请求信号
    generate_requested = Signal(dict, str, bool)
    
    def __init__(self, role_controller, inference_controller, shared_controls=True, parent=None):
        super().__init__(parent)
        
        self.role_controller = role_controller
        self.inference_controller = inference_controller
        self.shared_controls = shared_controls
        
        # 初始化
        self.init_ui()
        self.connect_signals()
    
    def init_ui(self):
        """初始化界面"""
        main_layout = QHBoxLayout(self)
        
        # 左侧面板
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # 文本输入
        text_group = QGroupBox("文本输入")
        text_layout = QVBoxLayout(text_group)
        
        # 使用优化后的文本编辑框
        self.text_edit = OptimizedTextEdit()
        self.text_edit.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.text_edit.setLineWrapMode(QTextEdit.WidgetWidth)
        
        text_layout.addWidget(self.text_edit)
        
        left_layout.addWidget(text_group)
        
        # 如果需要内置播放器和历史记录
        if not self.shared_controls:
            # 右侧面板
            right_panel = QWidget()
            right_layout = QVBoxLayout(right_panel)
            
            # 音频播放器
            self.audio_player = AudioPlayer()
            right_layout.addWidget(self.audio_player)
            
            # 历史记录
            self.history_list = HistoryList()
            right_layout.addWidget(self.history_list)
            
            # 添加到主布局
            splitter = QSplitter(Qt.Horizontal)
            splitter.addWidget(left_panel)
            splitter.addWidget(right_panel)
            splitter.setSizes([400, 300])
            
            main_layout.addWidget(splitter)
        else:
            # 只添加左侧配置面板
            main_layout.addWidget(left_panel)
    
    def connect_signals(self):
        """连接信号槽"""
        self.inference_controller.inference_failed.connect(self.on_inference_failed)
        
        # 只有在非共享模式下才需要连接这些信号
        if not self.shared_controls:
            self.inference_controller.inference_completed.connect(self.on_inference_completed)
            self.inference_controller.history_updated.connect(self.update_history)
            self.history_list.audio_selected.connect(self.audio_player.load_audio)
            self.history_list.history_cleared.connect(self.inference_controller.clear_history)
    
    def update_history(self):
        """更新历史记录"""
        if not self.shared_controls:
            history = self.inference_controller.get_history()
            self.history_list.update_history(history)
    
    def on_inference_failed(self, error_msg: str):
        """推理失败回调"""
        if not self.shared_controls:
            self.generate_button.setEnabled(True)
            self.progress_label.setText(f"生成失败: {error_msg}") 