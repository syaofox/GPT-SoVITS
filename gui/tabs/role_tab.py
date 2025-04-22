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

from gui.components.audio_player import AudioPlayer
from gui.components.history_list import HistoryList


class RoleTab(QWidget):
    """角色选项卡，用于使用已保存的角色进行推理"""
    
    # 添加生成请求信号
    generate_requested = Signal(dict, str, bool)
    # 添加角色配置更新信号
    role_config_selected = Signal(dict)
    # 添加角色名和情感名更新信号
    role_info_selected = Signal(str, str)
    
    def __init__(self, role_controller, inference_controller, shared_controls=True, parent=None):
        super().__init__(parent)
        
        self.role_controller = role_controller
        self.inference_controller = inference_controller
        self.shared_controls = shared_controls
        
        self.current_role = ""
        self.current_emotion = ""
        
        # 初始化
        self.init_ui()
        self.connect_signals()
        self.refresh_roles()
    
    def init_ui(self):
        """初始化界面"""
        main_layout = QHBoxLayout(self)
        
        # 左侧面板
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # 角色选择
        role_group = QGroupBox("角色选择")
        role_layout = QGridLayout(role_group)
        
        role_layout.addWidget(QLabel("角色:"), 0, 0)
        self.role_combo = QComboBox()
        self.role_combo.currentIndexChanged.connect(self.on_role_changed)
        role_layout.addWidget(self.role_combo, 0, 1)
        
        role_layout.addWidget(QLabel("情感:"), 1, 0)
        self.emotion_combo = QComboBox()
        self.emotion_combo.currentIndexChanged.connect(self.on_emotion_changed)
        role_layout.addWidget(self.emotion_combo, 1, 1)
        
        self.refresh_button = QPushButton("刷新列表")
        self.refresh_button.clicked.connect(self.refresh_roles)
        role_layout.addWidget(self.refresh_button, 2, 0, 1, 2)
        
        left_layout.addWidget(role_group)
        
        # 文本输入
        text_group = QGroupBox("文本输入")
        text_layout = QVBoxLayout(text_group)
        
        self.text_edit = QTextEdit()
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
    
    def refresh_roles(self):
        """刷新角色列表"""
        self.role_controller.refresh_roles()
        roles = self.role_controller.get_role_names()
        
        # 保存当前选择
        current_role = self.role_combo.currentText()
        
        # 更新角色下拉框
        self.role_combo.clear()
        self.role_combo.addItems(roles)
        
        # 恢复之前的选择
        if current_role and current_role in roles:
            index = self.role_combo.findText(current_role)
            self.role_combo.setCurrentIndex(index)
    
    def on_role_changed(self, index):
        """角色改变回调"""
        if index < 0:
            return
            
        self.current_role = self.role_combo.currentText()
        self.update_emotions()
        
        # 发射角色和情感更新信号
        if self.current_role and self.current_emotion:
            self.role_info_selected.emit(self.current_role, self.current_emotion)
    
    def update_emotions(self):
        """更新情感列表"""
        if not self.current_role:
            self.emotion_combo.clear()
            return
            
        emotions = self.role_controller.get_emotion_names(self.current_role)
        
        # 保存当前选择
        current_emotion = self.emotion_combo.currentText()
        
        # 更新情感下拉框
        self.emotion_combo.clear()
        self.emotion_combo.addItems(emotions)
        
        # 恢复之前的选择
        if current_emotion and current_emotion in emotions:
            index = self.emotion_combo.findText(current_emotion)
            self.emotion_combo.setCurrentIndex(index)
        elif emotions:
            # 默认选择第一个情感并更新配置
            self.emotion_combo.setCurrentIndex(0)
        
        # 情感更新后获取当前配置
        self.get_current_emotion_config()
    
    def on_emotion_changed(self, index):
        """情感改变回调"""
        if index < 0:
            return
            
        self.current_emotion = self.emotion_combo.currentText()
        # 情感改变时获取当前配置
        self.get_current_emotion_config()
        
        # 发射角色和情感更新信号
        if self.current_role and self.current_emotion:
            self.role_info_selected.emit(self.current_role, self.current_emotion)
    
    def get_current_emotion_config(self):
        """获取当前选中角色和情感的配置，并发射信号更新试听配置"""
        if not self.current_role or not self.current_emotion:
            return
            
        # 获取当前情感配置
        emotion_config = self.role_controller.get_emotion_config(self.current_role, self.current_emotion)
        if emotion_config:
            # 打印调试信息以便追踪
            print(f"获取角色配置: {self.current_role}/{self.current_emotion}")
            print(f"配置中的模型: gpt={emotion_config.get('gpt_path', '未设置')}, sovits={emotion_config.get('sovits_path', '未设置')}")
            
            # 发射配置更新信号
            self.role_config_selected.emit(emotion_config)
    
    def update_history(self):
        """更新历史记录"""
        if not self.shared_controls:
            history = self.inference_controller.get_history()
            self.history_list.update_history(history)
    
    def get_inference_config(self):
        """获取用于推理的配置"""
        if not self.current_role or not self.current_emotion:
            return None
            
        # 获取角色配置
        config = self.role_controller.get_emotion_config(self.current_role, self.current_emotion)
        if not config:
            return None
        
        # 添加文本
        config["text"] = self.text_edit.toPlainText()
        
        # 确保角色名和情绪名存在于配置中
        config["role_name"] = self.current_role
        config["emotion_name"] = self.current_emotion
        
        return config
    
    def on_inference_failed(self, error_msg: str):
        """推理失败回调"""
        if not self.shared_controls:
            self.generate_button.setEnabled(True)
            self.progress_label.setText(f"生成失败: {error_msg}") 