"""
角色推理标签页

提供基于保存角色配置的语音生成功能
"""

import os
from typing import Dict
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, 
    QGroupBox, QLabel, QTextEdit, QComboBox, QPushButton, 
    QMessageBox, QSplitter
)

from gui.components.audio_player import AudioPlayer
from gui.components.history_list import HistoryList


class RoleTab(QWidget):
    """角色选项卡，用于使用已保存的角色进行推理"""
    
    def __init__(self, role_controller, inference_controller, parent=None):
        super().__init__(parent)
        
        self.role_controller = role_controller
        self.inference_controller = inference_controller
        
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
        
        # 进度信息
        self.progress_label = QLabel("就绪")
        left_layout.addWidget(self.progress_label)
        
        # 生成按钮
        self.generate_button = QPushButton("生成语音")
        self.generate_button.clicked.connect(self.generate_speech)
        left_layout.addWidget(self.generate_button)
        
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
    
    def connect_signals(self):
        """连接信号槽"""
        self.inference_controller.inference_completed.connect(self.on_inference_completed)
        self.inference_controller.inference_failed.connect(self.on_inference_failed)
        self.inference_controller.history_updated.connect(self.update_history)
        self.inference_controller.progress_updated.connect(self.update_progress)
        self.inference_controller.inference_started.connect(self.on_inference_started)
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
    
    def on_emotion_changed(self, index):
        """情感改变回调"""
        if index < 0:
            return
            
        self.current_emotion = self.emotion_combo.currentText()
    
    def update_history(self):
        """更新历史记录"""
        history = self.inference_controller.get_history()
        self.history_list.update_history(history)
    
    def update_progress(self, message: str):
        """更新进度信息"""
        self.progress_label.setText(message)
    
    def on_inference_started(self):
        """推理开始回调"""
        self.generate_button.setEnabled(False)
        self.progress_label.setText("正在初始化...")
    
    def generate_speech(self):
        """生成语音"""
        if not self.current_role or not self.current_emotion:
            QMessageBox.warning(self, "警告", "请先选择角色和情感")
            return
            
        text = self.text_edit.toPlainText()
        if not text:
            QMessageBox.warning(self, "警告", "请输入要合成的文本")
            return
            
        # 获取角色配置
        config = self.role_controller.get_emotion_config(self.current_role, self.current_emotion)
        if not config:
            QMessageBox.warning(self, "警告", "无法获取角色配置")
            return
            
        # 检查参考音频路径是否存在
        ref_audio = config.get("ref_audio", "")
        if ref_audio and not os.path.exists(ref_audio):
            QMessageBox.warning(self, "警告", f"参考音频文件不存在: {ref_audio}")
            return
            
        # 检查辅助参考音频
        aux_refs = config.get("aux_refs", [])
        valid_aux_refs = []
        for aux_ref in aux_refs:
            if os.path.exists(aux_ref):
                valid_aux_refs.append(aux_ref)
            else:
                print(f"辅助参考音频不存在，已忽略: {aux_ref}")
        config["aux_refs"] = valid_aux_refs
            
        # 更新文本
        config["text"] = text
        
        # 生成语音（异步）
        self.inference_controller.generate_speech_async(config)
    
    def on_inference_completed(self, file_path: str):
        """推理完成回调"""
        self.generate_button.setEnabled(True)
        self.progress_label.setText("生成完成")
        self.audio_player.load_audio(file_path)
        QMessageBox.information(self, "生成成功", f"音频已保存至: {file_path}")
    
    def on_inference_failed(self, error_msg: str):
        """推理失败回调"""
        self.generate_button.setEnabled(True)
        self.progress_label.setText("生成失败")
        QMessageBox.critical(self, "生成失败", error_msg) 