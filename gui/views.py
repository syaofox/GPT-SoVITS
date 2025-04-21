"""
GUI视图层

处理用户界面展示和交互
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
from PySide6.QtCore import Qt, Signal, Slot, Property, QUrl
from PySide6.QtGui import QIcon, QPixmap, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QTextEdit, QComboBox, QPushButton, QFileDialog,
    QTabWidget, QGroupBox, QListView, QSlider, QDoubleSpinBox, QSpinBox,
    QCheckBox, QMessageBox, QSplitter, QScrollArea, QFrame, QProgressBar
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput, QMediaDevices

from gui.controllers import RoleController, InferenceController


class AudioPlayer(QWidget):
    """音频播放器组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 创建媒体播放器
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        
        # 创建UI组件
        layout = QVBoxLayout(self)
        
        # 播放控制按钮
        controls_layout = QHBoxLayout()
        
        self.play_button = QPushButton("播放")
        self.play_button.clicked.connect(self.toggle_playback)
        controls_layout.addWidget(self.play_button)
        
        self.stop_button = QPushButton("停止")
        self.stop_button.clicked.connect(self.stop_playback)
        controls_layout.addWidget(self.stop_button)
        
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setRange(0, 100)
        self.progress_slider.sliderMoved.connect(self.set_position)
        controls_layout.addWidget(self.progress_slider)
        
        # 音量控制
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(70)
        self.volume_slider.valueChanged.connect(self.set_volume)
        controls_layout.addWidget(self.volume_slider)
        
        layout.addLayout(controls_layout)
        
        # 设置信号连接
        self.player.positionChanged.connect(self.position_changed)
        self.player.durationChanged.connect(self.duration_changed)
        self.player.playbackStateChanged.connect(self.state_changed)
        
        # 设置初始音量
        self.set_volume(70)
    
    def load_audio(self, file_path: str):
        """加载音频文件"""
        self.player.setSource(QUrl.fromLocalFile(file_path))
        self.play_button.setText("播放")
    
    def toggle_playback(self):
        """切换播放/暂停状态"""
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()
    
    def stop_playback(self):
        """停止播放"""
        self.player.stop()
    
    def set_position(self, position):
        """设置播放位置"""
        self.player.setPosition(position)
    
    def set_volume(self, volume):
        """设置音量"""
        self.audio_output.setVolume(volume / 100.0)
    
    def position_changed(self, position):
        """播放位置变化回调"""
        self.progress_slider.setValue(position)
    
    def duration_changed(self, duration):
        """音频时长变化回调"""
        self.progress_slider.setRange(0, duration)
    
    def state_changed(self, state):
        """播放状态变化回调"""
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.play_button.setText("暂停")
        else:
            self.play_button.setText("播放")


class HistoryList(QWidget):
    """历史记录列表组件"""
    
    audio_selected = Signal(str)
    
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
    
    def clear_history(self):
        """清空历史记录"""
        self.history_model.clear()


class ExperimentTab(QWidget):
    """实验选项卡，用于试听和创建角色配置"""
    
    def __init__(self, role_controller, inference_controller, parent=None):
        super().__init__(parent)
        
        self.role_controller = role_controller
        self.inference_controller = inference_controller
        
        # 初始化
        self.init_ui()
        self.connect_signals()
    
    def init_ui(self):
        """初始化界面"""
        main_layout = QHBoxLayout(self)
        
        # 左侧配置面板
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # 参考音频设置
        ref_group = QGroupBox("参考音频设置")
        ref_layout = QGridLayout(ref_group)
        
        ref_layout.addWidget(QLabel("参考音频:"), 0, 0)
        self.ref_path_edit = QLineEdit()
        self.ref_path_edit.setReadOnly(True)
        ref_layout.addWidget(self.ref_path_edit, 0, 1)
        
        self.ref_browse_button = QPushButton("浏览...")
        self.ref_browse_button.clicked.connect(self.browse_ref_audio)
        ref_layout.addWidget(self.ref_browse_button, 0, 2)
        
        ref_layout.addWidget(QLabel("参考文本:"), 1, 0)
        self.prompt_text_edit = QTextEdit()
        self.prompt_text_edit.setMaximumHeight(60)
        ref_layout.addWidget(self.prompt_text_edit, 1, 1, 1, 2)
        
        ref_layout.addWidget(QLabel("参考语言:"), 2, 0)
        self.prompt_lang_combo = QComboBox()
        self.prompt_lang_combo.addItems(["中文", "英文", "日文"])
        ref_layout.addWidget(self.prompt_lang_combo, 2, 1, 1, 2)
        
        left_layout.addWidget(ref_group)
        
        # 合成设置
        synth_group = QGroupBox("合成设置")
        synth_layout = QGridLayout(synth_group)
        
        synth_layout.addWidget(QLabel("合成文本:"), 0, 0)
        self.text_edit = QTextEdit()
        synth_layout.addWidget(self.text_edit, 0, 1, 2, 2)
        
        synth_layout.addWidget(QLabel("文本语言:"), 2, 0)
        self.text_lang_combo = QComboBox()
        self.text_lang_combo.addItems(["中文", "英文", "日文"])
        synth_layout.addWidget(self.text_lang_combo, 2, 1, 1, 2)
        
        synth_layout.addWidget(QLabel("文本切分:"), 3, 0)
        self.cut_method_combo = QComboBox()
        self.cut_method_combo.addItems(["凑四句一切", "按标点切", "按句切", "按段切"])
        synth_layout.addWidget(self.cut_method_combo, 3, 1, 1, 2)
        
        synth_layout.addWidget(QLabel("GPT模型:"), 4, 0)
        self.gpt_path_edit = QLineEdit()
        self.gpt_path_edit.setReadOnly(True)
        synth_layout.addWidget(self.gpt_path_edit, 4, 1)
        
        self.gpt_browse_button = QPushButton("浏览...")
        self.gpt_browse_button.clicked.connect(self.browse_gpt_model)
        synth_layout.addWidget(self.gpt_browse_button, 4, 2)
        
        synth_layout.addWidget(QLabel("SoVITS模型:"), 5, 0)
        self.sovits_path_edit = QLineEdit()
        self.sovits_path_edit.setReadOnly(True)
        synth_layout.addWidget(self.sovits_path_edit, 5, 1)
        
        self.sovits_browse_button = QPushButton("浏览...")
        self.sovits_browse_button.clicked.connect(self.browse_sovits_model)
        synth_layout.addWidget(self.sovits_browse_button, 5, 2)
        
        left_layout.addWidget(synth_group)
        
        # 高级设置
        adv_group = QGroupBox("高级设置")
        adv_layout = QGridLayout(adv_group)
        
        adv_layout.addWidget(QLabel("语速:"), 0, 0)
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.6, 1.65)
        self.speed_spin.setSingleStep(0.05)
        self.speed_spin.setValue(1.0)
        adv_layout.addWidget(self.speed_spin, 0, 1)
        
        adv_layout.addWidget(QLabel("Top K:"), 1, 0)
        self.top_k_spin = QSpinBox()
        self.top_k_spin.setRange(1, 50)
        self.top_k_spin.setValue(20)
        adv_layout.addWidget(self.top_k_spin, 1, 1)
        
        adv_layout.addWidget(QLabel("Top P:"), 2, 0)
        self.top_p_spin = QDoubleSpinBox()
        self.top_p_spin.setRange(0.0, 1.0)
        self.top_p_spin.setSingleStep(0.05)
        self.top_p_spin.setValue(0.6)
        adv_layout.addWidget(self.top_p_spin, 2, 1)
        
        adv_layout.addWidget(QLabel("温度:"), 3, 0)
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.1, 2.0)
        self.temperature_spin.setSingleStep(0.05)
        self.temperature_spin.setValue(0.6)
        adv_layout.addWidget(self.temperature_spin, 3, 1)
        
        adv_layout.addWidget(QLabel("采样步数:"), 4, 0)
        self.sample_steps_spin = QSpinBox()
        self.sample_steps_spin.setRange(1, 50)
        self.sample_steps_spin.setValue(8)
        adv_layout.addWidget(self.sample_steps_spin, 4, 1)
        
        adv_layout.addWidget(QLabel("句间停顿:"), 5, 0)
        self.pause_spin = QDoubleSpinBox()
        self.pause_spin.setRange(0.0, 2.0)
        self.pause_spin.setSingleStep(0.1)
        self.pause_spin.setValue(0.3)
        adv_layout.addWidget(self.pause_spin, 5, 1)
        
        self.ref_free_check = QCheckBox("无参考文本")
        adv_layout.addWidget(self.ref_free_check, 6, 0)
        
        self.sr_check = QCheckBox("超采样(V3)")
        adv_layout.addWidget(self.sr_check, 6, 1)
        
        left_layout.addWidget(adv_group)
        
        # 操作按钮
        buttons_layout = QHBoxLayout()
        
        self.generate_button = QPushButton("生成语音")
        self.generate_button.clicked.connect(self.generate_speech)
        buttons_layout.addWidget(self.generate_button)
        
        self.save_role_button = QPushButton("保存为角色")
        self.save_role_button.clicked.connect(self.save_as_role)
        buttons_layout.addWidget(self.save_role_button)
        
        left_layout.addLayout(buttons_layout)
        
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
        splitter.setSizes([500, 300])
        
        main_layout.addWidget(splitter)
    
    def connect_signals(self):
        """连接信号槽"""
        self.inference_controller.inference_completed.connect(self.on_inference_completed)
        self.inference_controller.inference_failed.connect(self.on_inference_failed)
        self.inference_controller.history_updated.connect(self.update_history)
        self.history_list.audio_selected.connect(self.audio_player.load_audio)
    
    def update_history(self):
        """更新历史记录"""
        history = self.inference_controller.get_history()
        self.history_list.update_history(history)
    
    def browse_ref_audio(self):
        """浏览参考音频文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择参考音频", "", "音频文件 (*.wav *.mp3);;所有文件 (*.*)"
        )
        if file_path:
            self.ref_path_edit.setText(file_path)
    
    def browse_gpt_model(self):
        """浏览GPT模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择GPT模型", "", "模型文件 (*.ckpt *.pth);;所有文件 (*.*)"
        )
        if file_path:
            self.gpt_path_edit.setText(file_path)
    
    def browse_sovits_model(self):
        """浏览SoVITS模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择SoVITS模型", "", "模型文件 (*.pth);;所有文件 (*.*)"
        )
        if file_path:
            self.sovits_path_edit.setText(file_path)
    
    def get_current_config(self) -> Dict:
        """获取当前配置"""
        return {
            "ref_audio": self.ref_path_edit.text(),
            "prompt_text": self.prompt_text_edit.toPlainText(),
            "prompt_lang": self.prompt_lang_combo.currentText(),
            "text": self.text_edit.toPlainText(),
            "text_lang": self.text_lang_combo.currentText(),
            "how_to_cut": self.cut_method_combo.currentText(),
            "gpt_path": self.gpt_path_edit.text(),
            "sovits_path": self.sovits_path_edit.text(),
            "speed": self.speed_spin.value(),
            "top_k": self.top_k_spin.value(),
            "top_p": self.top_p_spin.value(),
            "temperature": self.temperature_spin.value(),
            "sample_steps": self.sample_steps_spin.value(),
            "pause_second": self.pause_spin.value(),
            "ref_free": self.ref_free_check.isChecked(),
            "if_sr": self.sr_check.isChecked(),
            "aux_refs": []  # 实验选项卡不支持多参考音频融合
        }
    
    def generate_speech(self):
        """生成语音"""
        config = self.get_current_config()
        result = self.inference_controller.generate_speech(config)
    
    def on_inference_completed(self, file_path: str):
        """推理完成回调"""
        self.audio_player.load_audio(file_path)
        QMessageBox.information(self, "生成成功", f"音频已保存至: {file_path}")
    
    def on_inference_failed(self, error_msg: str):
        """推理失败回调"""
        QMessageBox.critical(self, "生成失败", error_msg)
    
    def save_as_role(self):
        """保存为角色配置"""
        role_name, ok = QInputDialog.getText(self, "保存角色", "角色名称:")
        if not ok or not role_name:
            return
            
        emotion_name, ok = QInputDialog.getText(self, "保存情感", "情感名称:")
        if not ok or not emotion_name:
            return
            
        config = self.get_current_config()
        role_config = {
            "emotions": {
                emotion_name: config
            }
        }
        
        success = self.role_controller.save_role_config(role_name, role_config)
        if success:
            QMessageBox.information(self, "保存成功", f"角色 {role_name} 保存成功")


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
        self.history_list.audio_selected.connect(self.audio_player.load_audio)
    
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
            
        # 更新文本
        config["text"] = text
        
        # 生成语音
        result = self.inference_controller.generate_speech(config)
    
    def on_inference_completed(self, file_path: str):
        """推理完成回调"""
        self.audio_player.load_audio(file_path)
        QMessageBox.information(self, "生成成功", f"音频已保存至: {file_path}")
    
    def on_inference_failed(self, error_msg: str):
        """推理失败回调"""
        QMessageBox.critical(self, "生成失败", error_msg)


class MainWindow(QMainWindow):
    """主窗口"""
    
    def __init__(self):
        super().__init__()
        
        # 初始化控制器
        self.role_controller = RoleController()
        self.inference_controller = InferenceController()
        
        self.init_ui()
        self.connect_signals()
    
    def init_ui(self):
        """初始化界面"""
        self.setWindowTitle("GPT-SoVITS 语音合成")
        self.setMinimumSize(900, 600)
        
        # 中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 选项卡
        self.tab_widget = QTabWidget()
        
        # 实验选项卡
        self.experiment_tab = ExperimentTab(self.role_controller, self.inference_controller)
        self.tab_widget.addTab(self.experiment_tab, "试听配置")
        
        # 角色选项卡
        self.role_tab = RoleTab(self.role_controller, self.inference_controller)
        self.tab_widget.addTab(self.role_tab, "角色推理")
        
        main_layout.addWidget(self.tab_widget)
        
        # 状态栏
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("就绪")
    
    def connect_signals(self):
        """连接信号槽"""
        self.role_controller.error_occurred.connect(self.show_error)
        self.inference_controller.error_occurred.connect(self.show_error)
        
        self.inference_controller.inference_started.connect(lambda: self.status_bar.showMessage("正在生成语音..."))
        self.inference_controller.inference_completed.connect(lambda _: self.status_bar.showMessage("生成完成"))
        self.inference_controller.inference_failed.connect(lambda err: self.status_bar.showMessage(f"生成失败: {err}"))
    
    def show_error(self, message: str):
        """显示错误信息"""
        QMessageBox.critical(self, "错误", message)


from PySide6.QtWidgets import QInputDialog 