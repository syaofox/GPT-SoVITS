"""
试听配置标签页

提供语音合成试听和角色创建功能
"""

import os
from typing import Dict
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, 
    QGroupBox, QLabel, QLineEdit, QTextEdit, QComboBox, 
    QPushButton, QFileDialog, QMessageBox, QSpinBox,
    QDoubleSpinBox, QCheckBox, QSplitter, QInputDialog
)

from gui.components.audio_player import AudioPlayer
from gui.components.history_list import HistoryList


class ExperimentTab(QWidget):
    """实验选项卡，用于试听和创建角色配置"""
    
    def __init__(self, role_controller, inference_controller, parent=None):
        super().__init__(parent)
        
        self.role_controller = role_controller
        self.inference_controller = inference_controller
        
        # 模型路径字典
        self.gpt_model_paths = {}
        self.sovits_model_paths = {}
        
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
        self.cut_method_combo.addItems(["按句切", "凑四句一切", "按标点切", "按段切"])        
        synth_layout.addWidget(self.cut_method_combo, 3, 1, 1, 2)
        
        synth_layout.addWidget(QLabel("GPT模型:"), 4, 0)
        self.gpt_model_combo = QComboBox()
        synth_layout.addWidget(self.gpt_model_combo, 4, 1, 1, 2)
        
        synth_layout.addWidget(QLabel("SoVITS模型:"), 5, 0)
        self.sovits_model_combo = QComboBox()
        synth_layout.addWidget(self.sovits_model_combo, 5, 1, 1, 2)
        
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
        self.top_k_spin.setValue(15)
        adv_layout.addWidget(self.top_k_spin, 1, 1)
        
        adv_layout.addWidget(QLabel("Top P:"), 2, 0)
        self.top_p_spin = QDoubleSpinBox()
        self.top_p_spin.setRange(0.0, 1.0)
        self.top_p_spin.setSingleStep(0.05)
        self.top_p_spin.setValue(1.0)
        adv_layout.addWidget(self.top_p_spin, 2, 1)
        
        adv_layout.addWidget(QLabel("温度:"), 3, 0)
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.1, 2.0)
        self.temperature_spin.setSingleStep(0.05)
        self.temperature_spin.setValue(1.0)
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

        # 进度条
        progress_layout = QHBoxLayout()
        self.progress_label = QLabel("就绪")
        progress_layout.addWidget(self.progress_label)
        left_layout.addLayout(progress_layout)
        
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
        self.inference_controller.progress_updated.connect(self.update_progress)
        self.inference_controller.inference_started.connect(self.on_inference_started)
        self.history_list.audio_selected.connect(self.audio_player.load_audio)
        self.history_list.history_cleared.connect(self.inference_controller.clear_history)
    
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
        self.save_role_button.setEnabled(False)
        self.progress_label.setText("正在初始化...")
    
    def browse_ref_audio(self):
        """浏览参考音频文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择参考音频", "", "音频文件 (*.wav *.mp3);;所有文件 (*.*)"
        )
        if file_path:
            self.ref_path_edit.setText(file_path)
            
            # 提取文件名作为参考文本（去除路径和扩展名）
            file_name = os.path.basename(file_path)
            file_name_without_ext = os.path.splitext(file_name)[0]
            
            # 如果参考文本框为空，则填入文件名
            if not self.prompt_text_edit.toPlainText().strip():
                self.prompt_text_edit.setText(file_name_without_ext)
    
    def load_gpt_models(self, model_dict):
        """加载GPT模型文件到下拉框"""
        self.gpt_model_combo.clear()
        self.gpt_model_paths = model_dict
        
        for display_name in model_dict.keys():
            self.gpt_model_combo.addItem(display_name)
    
    def load_sovits_models(self, model_dict):
        """加载SoVITS模型文件到下拉框"""
        self.sovits_model_combo.clear()
        self.sovits_model_paths = model_dict
        
        for display_name in model_dict.keys():
            self.sovits_model_combo.addItem(display_name)
    
    def get_current_config(self) -> Dict:
        """获取当前配置"""
        # 获取选中的模型实际路径
        gpt_display_name = self.gpt_model_combo.currentText()
        sovits_display_name = self.sovits_model_combo.currentText()
        
        gpt_path = self.gpt_model_paths.get(gpt_display_name, "")
        sovits_path = self.sovits_model_paths.get(sovits_display_name, "")
        
        return {
            "ref_audio": self.ref_path_edit.text(),
            "prompt_text": self.prompt_text_edit.toPlainText(),
            "prompt_lang": self.prompt_lang_combo.currentText(),
            "text": self.text_edit.toPlainText(),
            "text_lang": self.text_lang_combo.currentText(),
            "how_to_cut": self.cut_method_combo.currentText(),
            "gpt_path": gpt_path,
            "sovits_path": sovits_path,
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
        self.inference_controller.generate_speech_async(config)
    
    def on_inference_completed(self, file_path: str):
        """推理完成回调"""
        self.generate_button.setEnabled(True)
        self.save_role_button.setEnabled(True)
        self.progress_label.setText("生成完成")
        self.audio_player.load_audio(file_path)
        QMessageBox.information(self, "生成成功", f"音频已保存至: {file_path}")
    
    def on_inference_failed(self, error_msg: str):
        """推理失败回调"""
        self.generate_button.setEnabled(True)
        self.save_role_button.setEnabled(True)
        self.progress_label.setText("生成失败")
        QMessageBox.critical(self, "生成失败", error_msg)
    
    def save_as_role(self):
        """保存为角色配置"""
        role_name, ok = QInputDialog.getText(self, "保存角色", "角色名称:")
        if not ok or not role_name:
            return
            
        emotion_name, ok = QInputDialog.getText(self, "保存情感", "情感名称:")
        if not ok or not emotion_name:
            return
        
        # 检查角色是否存在
        exists = role_name in self.role_controller.get_role_names()
        if exists:
            emotions = self.role_controller.get_emotion_names(role_name)
            if emotion_name in emotions:
                # 情感已存在，提示将覆盖
                reply = QMessageBox.question(
                    self, 
                    "情感已存在", 
                    f"角色 '{role_name}' 中已存在名为 '{emotion_name}' 的情感，是否覆盖？",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return
            else:
                # 角色存在但情感不存在，提示将添加
                QMessageBox.information(
                    self, 
                    "添加情感", 
                    f"将向角色 '{role_name}' 添加名为 '{emotion_name}' 的新情感"
                )
            
        config = self.get_current_config()
        role_config = {
            "emotions": {
                emotion_name: config
            }
        }
        
        success = self.role_controller.save_role_config(role_name, role_config)
        if success:
            if exists:
                QMessageBox.information(self, "保存成功", f"角色 '{role_name}' 的情感 '{emotion_name}' 保存成功")
            else:
                QMessageBox.information(self, "保存成功", f"新角色 '{role_name}' 创建成功")
                
            # 刷新角色列表以显示更新
            self.role_controller.refresh_roles() 