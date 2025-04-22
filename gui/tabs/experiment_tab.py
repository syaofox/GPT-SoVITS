"""
试听配置标签页

提供语音合成试听和角色创建功能
"""

import os
from typing import Dict
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, 
    QGroupBox, QLabel, QLineEdit, QTextEdit, QComboBox, 
    QPushButton, QFileDialog, QMessageBox, QSpinBox,
    QDoubleSpinBox, QCheckBox, QSplitter, QInputDialog,
    QListWidget, QListWidgetItem
)

from gui.components.audio_player import AudioPlayer
from gui.components.history_list import HistoryList


class ExperimentTab(QWidget):
    """实验选项卡，用于试听和创建角色配置"""
    
    # 添加生成请求信号
    generate_requested = Signal(dict, str, bool)
    
    def __init__(self, role_controller, inference_controller, shared_controls=True, parent=None):
        super().__init__(parent)
        
        self.role_controller = role_controller
        self.inference_controller = inference_controller
        self.shared_controls = shared_controls
        
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
        
        # 辅助参考音频
        ref_layout.addWidget(QLabel("辅助参考音频:"), 3, 0)
        aux_refs_layout = QHBoxLayout()
        
        self.aux_refs_list = QListWidget()
        self.aux_refs_list.setMaximumHeight(80)
        aux_refs_layout.addWidget(self.aux_refs_list, 3)
        
        aux_buttons_layout = QVBoxLayout()
        self.add_aux_ref_button = QPushButton("+")
        self.add_aux_ref_button.setMaximumWidth(40)
        self.add_aux_ref_button.clicked.connect(self.add_aux_ref_audio)
        aux_buttons_layout.addWidget(self.add_aux_ref_button)
        
        self.remove_aux_ref_button = QPushButton("-")
        self.remove_aux_ref_button.setMaximumWidth(40)
        self.remove_aux_ref_button.clicked.connect(self.remove_aux_ref_audio)
        aux_buttons_layout.addWidget(self.remove_aux_ref_button)
        
        aux_refs_layout.addLayout(aux_buttons_layout)
        ref_layout.addLayout(aux_refs_layout, 3, 1, 1, 2)
        
        left_layout.addWidget(ref_group)
        
        # 合成设置
        synth_group = QGroupBox("合成设置")
        synth_layout = QGridLayout(synth_group)
        
        synth_layout.addWidget(QLabel("合成文本:"), 0, 0)
        self.text_edit = QTextEdit()
        self.text_edit.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.text_edit.setLineWrapMode(QTextEdit.WidgetWidth)
        self.text_edit.setAcceptRichText(False)
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

        # 保存为角色相关控件放在同一行
        role_save_layout = QHBoxLayout()
        
        # 角色名输入框
        role_save_layout.addWidget(QLabel("角色名:"))
        self.role_name_edit = QLineEdit()
        role_save_layout.addWidget(self.role_name_edit)
        
        # 情绪输入框
        role_save_layout.addWidget(QLabel("情绪:"))
        self.emotion_name_edit = QLineEdit()
        role_save_layout.addWidget(self.emotion_name_edit)
        
        # 保存按钮
        self.save_role_button = QPushButton("保存为角色")
        self.save_role_button.clicked.connect(self.save_as_role)
        role_save_layout.addWidget(self.save_role_button)
        
        left_layout.addLayout(role_save_layout)
        
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
            splitter.setSizes([500, 300])
            
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
    
    def browse_ref_audio(self):
        """浏览参考音频文件"""
        # 打开文件对话框
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择参考音频文件",
            "",
            "音频文件 (*.wav *.mp3 *.flac *.ogg);;所有文件 (*)"
        )
        
        if file_path:
            # 设置文件路径到参考音频文本框
            self.ref_path_edit.setText(file_path)
            
            # 提取文件名（不含扩展名）作为参考文本
            filename = os.path.basename(file_path)
            filename_without_ext = os.path.splitext(filename)[0]
            
            # 将文件名设置到参考文本编辑框（直接覆盖）
            self.prompt_text_edit.setPlainText(filename_without_ext)
    
    def load_gpt_models(self, model_dict):
        """加载GPT模型列表"""
        self.gpt_model_paths = model_dict
        self.gpt_model_combo.clear()
        self.gpt_model_combo.addItems(list(model_dict.keys()))
    
    def load_sovits_models(self, model_dict):
        """加载SoVITS模型列表"""
        self.sovits_model_paths = model_dict
        self.sovits_model_combo.clear()
        self.sovits_model_combo.addItems(list(model_dict.keys()))
    
    def get_inference_config(self):
        """获取当前推理配置"""
        config = {}
        
        # 获取基本配置
        text = self.text_edit.toPlainText().strip()
        if not text:
            return None
        
        config["text"] = text
        config["text_lang"] = self.text_lang_combo.currentText()
        config["how_to_cut"] = self.cut_method_combo.currentText()
        
        # 获取角色名和情绪名
        config["role_name"] = self.role_name_edit.text().strip()
        config["emotion_name"] = self.emotion_name_edit.text().strip()
        
        # 获取模型选择
        gpt_model_name = self.gpt_model_combo.currentText()
        sovits_model_name = self.sovits_model_combo.currentText()
        
        if not gpt_model_name or not sovits_model_name:
            return None
            
        # 转换为实际路径
        if hasattr(self, "gpt_model_paths") and gpt_model_name in self.gpt_model_paths:
            config["gpt_path"] = self.gpt_model_paths[gpt_model_name]
        else:
            return None
            
        if hasattr(self, "sovits_model_paths") and sovits_model_name in self.sovits_model_paths:
            config["sovits_path"] = self.sovits_model_paths[sovits_model_name]
        else:
            return None
        
        # 获取参考音频
        ref_audio_path = self.ref_path_edit.text()
        if not ref_audio_path and not self.ref_free_check.isChecked():
            return None
            
        config["ref_audio"] = ref_audio_path
        config["prompt_text"] = self.prompt_text_edit.toPlainText().strip()
        config["prompt_lang"] = self.prompt_lang_combo.currentText()
        
        # 高级设置
        config["speed"] = self.speed_spin.value()
        config["top_k"] = self.top_k_spin.value()
        config["top_p"] = self.top_p_spin.value()
        config["temperature"] = self.temperature_spin.value()
        config["sample_steps"] = self.sample_steps_spin.value()
        config["pause_second"] = self.pause_spin.value()
        
        # 选项
        config["ref_free"] = self.ref_free_check.isChecked()
        config["if_sr"] = self.sr_check.isChecked()
        
        # 辅助参考音频
        aux_refs = []
        for i in range(self.aux_refs_list.count()):
            aux_refs.append(self.aux_refs_list.item(i).data(Qt.UserRole))
        
        if aux_refs:
            config["aux_refs"] = aux_refs
            
        return config
    
    def generate_with_config(self):
        """使用当前配置生成语音"""
        # 获取合成文本
        text = self.text_edit.toPlainText()
        if not text:
            QMessageBox.warning(self, "警告", "请输入要合成的文本")
            return False
            
        # 获取当前配置
        config = self.get_inference_config()
        
        # 检查必要参数
        if not config["gpt_path"]:
            QMessageBox.warning(self, "警告", "请选择GPT模型")
            return False
            
        if not config["sovits_path"]:
            QMessageBox.warning(self, "警告", "请选择SoVITS模型")
            return False
            
        if not config["ref_free"] and not config["ref_audio"]:
            QMessageBox.warning(self, "警告", "请选择参考音频文件或勾选无参考文本")
            return False
            
        if not config["ref_free"] and not os.path.exists(config["ref_audio"]):
            QMessageBox.warning(self, "警告", f"参考音频文件不存在: {config['ref_audio']}")
            return False
            
        # 调用推理控制器
        self.inference_controller.generate_speech_async(config)
        return True
    
    def generate_speech(self):
        """生成语音"""
        if self.shared_controls:
            # 在共享模式下，发送信号给主窗口处理
            config = self.get_inference_config()
            text = self.text_edit.toPlainText()
            self.generate_requested.emit(config, text, False)
        else:
            # 在非共享模式下，直接处理
            self.generate_with_config()
    
    def on_inference_completed(self, file_path: str):
        """推理完成回调"""
        if not self.shared_controls:
            self.generate_button.setEnabled(True)
            self.progress_label.setText("生成完成")
            self.audio_player.load_audio(file_path)
    
    def on_inference_failed(self, error_msg: str):
        """推理失败回调"""
        if not self.shared_controls:
            self.generate_button.setEnabled(True)
            self.progress_label.setText(f"生成失败: {error_msg}")
    
    def save_as_role(self):
        """保存为角色配置"""
        # 从输入框获取角色名称和情绪
        role_name = self.role_name_edit.text().strip()
        emotion_name = self.emotion_name_edit.text().strip()
        
        # 验证输入
        if not role_name:
            QMessageBox.warning(self, "警告", "请输入角色名称")
            return
            
        if not emotion_name:
            QMessageBox.warning(self, "警告", "请输入情绪名称")
            return
            
        # 获取当前配置
        config = self.get_inference_config()
        
        # 保存角色
        try:
            self.role_controller.save_role(role_name, emotion_name, config)
            QMessageBox.information(self, "成功", f"角色 [{role_name}] 的情感 [{emotion_name}] 已保存")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存角色失败: {str(e)}")
    
    def add_aux_ref_audio(self):
        """添加辅助参考音频"""
        # 打开文件对话框
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择辅助参考音频文件",
            "",
            "音频文件 (*.wav *.mp3 *.flac *.ogg);;所有文件 (*)"
        )
        
        for file_path in file_paths:
            if file_path:
                # 检查是否已经存在
                exists = False
                for i in range(self.aux_refs_list.count()):
                    item = self.aux_refs_list.item(i)
                    if item.data(Qt.UserRole) == file_path:
                        exists = True
                        break
                
                if not exists:
                    # 添加到列表
                    item = self.create_aux_ref_item(file_path)
                    self.aux_refs_list.addItem(item)
    
    def create_aux_ref_item(self, file_path):
        """创建辅助参考音频列表项"""
        item = QListWidgetItem(os.path.basename(file_path))
        item.setData(Qt.UserRole, file_path)
        return item
    
    def remove_aux_ref_audio(self):
        """删除选中的辅助参考音频"""
        selected_items = self.aux_refs_list.selectedItems()
        for item in selected_items:
            self.aux_refs_list.takeItem(self.aux_refs_list.row(item)) 