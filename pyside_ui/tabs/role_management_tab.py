#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
角色管理标签页UI
"""

import os
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QTextEdit, QComboBox, QRadioButton, QButtonGroup, QGroupBox,
    QFileDialog, QLineEdit, QProgressBar, QMessageBox, QSplitter,
    QProgressDialog, QSlider, QCheckBox, QScrollArea, QTabWidget
)
from PySide6.QtCore import Qt, QThread, Signal

from pyside_ui.controllers.role_management_controller import RoleManagementController


class RoleManagementTab(QWidget):
    def __init__(self):
        super().__init__()
        
        # 创建控制器
        self.controller = RoleManagementController()
        
        # 初始化UI
        self.init_ui()
        
        # 连接信号
        self.connect_signals()
    
    def init_ui(self):
        """初始化UI"""
        # 主布局
        main_layout = QVBoxLayout(self)
        
        # 创建子标签页
        self.tab_widget = QTabWidget()
        
        # 创建试听标签页
        self.listening_tab = QWidget()
        listening_layout = QVBoxLayout(self.listening_tab)
        
        # 创建角色管理标签页
        self.role_management_tab = QWidget()
        role_management_layout = QVBoxLayout(self.role_management_tab)
        
        # 创建滚动区域 - 试听标签页
        listening_scroll_area = QScrollArea()
        listening_scroll_area.setWidgetResizable(True)
        listening_scroll_content = QWidget()
        listening_scroll_layout = QVBoxLayout(listening_scroll_content)
        listening_scroll_area.setWidget(listening_scroll_content)
        
        # 创建滚动区域 - 角色管理标签页
        role_management_scroll_area = QScrollArea()
        role_management_scroll_area.setWidgetResizable(True)
        role_management_scroll_content = QWidget()
        role_management_scroll_layout = QVBoxLayout(role_management_scroll_content)
        role_management_scroll_area.setWidget(role_management_scroll_content)
        
        # 模型选择区域 - 两个标签页共用
        model_group = QGroupBox("模型选择")
        model_layout = QHBoxLayout(model_group)
        
        gpt_models, sovits_models = self.controller.get_model_lists()
        
        # GPT模型选择
        gpt_layout = QVBoxLayout()
        gpt_layout.addWidget(QLabel("GPT模型列表:"))
        self.gpt_model_combo = QComboBox()
        self.gpt_model_combo.addItems(gpt_models)
        gpt_layout.addWidget(self.gpt_model_combo)
        model_layout.addLayout(gpt_layout)
        
        # SoVITS模型选择
        sovits_layout = QVBoxLayout()
        sovits_layout.addWidget(QLabel("SoVITS模型列表:"))
        self.sovits_model_combo = QComboBox()
        self.sovits_model_combo.addItems(sovits_models)
        sovits_layout.addWidget(self.sovits_model_combo)
        model_layout.addLayout(sovits_layout)
        
        # 刷新按钮
        refresh_btn = QPushButton("刷新模型列表")
        model_layout.addWidget(refresh_btn)
        
        # 添加到试听标签页
        listening_scroll_layout.addWidget(model_group)
        
        # 参考音频区域
        ref_group = QGroupBox("参考音频")
        ref_layout = QVBoxLayout(ref_group)
        
        # 参考音频文件选择
        ref_file_layout = QHBoxLayout()
        ref_file_layout.addWidget(QLabel("参考音频文件:"))
        self.ref_audio_edit = QLineEdit()
        ref_file_layout.addWidget(self.ref_audio_edit)
        browse_btn = QPushButton("浏览...")
        ref_file_layout.addWidget(browse_btn)
        ref_layout.addLayout(ref_file_layout)
        
        # 无参考文本模式
        self.ref_free_check = QCheckBox("开启无参考文本模式。不填参考文本亦相当于开启。v3暂不支持该模式，使用了会报错。")
        ref_layout.addWidget(self.ref_free_check)
        ref_layout.addWidget(QLabel("使用无参考文本模式时建议使用微调的GPT\n听不清参考音频说的啥(不晓得写啥)可以开。开启后无视填写的参考文本。"))
        
        # 参考音频文本
        ref_layout.addWidget(QLabel("参考音频的文本:"))
        self.prompt_text_edit = QTextEdit()
        self.prompt_text_edit.setMaximumHeight(100)
        ref_layout.addWidget(self.prompt_text_edit)
        
        # 语言设置
        lang_layout = QHBoxLayout()
        
        # 参考音频语种
        lang_layout.addWidget(QLabel("参考音频的语种:"))
        self.prompt_lang_combo = QComboBox()
        self.prompt_lang_combo.addItems([lang[0] for lang in self.controller.get_language_options()])
        self.prompt_lang_combo.setCurrentText("中文")
        lang_layout.addWidget(self.prompt_lang_combo)
        
        # 目标合成语种
        lang_layout.addWidget(QLabel("目标合成的语种:"))
        self.text_lang_combo = QComboBox()
        self.text_lang_combo.addItems([lang[0] for lang in self.controller.get_language_options()])
        self.text_lang_combo.setCurrentText("中文")
        lang_layout.addWidget(self.text_lang_combo)
        
        ref_layout.addLayout(lang_layout)
        
        # 辅助参考音频
        ref_layout.addWidget(QLabel("辅助参考音频（可选）：每行输入一个音频文件路径，用于融合多个音色"))
        self.aux_refs_edit = QTextEdit()
        self.aux_refs_edit.setMaximumHeight(80)
        ref_layout.addWidget(self.aux_refs_edit)
        
        # 采样步数
        sample_layout = QHBoxLayout()
        sample_layout.addWidget(QLabel("采样步数:"))
        self.sample_steps_group = QButtonGroup(self)
        for step in [4, 8, 16, 32, 64, 128]:
            radio = QRadioButton(str(step))
            self.sample_steps_group.addButton(radio, step)
            sample_layout.addWidget(radio)
            if step == 32:
                radio.setChecked(True)
        ref_layout.addLayout(sample_layout)
        
        # 超采样选项
        self.if_sr_check = QCheckBox("启用超采样提高语音质量(会增加延迟)")
        ref_layout.addWidget(self.if_sr_check)
        
        # 添加到试听标签页
        listening_scroll_layout.addWidget(ref_group)
        
        # 合成区域
        synth_group = QGroupBox("参数区")
        synth_layout = QHBoxLayout(synth_group)
        
        # 需要合成的文本
        text_layout = QVBoxLayout()
        text_layout.addWidget(QLabel("需要合成的文本:"))
        self.target_text_edit = QTextEdit()
        text_layout.addWidget(self.target_text_edit)
        synth_layout.addLayout(text_layout, 3)
        
        # 合成参数
        param_layout = QVBoxLayout()
        
        # 断句符号
        param_layout.addWidget(QLabel("断句符号:"))
        self.cut_punc_edit = QLineEdit("。！？：.!?:")
        param_layout.addWidget(self.cut_punc_edit)
        
        # 语速
        param_layout.addWidget(QLabel("语速:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(60)
        self.speed_slider.setMaximum(165)
        self.speed_slider.setValue(100)
        self.speed_label = QLabel("1.0")
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(self.speed_slider)
        speed_layout.addWidget(self.speed_label)
        param_layout.addLayout(speed_layout)
        
        # 句间停顿
        param_layout.addWidget(QLabel("句间停顿秒数:"))
        self.pause_slider = QSlider(Qt.Horizontal)
        self.pause_slider.setMinimum(10)
        self.pause_slider.setMaximum(50)
        self.pause_slider.setValue(30)
        self.pause_label = QLabel("0.3")
        pause_layout = QHBoxLayout()
        pause_layout.addWidget(self.pause_slider)
        pause_layout.addWidget(self.pause_label)
        param_layout.addLayout(pause_layout)
        
        # GPT推理参数
        param_layout.addWidget(QLabel("GPT推理参数(不懂就用默认):"))
        
        # Top-K
        param_layout.addWidget(QLabel("top_k:"))
        self.top_k_slider = QSlider(Qt.Horizontal)
        self.top_k_slider.setMinimum(1)
        self.top_k_slider.setMaximum(100)
        self.top_k_slider.setValue(15)
        self.top_k_label = QLabel("15")
        top_k_layout = QHBoxLayout()
        top_k_layout.addWidget(self.top_k_slider)
        top_k_layout.addWidget(self.top_k_label)
        param_layout.addLayout(top_k_layout)
        
        # Top-P
        param_layout.addWidget(QLabel("top_p:"))
        self.top_p_slider = QSlider(Qt.Horizontal)
        self.top_p_slider.setMinimum(0)
        self.top_p_slider.setMaximum(100)
        self.top_p_slider.setValue(100)
        self.top_p_label = QLabel("1.0")
        top_p_layout = QHBoxLayout()
        top_p_layout.addWidget(self.top_p_slider)
        top_p_layout.addWidget(self.top_p_label)
        param_layout.addLayout(top_p_layout)
        
        # Temperature
        param_layout.addWidget(QLabel("temperature:"))
        self.temperature_slider = QSlider(Qt.Horizontal)
        self.temperature_slider.setMinimum(0)
        self.temperature_slider.setMaximum(100)
        self.temperature_slider.setValue(100)
        self.temperature_label = QLabel("1.0")
        temperature_layout = QHBoxLayout()
        temperature_layout.addWidget(self.temperature_slider)
        temperature_layout.addWidget(self.temperature_label)
        param_layout.addLayout(temperature_layout)
        
        synth_layout.addLayout(param_layout, 2)
        listening_scroll_layout.addWidget(synth_group)
        
        # 合成按钮和结果
        synth_btn_layout = QHBoxLayout()
        self.synthesis_btn = QPushButton("合成语音")
        synth_btn_layout.addWidget(self.synthesis_btn)
        
        listening_scroll_layout.addLayout(synth_btn_layout)
        
        # 音频输出路径
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("输出音频:"))
        self.audio_output_path = QLineEdit()
        self.audio_output_path.setReadOnly(True)
        output_layout.addWidget(self.audio_output_path)
        self.play_btn = QPushButton("播放")
        output_layout.addWidget(self.play_btn)
        listening_scroll_layout.addLayout(output_layout)
        
        # 角色管理区域
        role_group = QGroupBox("角色管理")
        role_layout = QHBoxLayout(role_group)
        
        # 角色创建/更新区域
        create_layout = QVBoxLayout()
        create_layout.addWidget(QLabel("角色名称:"))
        self.role_name_edit = QLineEdit()
        create_layout.addWidget(self.role_name_edit)
        
        create_layout.addWidget(QLabel("角色描述(可选):"))
        self.description_edit = QTextEdit()
        self.description_edit.setMaximumHeight(80)
        create_layout.addWidget(self.description_edit)
        
        self.create_btn = QPushButton("新建/更新角色")
        create_layout.addWidget(self.create_btn)
        role_layout.addLayout(create_layout)
        
        # 角色选择区域
        select_layout = QVBoxLayout()
        select_layout.addWidget(QLabel("现有角色列表:"))
        self.role_list_combo = QComboBox()
        self.role_list_combo.addItems(self.controller.get_roles())
        
        # 设置默认选中的角色为"凡子霞"
        default_role = "凡子霞"
        default_index = self.role_list_combo.findText(default_role)
        if default_index >= 0:
            self.role_list_combo.setCurrentIndex(default_index)
        select_layout.addWidget(self.role_list_combo)
        
        select_layout.addWidget(QLabel("情绪选择:"))
        self.emotion_list_combo = QComboBox()
        select_layout.addWidget(self.emotion_list_combo)
        
        # 角色操作按钮
        role_btn_layout = QHBoxLayout()
        self.load_role_btn = QPushButton("加载角色")
        self.delete_role_btn = QPushButton("删除角色")
        self.refresh_role_list_btn = QPushButton("刷新列表")
        role_btn_layout.addWidget(self.load_role_btn)
        role_btn_layout.addWidget(self.delete_role_btn)
        role_btn_layout.addWidget(self.refresh_role_list_btn)
        select_layout.addLayout(role_btn_layout)
        
        # 状态信息
        self.status_label = QLabel("")
        select_layout.addWidget(self.status_label)
        
        role_layout.addLayout(select_layout)
        role_management_scroll_layout.addWidget(role_group)
        
        # 添加滚动区域到各自的标签页
        listening_layout.addWidget(listening_scroll_area)
        role_management_layout.addWidget(role_management_scroll_area)
        
        # 添加标签页到标签页控件
        self.tab_widget.addTab(self.listening_tab, "试听")
        self.tab_widget.addTab(self.role_management_tab, "角色管理")
        
        # 添加标签页控件到主布局
        main_layout.addWidget(self.tab_widget)
        
        # 保存按钮引用
        self.refresh_models_btn = refresh_btn
        self.browse_btn = browse_btn
    
    def connect_signals(self):
        """连接信号"""
        # 刷新模型列表
        self.refresh_models_btn.clicked.connect(self.refresh_models)
        
        # 浏览参考音频
        self.browse_btn.clicked.connect(self.browse_ref_audio)
        
        # 参考音频路径变化时，尝试提取文本
        self.ref_audio_edit.textChanged.connect(self.extract_ref_text)
        
        # 滑块值变化时更新标签
        self.speed_slider.valueChanged.connect(self.update_speed_label)
        self.pause_slider.valueChanged.connect(self.update_pause_label)
        self.top_k_slider.valueChanged.connect(self.update_top_k_label)
        self.top_p_slider.valueChanged.connect(self.update_top_p_label)
        self.temperature_slider.valueChanged.connect(self.update_temperature_label)
        
        # 合成按钮
        self.synthesis_btn.clicked.connect(self.synthesize_speech)
        
        # 播放按钮
        self.play_btn.clicked.connect(self.play_audio)
        
        # 角色管理按钮
        self.create_btn.clicked.connect(self.create_or_update_role)
        self.load_role_btn.clicked.connect(self.load_role)
        self.delete_role_btn.clicked.connect(self.delete_role)
        self.refresh_role_list_btn.clicked.connect(self.refresh_role_list)
        
        # 角色选择变化时更新情绪列表
        self.role_list_combo.currentTextChanged.connect(self.update_emotion_list)
        
        # 初始化情绪列表
        self.update_emotion_list()
    
    def refresh_models(self):
        """刷新模型列表"""
        gpt_models, sovits_models = self.controller.get_model_lists()
        
        self.gpt_model_combo.clear()
        self.gpt_model_combo.addItems(gpt_models)
        
        self.sovits_model_combo.clear()
        self.sovits_model_combo.addItems(sovits_models)
        
        QMessageBox.information(self, "刷新成功", "模型列表已更新")
    
    def browse_ref_audio(self):
        """浏览选择参考音频文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择参考音频文件",
            "",
            "音频文件 (*.wav *.mp3 *.ogg);;所有文件 (*.*)"
        )
        
        if file_path:
            self.ref_audio_edit.setText(file_path)
    
    def extract_ref_text(self):
        """从文件名提取参考文本"""
        file_path = self.ref_audio_edit.text().strip()
        if file_path:
            text = self.controller.extract_text_from_filename(file_path)
            if text:
                self.prompt_text_edit.setText(text)
    
    def update_speed_label(self, value):
        """更新语速标签"""
        speed = value / 100.0
        self.speed_label.setText(f"{speed:.2f}")
    
    def update_pause_label(self, value):
        """更新停顿标签"""
        pause = value / 100.0
        self.pause_label.setText(f"{pause:.2f}")
    
    def update_top_k_label(self, value):
        """更新top_k标签"""
        self.top_k_label.setText(str(value))
    
    def update_top_p_label(self, value):
        """更新top_p标签"""
        top_p = value / 100.0
        self.top_p_label.setText(f"{top_p:.2f}")
    
    def update_temperature_label(self, value):
        """更新temperature标签"""
        temperature = value / 100.0
        self.temperature_label.setText(f"{temperature:.2f}")
    
    def synthesize_speech(self):
        """合成语音"""
        # 获取参数
        target_text = self.target_text_edit.toPlainText()
        gpt_model = self.gpt_model_combo.currentText()
        sovits_model = self.sovits_model_combo.currentText()
        ref_audio = self.ref_audio_edit.text()
        prompt_text = self.prompt_text_edit.toPlainText()
        prompt_lang = self.prompt_lang_combo.currentText()
        text_lang = self.text_lang_combo.currentText()
        speed = self.speed_slider.value() / 100.0
        ref_free = self.ref_free_check.isChecked()
        if_sr = self.if_sr_check.isChecked()
        top_k = self.top_k_slider.value()
        top_p = self.top_p_slider.value() / 100.0
        temperature = self.temperature_slider.value() / 100.0
        sample_steps = int(self.sample_steps_group.checkedButton().text())
        cut_punc = self.cut_punc_edit.text()
        role_name = self.role_name_edit.text()
        aux_refs = self.aux_refs_edit.toPlainText()
        emotion = self.emotion_list_combo.currentText() if self.emotion_list_combo.count() > 0 else ""
        pause_second = self.pause_slider.value() / 100.0
        
        if not target_text.strip():
            QMessageBox.warning(self, "错误", "请输入需要合成的文本")
            return
        
        if not ref_audio and not role_name:
            QMessageBox.warning(self, "错误", "请提供参考音频或选择角色")
            return
        
        # 创建进度对话框
        progress_dialog = QProgressDialog("正在合成语音，请稍候...", "取消", 0, 0, self)
        progress_dialog.setWindowTitle("处理中")
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setCancelButton(None)
        progress_dialog.setMinimumDuration(0)
        progress_dialog.setValue(0)
        progress_dialog.setMaximum(0)
        progress_dialog.show()
        
        # 创建合成线程
        self.synthesis_thread = SynthesisThread(
            self.controller,
            target_text,
            gpt_model,
            sovits_model,
            ref_audio,
            prompt_text,
            prompt_lang,
            text_lang,
            speed,
            ref_free,
            if_sr,
            top_k,
            top_p,
            temperature,
            sample_steps,
            cut_punc,
            role_name,
            aux_refs,
            emotion,
            pause_second
        )
        
        # 连接信号
        self.synthesis_thread.finished.connect(lambda output_path: self.synthesis_finished(output_path, progress_dialog))
        self.synthesis_thread.error.connect(lambda error_msg: self.synthesis_error(error_msg, progress_dialog))
        
        # 启动线程
        self.synthesis_thread.start()
    
    def synthesis_finished(self, output_path, progress_dialog):
        """合成完成"""
        progress_dialog.close()
        self.audio_output_path.setText(output_path)
        QMessageBox.information(self, "合成完成", f"音频已生成：{output_path}")
    
    def synthesis_error(self, error_msg, progress_dialog):
        """合成错误"""
        progress_dialog.close()
        QMessageBox.critical(self, "合成失败", error_msg)
    
    def play_audio(self):
        """播放音频"""
        audio_path = self.audio_output_path.text()
        if not audio_path or not os.path.exists(audio_path):
            QMessageBox.warning(self, "错误", "没有可播放的音频文件")
            return
        
        try:
            self.controller.play_audio(audio_path)
        except Exception as e:
            QMessageBox.critical(self, "播放失败", str(e))
    
    def create_or_update_role(self):
        """创建或更新角色"""
        role_name = self.role_name_edit.text().strip()
        if not role_name:
            QMessageBox.warning(self, "错误", "请输入角色名称")
            return
        
        # 获取角色配置参数
        gpt_model = self.gpt_model_combo.currentText()
        sovits_model = self.sovits_model_combo.currentText()
        ref_audio = self.ref_audio_edit.text()
        prompt_text = self.prompt_text_edit.toPlainText()
        prompt_lang = self.prompt_lang_combo.currentText()
        text_lang = self.text_lang_combo.currentText()
        speed = self.speed_slider.value() / 100.0
        ref_free = self.ref_free_check.isChecked()
        if_sr = self.if_sr_check.isChecked()
        top_k = self.top_k_slider.value()
        top_p = self.top_p_slider.value() / 100.0
        temperature = self.temperature_slider.value() / 100.0
        sample_steps = int(self.sample_steps_group.checkedButton().text())
        pause_second = self.pause_slider.value() / 100.0
        description = self.description_edit.toPlainText()
        aux_refs = self.aux_refs_edit.toPlainText()
        
        try:
            result = self.controller.save_role_config(
                role_name, gpt_model, sovits_model, ref_audio, prompt_text,
                prompt_lang, text_lang, speed, ref_free, if_sr, top_k, top_p,
                temperature, sample_steps, pause_second, description, aux_refs
            )
            
            self.status_label.setText(result)
            self.refresh_role_list()
            QMessageBox.information(self, "成功", result)
        except Exception as e:
            QMessageBox.critical(self, "保存失败", str(e))
    
    def load_role(self):
        """加载角色"""
        role = self.role_list_combo.currentText()
        emotion = self.emotion_list_combo.currentText() if self.emotion_list_combo.count() > 0 else ""
        
        if not role:
            QMessageBox.warning(self, "错误", "请选择角色")
            return
        
        try:
            # 加载配置
            result = self.controller.load_role_config(role, emotion)
            
            # 设置基本文本字段
            try:
                self.gpt_model_combo.setCurrentText(str(result.get("gpt_model", "")))
                self.sovits_model_combo.setCurrentText(str(result.get("sovits_model", "")))
                self.ref_audio_edit.setText(str(result.get("ref_audio", "")))
                self.prompt_text_edit.setText(str(result.get("prompt_text", "")))
                
                # 处理aux_refs
                aux_refs = result.get("aux_refs", "")
                self.aux_refs_edit.setText(str(aux_refs))
                
                # 设置语言下拉框
                self.prompt_lang_combo.setCurrentText(str(result.get("prompt_lang", "中文")))
                self.text_lang_combo.setCurrentText(str(result.get("text_lang", "中文")))
            except Exception as e:
                QMessageBox.warning(self, "警告", f"设置文本字段时出错: {str(e)}")
            
            # 设置滑块值
            try:
                self.speed_slider.setValue(int(float(result.get("speed", 1.0)) * 100))
                self.pause_slider.setValue(int(float(result.get("pause_second", 0.3)) * 100))
                self.top_k_slider.setValue(int(float(result.get("top_k", 15))))
                self.top_p_slider.setValue(int(float(result.get("top_p", 1.0)) * 100))
                self.temperature_slider.setValue(int(float(result.get("temperature", 1.0)) * 100))
            except Exception as e:
                QMessageBox.warning(self, "警告", f"设置滑块值时出错: {str(e)}，使用默认值")
                self.speed_slider.setValue(100)
                self.pause_slider.setValue(30)
                self.top_k_slider.setValue(15)
                self.top_p_slider.setValue(100)
                self.temperature_slider.setValue(100)
            
            # 设置复选框
            try:
                self.ref_free_check.setChecked(bool(result.get("ref_free", False)))
                self.if_sr_check.setChecked(bool(result.get("if_sr", False)))
            except Exception as e:
                QMessageBox.warning(self, "警告", f"设置复选框时出错: {str(e)}")
                self.ref_free_check.setChecked(False)
                self.if_sr_check.setChecked(False)
            
            # 设置采样步数
            try:
                steps = int(float(result.get("sample_steps", 32)))
                valid_steps = [4, 8, 16, 32, 64, 128]
                if steps not in valid_steps:
                    steps = min(valid_steps, key=lambda x: abs(x - steps))
                
                for button in self.sample_steps_group.buttons():
                    if int(button.text()) == steps:
                        button.setChecked(True)
                        break
            except Exception as e:
                QMessageBox.warning(self, "警告", f"设置采样步数时出错: {str(e)}")
                # 默认选择32
                for button in self.sample_steps_group.buttons():
                    if int(button.text()) == 32:
                        button.setChecked(True)
                        break
            
            # 设置描述
            try:
                self.description_edit.setText(str(result.get("description", "")))
            except Exception as e:
                QMessageBox.warning(self, "警告", f"设置描述时出错: {str(e)}")
            
            # 更新状态
            self.status_label.setText(f"已加载角色: {role}")
            QMessageBox.information(self, "加载成功", f"已加载角色: {role}")
        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"加载角色配置时发生错误: {str(e)}")
    
    def delete_role(self):
        """删除角色"""
        role = self.role_list_combo.currentText()
        if not role:
            QMessageBox.warning(self, "错误", "请选择要删除的角色")
            return
        
        # 确认删除
        if QMessageBox.question(self, "确认删除", f"确定要删除角色 {role} 吗？此操作不可恢复。", 
                               QMessageBox.Yes | QMessageBox.No) == QMessageBox.No:
            return
        
        try:
            result = self.controller.delete_role_config(role)
            self.status_label.setText(result)
            self.refresh_role_list()
            QMessageBox.information(self, "成功", result)
        except Exception as e:
            QMessageBox.critical(self, "删除失败", str(e))
    
    def refresh_role_list(self):
        """刷新角色列表"""
        roles = self.controller.get_roles()
        
        # 更新角色列表
        current_role = self.role_list_combo.currentText()
        self.role_list_combo.clear()
        self.role_list_combo.addItems(roles)
        
        # 尝试恢复之前选择的角色
        index = self.role_list_combo.findText(current_role)
        if index >= 0:
            self.role_list_combo.setCurrentIndex(index)
        
        # 更新情绪列表
        self.update_emotion_list()
    
    def update_emotion_list(self, role=None):
        """更新情绪列表"""
        if role is None:
            role = self.role_list_combo.currentText()
        
        if not role:
            self.emotion_list_combo.clear()
            return
        
        emotions = self.controller.get_emotions(role)
        
        self.emotion_list_combo.clear()
        self.emotion_list_combo.addItems(emotions)


class SynthesisThread(QThread):
    """合成语音线程"""
    finished = Signal(str)  # 完成信号，传递输出路径
    error = Signal(str)     # 错误信号，传递错误信息
    
    def __init__(self, controller, target_text, gpt_model, sovits_model, ref_audio, 
                 prompt_text, prompt_lang, text_lang, speed, ref_free, if_sr, 
                 top_k, top_p, temperature, sample_steps, cut_punc, role_name, 
                 aux_refs, emotion, pause_second):
        super().__init__()
        self.controller = controller
        self.target_text = target_text
        self.gpt_model = gpt_model
        self.sovits_model = sovits_model
        self.ref_audio = ref_audio
        self.prompt_text = prompt_text
        self.prompt_lang = prompt_lang
        self.text_lang = text_lang
        self.speed = speed
        self.ref_free = ref_free
        self.if_sr = if_sr
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.sample_steps = sample_steps
        self.cut_punc = cut_punc
        self.role_name = role_name
        self.aux_refs = aux_refs
        self.emotion = emotion
        self.pause_second = pause_second
    
    def run(self):
        """运行线程"""
        try:
            output_path = self.controller.test_role_synthesis(
                self.target_text, self.gpt_model, self.sovits_model, self.ref_audio,
                self.prompt_text, self.prompt_lang, self.text_lang, self.speed,
                self.ref_free, self.if_sr, self.top_k, self.top_p, self.temperature,
                self.sample_steps, self.cut_punc, self.role_name, self.aux_refs,
                self.emotion, self.pause_second
            )
            self.finished.emit(output_path)
        except Exception as e:
            self.error.emit(str(e))