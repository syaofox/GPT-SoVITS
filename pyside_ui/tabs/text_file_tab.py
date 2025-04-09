#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文本文件标签页UI
"""

import os
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QTextEdit, QComboBox, QRadioButton, QButtonGroup, QGroupBox,
    QFileDialog, QLineEdit, QProgressBar, QMessageBox, QSplitter,
    QProgressDialog
)
from PySide6.QtCore import Qt, QThread, Signal

from pyside_ui.controllers.text_file_controller import TextFileController


class TextFileTab(QWidget):
    def __init__(self):
        super().__init__()
        
        # 创建控制器
        self.controller = TextFileController()
        
        # 初始化UI
        self.init_ui()
        
        # 连接信号
        self.connect_signals()
    
    def init_ui(self):
        """初始化UI"""
        # 主布局
        main_layout = QVBoxLayout(self)
        
        # 处理模式选择区域
        mode_group = QGroupBox("选择处理模式")
        mode_layout = QHBoxLayout(mode_group)
        
        # 左侧：模式选择按钮组
        radio_layout = QVBoxLayout()
        self.mode_button_group = QButtonGroup(self)
        modes = ["逐行处理", "全文本处理", "分段处理"]
        for mode in modes:
            radio = QRadioButton(mode)
            self.mode_button_group.addButton(radio)
            radio_layout.addWidget(radio)
            if mode == "全文本处理":
                radio.setChecked(True)
        mode_layout.addLayout(radio_layout)
        
        # 右侧：模式说明文本
        mode_info = QLabel("""
        模式说明:
        - 逐行处理: 逐行分析角色和情绪标记，适用于对话场景，每行文本单独处理
        - 全文本处理: 使用默认角色和情绪处理整个文本，适用于大段独白
        - 分段处理: 将文本分成多个段落(不超过500字)，分别处理后合并，解决长文本问题
        """)
        mode_info.setWordWrap(True)
        mode_layout.addWidget(mode_info, 1)  # 设置拉伸因子为1，使说明文本占据更多空间
        
        main_layout.addWidget(mode_group)
        
        # 主要内容区域
        content_splitter = QSplitter(Qt.Horizontal)
        
        # 文本输入区域
        text_group = QGroupBox("文本输入")
        text_layout = QVBoxLayout(text_group)
        
        self.text_content = QTextEdit()
        self.text_content.setAcceptRichText(False)  # 禁止粘贴带格式文本
        self.text_content.setPlaceholderText("每行一句话，支持以下格式：\n(角色)文本内容\n(角色|情绪)文本内容\n直接输入文本")
        self.text_content.setFontPointSize(10)  # 设置字体大小为10
        self.text_content.setLineWrapMode(QTextEdit.WidgetWidth)  # 启用自动换行
        self.text_content.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 禁用水平滚动条
        text_layout.addWidget(self.text_content)
        
        content_splitter.addWidget(text_group)
        
        # 文件操作区域
        file_group = QGroupBox("文件操作")
        file_layout = QVBoxLayout(file_group)
        
        upload_btn = QPushButton("上传文本文件")
        file_layout.addWidget(upload_btn)
        
        preprocess_btn = QPushButton("预处理文本")
        file_layout.addWidget(preprocess_btn)
        
        file_layout.addWidget(QLabel("输出音频:"))
        self.audio_output_path = QLineEdit()
        self.audio_output_path.setReadOnly(True)
        file_layout.addWidget(self.audio_output_path)
        
        play_btn = QPushButton("播放音频")
        file_layout.addWidget(play_btn)
        
        content_splitter.addWidget(file_group)
        
        # 设置分割比例
        content_splitter.setSizes([800, 200])
        
        main_layout.addWidget(content_splitter)
        
        # 角色和情绪设置区域
        settings_group = QGroupBox("角色和情绪设置")
        settings_layout = QVBoxLayout(settings_group)
        
        # 创建一个包含三列的布局
        roles_layout = QHBoxLayout()
        
        # 第一列：默认角色和情绪
        default_group = QGroupBox("默认角色和情绪")
        default_layout = QVBoxLayout(default_group)
        
        default_layout.addWidget(QLabel("默认角色(必选):"))
        self.default_role_combo = QComboBox()
        self.default_role_combo.addItems(self.controller.get_roles())
        
        # 设置默认选中的角色为"凡子霞"
        default_role = "凡子霞"
        default_index = self.default_role_combo.findText(default_role)
        if default_index >= 0:
            self.default_role_combo.setCurrentIndex(default_index)
        default_layout.addWidget(self.default_role_combo)
        
        default_layout.addWidget(QLabel("默认情绪:"))
        self.default_emotion_combo = QComboBox()
        default_layout.addWidget(self.default_emotion_combo)
        
        roles_layout.addWidget(default_group, 4)  # 设置拉伸因子为4
        
        # 第二列：强制角色和情绪
        force_group = QGroupBox("强制角色和情绪")
        force_layout = QVBoxLayout(force_group)
        
        force_layout.addWidget(QLabel("强制使用角色(可选):"))
        self.force_role_combo = QComboBox()
        self.force_role_combo.addItem("无")
        self.force_role_combo.addItems(self.controller.get_roles())
        force_layout.addWidget(self.force_role_combo)
        
        force_layout.addWidget(QLabel("强制使用情绪(可选):"))
        self.force_emotion_combo = QComboBox()
        self.force_emotion_combo.addItem("无")
        force_layout.addWidget(self.force_emotion_combo)
        
        roles_layout.addWidget(force_group, 4)  # 设置拉伸因子为4
        
        # 第三列：语言和切分符号（占用更少空间）
        lang_group = QGroupBox("语言和切分设置")
        lang_layout = QVBoxLayout(lang_group)
        lang_layout.setContentsMargins(6, 6, 6, 6)  # 减小内边距
        
        # 使用更紧凑的布局
        lang_layout.addWidget(QLabel("文本语言:"))
        self.text_lang_combo = QComboBox()
        self.text_lang_combo.addItems([lang[0] for lang in self.controller.get_language_options()])
        self.text_lang_combo.setCurrentText("中文")
        lang_layout.addWidget(self.text_lang_combo)
        
        lang_layout.addWidget(QLabel("切分符号(可选):"))
        self.cut_punc_edit = QLineEdit("。！？：.!?:")
        lang_layout.addWidget(self.cut_punc_edit)
        
        # 设置较小的拉伸因子，使第三列占用更少空间
        roles_layout.addWidget(lang_group, 2)  # 设置拉伸因子为2，比其他两列小
        
        settings_layout.addLayout(roles_layout)
        main_layout.addWidget(settings_group)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        
        process_btn = QPushButton("处理文本")
        button_layout.addWidget(process_btn)
        
        refresh_roles_btn = QPushButton("刷新角色列表")
        button_layout.addWidget(refresh_roles_btn)
        
        main_layout.addLayout(button_layout)
        
        # 使用说明区域
        help_group = QGroupBox("使用说明")
        help_layout = QVBoxLayout(help_group)
        
        help_text = QLabel("""
        使用说明:
        1. 文本输入：可以选择上传文本文件或直接在文本框中输入
        2. 处理模式：选择合适的处理模式
           - 逐行处理：每行文本单独处理，支持行内角色和情绪标记
           - 全文本处理：使用默认角色和情绪处理整个文本
           - 分段处理：将文本分段处理后合并
        3. 角色设置：
           - 默认角色：当文本没有指定角色时使用的角色
           - 强制角色：忽略文本中的角色标记，全部使用指定角色
        4. 预处理文本：将双引号内的文本作为对白（使用默认情绪），其他文本作为叙述
        """)
        help_text.setWordWrap(True)
        help_layout.addWidget(help_text)
        
        main_layout.addWidget(help_group)
        
        # 保存按钮引用
        self.upload_btn = upload_btn
        self.preprocess_btn = preprocess_btn
        self.play_btn = play_btn
        self.process_btn = process_btn
        self.refresh_roles_btn = refresh_roles_btn
    
    def connect_signals(self):
        """连接信号"""
        # 上传文件按钮
        self.upload_btn.clicked.connect(self.upload_text_file)
        
        # 预处理文本按钮
        self.preprocess_btn.clicked.connect(self.preprocess_text)
        
        # 播放音频按钮
        self.play_btn.clicked.connect(self.play_audio)
        
        # 处理文本按钮
        self.process_btn.clicked.connect(self.process_text)
        
        # 刷新角色列表按钮
        self.refresh_roles_btn.clicked.connect(self.refresh_roles)
        
        # 角色选择变化
        self.default_role_combo.currentTextChanged.connect(self.update_default_emotions)
        self.force_role_combo.currentTextChanged.connect(self.update_force_emotions)
        
        # 初始化情绪列表
        self.update_default_emotions()
        self.update_force_emotions()
    
    def upload_text_file(self):
        """上传文本文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择文本文件",
            "",
            "文本文件 (*.txt);;所有文件 (*.*)"
        )
        
        if file_path:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                self.text_content.setPlainText(content)
            except Exception as e:
                QMessageBox.critical(self, "文件读取错误", str(e))
    
    def preprocess_text(self):
        """预处理文本"""
        content = self.text_content.toPlainText()
        default_role = self.default_role_combo.currentText()
        default_emotion = self.default_emotion_combo.currentText()
        
        try:
            processed_text = self.controller.preprocess_text(content, default_role, default_emotion)
            self.text_content.setPlainText(processed_text)
        except Exception as e:
            QMessageBox.critical(self, "预处理失败", str(e))
    
    def update_default_emotions(self, role=None):
        """更新默认情绪下拉列表"""
        if role is None:
            role = self.default_role_combo.currentText()
        
        emotions = self.controller.get_emotions(role)
        
        self.default_emotion_combo.clear()
        self.default_emotion_combo.addItems(emotions)
    
    def update_force_emotions(self, role=None):
        """更新强制情绪下拉列表"""
        if role is None:
            role = self.force_role_combo.currentText()
        
        self.force_emotion_combo.clear()
        self.force_emotion_combo.addItem("无")
        
        if role != "无":
            emotions = self.controller.get_emotions(role)
            self.force_emotion_combo.addItems(emotions)
    
    def refresh_roles(self):
        """刷新角色列表"""
        roles = self.controller.refresh_roles()
        
        # 更新默认角色下拉列表
        self.default_role_combo.clear()
        self.default_role_combo.addItems(roles)
        
        # 更新强制角色下拉列表
        self.force_role_combo.clear()
        self.force_role_combo.addItem("无")
        self.force_role_combo.addItems(roles)
        
        # 更新对应的情绪列表
        self.update_default_emotions()
        self.update_force_emotions()
    
    def process_text(self):
        """处理文本生成音频"""
        text_content = self.text_content.toPlainText()
        force_role = "" if self.force_role_combo.currentText() == "无" else self.force_role_combo.currentText()
        default_role = self.default_role_combo.currentText()
        force_emotion = "" if self.force_emotion_combo.currentText() == "无" else self.force_emotion_combo.currentText()
        default_emotion = self.default_emotion_combo.currentText()
        text_lang = self.text_lang_combo.currentText()
        cut_punc = self.cut_punc_edit.text()
        process_mode = self.mode_button_group.checkedButton().text()
        
        if not text_content.strip():
            QMessageBox.warning(self, "错误", "请输入文本内容")
            return
        
        if not default_role:
            QMessageBox.warning(self, "错误", "请选择默认角色")
            return
        
        # 创建进度对话框
        progress_dialog = QProgressDialog("正在处理文本生成音频，请稍候...", "取消", 0, 0, self)
        progress_dialog.setWindowTitle("处理中")
        progress_dialog.setWindowModality(Qt.WindowModal)  # 仅对当前窗口模态，不阻塞整个应用
        progress_dialog.setCancelButton(None)  # 移除取消按钮
        progress_dialog.setMinimumDuration(0)  # 立即显示
        progress_dialog.setValue(0)
        progress_dialog.setMaximum(0)  # 设置为0表示不确定进度（显示为循环动画）
        
        # 创建处理线程
        self.process_thread = ProcessThread(
            self.controller,
            text_content,
            force_role,
            default_role,
            force_emotion,
            default_emotion,
            text_lang,
            cut_punc,
            process_mode
        )
        
        # 连接信号
        self.process_thread.finished.connect(lambda output_path: self.process_finished(output_path, progress_dialog))
        self.process_thread.error.connect(lambda error_msg: self.process_error(error_msg, progress_dialog))
        
        # 启动线程
        self.process_thread.start()
    
    def process_finished(self, output_path, progress_dialog):
        """处理完成"""
        progress_dialog.close()
        self.audio_output_path.setText(output_path)
        QMessageBox.information(self, "处理完成", f"音频已生成：{output_path}")
    
    def process_error(self, error_msg, progress_dialog):
        """处理错误"""
        progress_dialog.close()
        QMessageBox.critical(self, "处理失败", error_msg)
    
    def play_audio(self):
        """播放生成的音频"""
        audio_path = self.audio_output_path.text()
        if not audio_path or not os.path.exists(audio_path):
            QMessageBox.warning(self, "错误", "没有可播放的音频文件")
            return
        
        try:
            self.controller.play_audio(audio_path)
        except Exception as e:
            QMessageBox.critical(self, "播放失败", str(e))


class ProcessThread(QThread):
    """处理文本的线程"""
    finished = Signal(str)  # 完成信号，传递输出路径
    error = Signal(str)     # 错误信号，传递错误信息
    
    def __init__(self, controller, text_content, force_role, default_role, 
                 force_emotion, default_emotion, text_lang, cut_punc, process_mode):
        super().__init__()
        self.controller = controller
        self.text_content = text_content
        self.force_role = force_role
        self.default_role = default_role
        self.force_emotion = force_emotion
        self.default_emotion = default_emotion
        self.text_lang = text_lang
        self.cut_punc = cut_punc
        self.process_mode = process_mode
    
    def run(self):
        """运行线程"""
        try:
            output_path = self.controller.process_text_content(
                self.text_content,
                self.force_role,
                self.default_role,
                self.force_emotion,
                self.default_emotion,
                self.text_lang,
                self.cut_punc,
                self.process_mode
            )
            self.finished.emit(output_path)
        except Exception as e:
            self.error.emit(str(e))