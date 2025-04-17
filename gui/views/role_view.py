"""
角色配置视图
负责角色配置界面的显示
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QLabel, QLineEdit, QComboBox, QPushButton, QListWidget, 
    QTextEdit, QCheckBox, QSpinBox, QDoubleSpinBox, QScrollArea,
    QMessageBox, QInputDialog, QFileDialog
)
from PySide6.QtCore import Qt, Signal, Slot
import os

class RoleConfigView(QWidget):
    """角色配置视图类"""
    
    # 定义信号
    role_selected = Signal(str)  # 角色选择信号
    role_add = Signal(str)  # 添加角色信号
    role_delete = Signal(str)  # 删除角色信号
    role_import = Signal(str)  # 导入角色信号
    role_export = Signal(str, str)  # 导出角色信号
    emotion_selected = Signal(str)  # 音色选择信号
    emotion_add = Signal(str)  # 添加音色信号
    emotion_delete = Signal(str)  # 删除音色信号
    aux_ref_add = Signal(str)  # 添加辅助参考音频信号
    aux_ref_delete = Signal(str)  # 删除辅助参考音频信号
    config_save = Signal(dict)  # 保存配置信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.load_model_paths()
    
    def disable_wheel_event(self, widget):
        """禁用控件的滚轮事件"""
        widget.wheelEvent = lambda event: event.ignore()
    
    def init_ui(self):
        """初始化用户界面"""
        # 主布局
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # 左侧角色列表区域
        role_group = QGroupBox("角色列表")
        # 设置GroupBox的样式，减小内边距
        role_group.setStyleSheet("QGroupBox { padding-top: 15px; margin-top: 5px; }")
        role_layout = QVBoxLayout()
        role_layout.setSpacing(5)
        role_layout.setContentsMargins(5, 5, 5, 5)
        
        # 角色列表控件
        self.role_list = QListWidget()
        self.role_list.setMinimumWidth(200)
        self.role_list.currentItemChanged.connect(self.on_role_selection_changed)
        
        # 角色操作按钮
        role_btn_layout = QHBoxLayout()
        self.add_role_btn = QPushButton("新建")
        self.del_role_btn = QPushButton("删除")
        self.import_role_btn = QPushButton("导入")
        self.export_role_btn = QPushButton("导出")
        
        self.add_role_btn.clicked.connect(self.on_add_role_clicked)
        self.del_role_btn.clicked.connect(self.on_del_role_clicked)
        self.import_role_btn.clicked.connect(self.on_import_role_clicked)
        self.export_role_btn.clicked.connect(self.on_export_role_clicked)
        
        role_btn_layout.addWidget(self.add_role_btn)
        role_btn_layout.addWidget(self.del_role_btn)
        role_btn_layout.addWidget(self.import_role_btn)
        role_btn_layout.addWidget(self.export_role_btn)
        
        role_layout.addWidget(self.role_list)
        role_layout.addLayout(role_btn_layout)
        role_group.setLayout(role_layout)
        
        # 右侧配置区域
        config_layout = QVBoxLayout()
        # 设置右侧布局的间距更小，使界面更紧凑
        config_layout.setSpacing(5)
        config_layout.setContentsMargins(5, 5, 5, 5)
        
        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        # 减小滚动区域内布局的间距
        scroll_layout.setSpacing(5)
        scroll_layout.setContentsMargins(5, 5, 5, 5)
        
        # 基本配置
        basic_group = QGroupBox("基本配置")
        # 设置GroupBox的样式，减小内边距
        basic_group.setStyleSheet("QGroupBox { padding-top: 15px; margin-top: 5px; }")
        basic_form = QFormLayout()
        # 减小表单布局的间距
        basic_form.setSpacing(5)
        basic_form.setContentsMargins(5, 5, 5, 5)
        
        # 模型选择
        self.gpt_path = QComboBox()
        self.gpt_path.setEditable(True)
        # 使用更小的浏览按钮
        self.gpt_path_browse = QPushButton("浏览")
        self.gpt_path_browse.setMaximumWidth(50)  # 设置最大宽度
        self.gpt_path_browse.clicked.connect(self.on_browse_gpt_model)
        gpt_layout = QHBoxLayout()
        gpt_layout.setSpacing(3)  # 减小水平布局间距
        gpt_layout.addWidget(self.gpt_path, 4)  # 设置比例为4
        gpt_layout.addWidget(self.gpt_path_browse, 1)  # 设置比例为1
        
        self.sovits_path = QComboBox()
        self.sovits_path.setEditable(True)
        # 使用更小的浏览按钮
        self.sovits_path_browse = QPushButton("浏览")
        self.sovits_path_browse.setMaximumWidth(50)  # 设置最大宽度
        self.sovits_path_browse.clicked.connect(self.on_browse_sovits_model)
        sovits_layout = QHBoxLayout()
        sovits_layout.setSpacing(3)  # 减小水平布局间距
        sovits_layout.addWidget(self.sovits_path, 4)  # 设置比例为4
        sovits_layout.addWidget(self.sovits_path_browse, 1)  # 设置比例为1
        
        # 语言选择
        self.text_lang = QComboBox()
        self.text_lang.addItems(["中文", "英文", "日文", "中英混合", "日英混合", "多语种"])
        
        self.prompt_lang = QComboBox()
        self.prompt_lang.addItems(["中文", "英文", "日文", "中英混合", "日英混合", "多语种"])
        
        # 添加到表单布局
        basic_form.addRow("GPT 模型:", gpt_layout)
        basic_form.addRow("SoVITS 模型:", sovits_layout)
        basic_form.addRow("推理文本语言:", self.text_lang)
        basic_form.addRow("参考文本语言:", self.prompt_lang)
        basic_group.setLayout(basic_form)
        
        # 参考音频配置
        ref_group = QGroupBox("参考音频配置")
        # 设置GroupBox的样式，减小内边距
        ref_group.setStyleSheet("QGroupBox { padding-top: 15px; margin-top: 5px; }")
        ref_layout = QVBoxLayout()
        # 减小布局的间距
        ref_layout.setSpacing(5)
        ref_layout.setContentsMargins(5, 5, 5, 5)
        
        # 创建水平布局，左侧音色列表，右侧按钮
        emotion_header = QHBoxLayout()
        emotion_header.setSpacing(3)
        
        # 音色列表
        self.emotion_list = QListWidget()
        self.emotion_list.setMaximumHeight(100)  # 限制高度
        self.emotion_list.currentItemChanged.connect(self.on_emotion_selection_changed)
        
        # 音色操作按钮 - 垂直排列
        emotion_btn_layout = QVBoxLayout()
        emotion_btn_layout.setSpacing(3)
        self.add_emotion_btn = QPushButton("添加音色")
        self.add_emotion_btn.setMaximumWidth(80)
        self.del_emotion_btn = QPushButton("删除音色")
        self.del_emotion_btn.setMaximumWidth(80)
        self.add_emotion_btn.clicked.connect(self.on_add_emotion_clicked)
        self.del_emotion_btn.clicked.connect(self.on_del_emotion_clicked)
        
        emotion_btn_layout.addWidget(self.add_emotion_btn)
        emotion_btn_layout.addWidget(self.del_emotion_btn)
        emotion_btn_layout.addStretch()
        
        emotion_header.addWidget(self.emotion_list, 3)
        emotion_header.addLayout(emotion_btn_layout, 1)
        
        ref_layout.addLayout(emotion_header)
        
        # 音色编辑区
        emotion_edit_group = QGroupBox("音色编辑")
        # 设置GroupBox的样式，减小内边距
        emotion_edit_group.setStyleSheet("QGroupBox { padding-top: 15px; margin-top: 5px; }")
        emotion_edit_form = QFormLayout()
        # 减小表单布局的间距
        emotion_edit_form.setSpacing(5)
        emotion_edit_form.setContentsMargins(5, 5, 5, 5)
        
        self.emotion_name = QLineEdit()
        
        self.ref_audio = QComboBox()
        self.ref_audio.setEditable(True)
        # 使用更小的浏览按钮
        self.ref_audio_browse = QPushButton("浏览")
        self.ref_audio_browse.setMaximumWidth(50)  # 设置最大宽度
        self.ref_audio_browse.clicked.connect(self.on_browse_ref_audio)
        ref_audio_layout = QHBoxLayout()
        ref_audio_layout.setSpacing(3)  # 减小水平布局间距
        ref_audio_layout.addWidget(self.ref_audio, 4)  # 设置比例为4
        ref_audio_layout.addWidget(self.ref_audio_browse, 1)  # 设置比例为1
        
        self.prompt_text = QTextEdit()
        self.prompt_text.setMaximumHeight(60)
        
        emotion_edit_form.addRow("音色名称:", self.emotion_name)
        emotion_edit_form.addRow("参考音频:", ref_audio_layout)
        emotion_edit_form.addRow("参考文本:", self.prompt_text)
        emotion_edit_group.setLayout(emotion_edit_form)
        
        # 辅助参考音频列表 - 合并到水平布局中
        aux_header = QHBoxLayout()
        aux_header.setSpacing(3)
        
        self.aux_ref_list = QListWidget()
        self.aux_ref_list.setMaximumHeight(80)  # 限制高度
        
        # 辅助音频按钮 - 垂直排列
        aux_btn_layout = QVBoxLayout()
        aux_btn_layout.setSpacing(3)
        self.add_aux_btn = QPushButton("添加")
        self.add_aux_btn.setMaximumWidth(80)
        self.del_aux_btn = QPushButton("删除")
        self.del_aux_btn.setMaximumWidth(80)
        self.add_aux_btn.clicked.connect(self.on_add_aux_ref_clicked)
        self.del_aux_btn.clicked.connect(self.on_del_aux_ref_clicked)
        
        aux_btn_layout.addWidget(self.add_aux_btn)
        aux_btn_layout.addWidget(self.del_aux_btn)
        aux_btn_layout.addStretch()
        
        aux_header.addWidget(self.aux_ref_list, 3)
        aux_header.addLayout(aux_btn_layout, 1)
        
        ref_layout.addWidget(emotion_edit_group)
        ref_layout.addWidget(QLabel("辅助参考音频:"))
        ref_layout.addLayout(aux_header)
        ref_group.setLayout(ref_layout)
        
        # 参数配置
        params_group = QGroupBox("参数配置")
        # 设置GroupBox的样式，减小内边距
        params_group.setStyleSheet("QGroupBox { padding-top: 15px; margin-top: 5px; }")
        params_form = QFormLayout()
        # 减小参数配置表单的间距
        params_form.setSpacing(5)
        params_form.setContentsMargins(5, 5, 5, 5)
        
        # 将多个控件组合在同一行
        speed_container = QHBoxLayout()
        speed_container.setSpacing(3)
        
        self.speed = QDoubleSpinBox()
        self.speed.setRange(0.1, 5.0)
        self.speed.setSingleStep(0.1)
        self.speed.setValue(1.0)
        
        self.ref_free = QCheckBox("无参考推理")
        self.if_sr = QCheckBox("使用超分辨率")
        
        speed_container.addWidget(self.speed)
        speed_container.addWidget(self.ref_free)
        speed_container.addWidget(self.if_sr)
        speed_container.addStretch()
        
        # 将采样参数组合在同一行
        sampling_container = QHBoxLayout()
        sampling_container.setSpacing(10)
        
        top_k_container = QHBoxLayout()
        top_k_container.setSpacing(3)
        top_k_container.addWidget(QLabel("Top K:"))
        
        self.top_k = QSpinBox()
        self.top_k.setRange(1, 100)
        self.top_k.setValue(15)
        
        top_k_container.addWidget(self.top_k)
        
        top_p_container = QHBoxLayout()
        top_p_container.setSpacing(3)
        top_p_container.addWidget(QLabel("Top P:"))
        
        self.top_p = QDoubleSpinBox()
        self.top_p.setRange(0.1, 1.0)
        self.top_p.setSingleStep(0.05)
        self.top_p.setValue(1.0)
        
        top_p_container.addWidget(self.top_p)
        
        temp_container = QHBoxLayout()
        temp_container.setSpacing(3)
        temp_container.addWidget(QLabel("Temperature:"))
        
        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0.1, 1.0)
        self.temperature.setSingleStep(0.05)
        self.temperature.setValue(1.0)
        
        temp_container.addWidget(self.temperature)
        
        sampling_container.addLayout(top_k_container)
        sampling_container.addLayout(top_p_container)
        sampling_container.addLayout(temp_container)
        sampling_container.addStretch()
        
        # 将步数和停顿组合在同一行
        steps_container = QHBoxLayout()
        steps_container.setSpacing(10)
        
        sample_steps_container = QHBoxLayout()
        sample_steps_container.setSpacing(3)
        sample_steps_container.addWidget(QLabel("采样步数:"))
        
        self.sample_steps = QSpinBox()
        self.sample_steps.setRange(1, 100)
        self.sample_steps.setValue(32)
        
        sample_steps_container.addWidget(self.sample_steps)
        
        pause_container = QHBoxLayout()
        pause_container.setSpacing(3)
        pause_container.addWidget(QLabel("停顿时长:"))
        
        self.pause_second = QDoubleSpinBox()
        self.pause_second.setRange(0.1, 2.0)
        self.pause_second.setSingleStep(0.1)
        self.pause_second.setValue(0.3)
        
        pause_container.addWidget(self.pause_second)
        
        steps_container.addLayout(sample_steps_container)
        steps_container.addLayout(pause_container)
        steps_container.addStretch()
        
        self.description = QTextEdit()
        self.description.setMaximumHeight(60)
        
        # 添加到表单布局
        params_form.addRow("语速:", speed_container)
        params_form.addRow("采样参数:", sampling_container)
        params_form.addRow("步数设置:", steps_container)
        params_form.addRow("角色描述:", self.description)
        params_group.setLayout(params_form)
        
        # 保存按钮
        self.save_btn = QPushButton("保存配置")
        self.save_btn.clicked.connect(self.on_save_config_clicked)
        
        scroll_layout.addWidget(basic_group)
        scroll_layout.addWidget(ref_group)
        scroll_layout.addWidget(params_group)
        scroll_layout.addWidget(self.save_btn)
        scroll_layout.addStretch(1)
        
        scroll_area.setWidget(scroll_widget)
        config_layout.addWidget(scroll_area)
        
        # 添加到主布局
        main_layout.addWidget(role_group, 1)
        main_layout.addLayout(config_layout, 3)
        
        # 禁用右侧控件，直到选择角色
        self.set_config_widgets_enabled(False)
        
        # 禁用所有数值输入控件的滚轮事件
        self.disable_wheel_event(self.speed)
        self.disable_wheel_event(self.top_k)
        self.disable_wheel_event(self.top_p)
        self.disable_wheel_event(self.temperature)
        self.disable_wheel_event(self.sample_steps)
        self.disable_wheel_event(self.pause_second)
        
        # 禁用所有下拉框控件的滚轮事件
        self.disable_wheel_event(self.gpt_path)
        self.disable_wheel_event(self.sovits_path)
        self.disable_wheel_event(self.text_lang)
        self.disable_wheel_event(self.prompt_lang)
        self.disable_wheel_event(self.ref_audio)
        
        # 减小列表控件的行高
        self.role_list.setStyleSheet("QListWidget::item { height: 18px; }")
        self.emotion_list.setStyleSheet("QListWidget::item { height: 18px; }")
        self.aux_ref_list.setStyleSheet("QListWidget::item { height: 18px; }")
    
    def set_config_widgets_enabled(self, enabled):
        """启用或禁用配置控件"""
        for widget in [
            self.gpt_path, self.gpt_path_browse, 
            self.sovits_path, self.sovits_path_browse,
            self.text_lang, self.prompt_lang, 
            self.emotion_list, self.add_emotion_btn, self.del_emotion_btn,
            self.emotion_name, self.ref_audio, self.ref_audio_browse, self.prompt_text,
            self.aux_ref_list, self.add_aux_btn, self.del_aux_btn,
            self.speed, self.ref_free, self.if_sr, self.top_k, self.top_p,
            self.temperature, self.sample_steps, self.pause_second, self.description,
            self.save_btn
        ]:
            widget.setEnabled(enabled)
    
    def update_role_list(self, roles):
        """更新角色列表"""
        self.role_list.clear()
        for role in roles:
            self.role_list.addItem(role)
    
    def update_emotion_list(self, emotions):
        """更新音色列表"""
        self.emotion_list.clear()
        for emotion in emotions:
            self.emotion_list.addItem(emotion)
    
    def update_aux_ref_list(self, refs):
        """更新辅助参考音频列表"""
        self.aux_ref_list.clear()
        for ref in refs:
            self.aux_ref_list.addItem(ref)
    
    def show_message(self, title, message, icon=QMessageBox.Information):
        """显示消息对话框"""
        if icon == QMessageBox.Information:
            QMessageBox.information(self, title, message)
        elif icon == QMessageBox.Warning:
            QMessageBox.warning(self, title, message)
        elif icon == QMessageBox.Critical:
            QMessageBox.critical(self, title, message)
        else:
            QMessageBox.information(self, title, message)
    
    def load_config_to_ui(self, config):
        """加载配置到UI控件"""
        if not config:
            return
        
        # 基本配置
        self.gpt_path.setCurrentText(config.get("gpt_path", ""))
        self.sovits_path.setCurrentText(config.get("sovits_path", ""))
        
        # 设置语言选择
        text_lang = config.get("text_lang", "中文")
        prompt_lang = config.get("prompt_lang", "中文")
        
        index = self.text_lang.findText(text_lang)
        if index >= 0:
            self.text_lang.setCurrentIndex(index)
        
        index = self.prompt_lang.findText(prompt_lang)
        if index >= 0:
            self.prompt_lang.setCurrentIndex(index)
        
        # 更新音色列表
        self.update_emotion_list(config.get("emotions", {}).keys())
        
        # 参数配置
        self.speed.setValue(config.get("speed", 1.0))
        self.ref_free.setChecked(config.get("ref_free", False))
        self.if_sr.setChecked(config.get("if_sr", False))
        self.top_k.setValue(config.get("top_k", 15))
        self.top_p.setValue(config.get("top_p", 1.0))
        self.temperature.setValue(config.get("temperature", 1.0))
        self.sample_steps.setValue(config.get("sample_steps", 32))
        self.pause_second.setValue(config.get("pause_second", 0.3))
        self.description.setText(config.get("description", ""))
        
        # 清空音色编辑区
        self.emotion_name.clear()
        self.ref_audio.clear()
        self.prompt_text.clear()
        self.aux_ref_list.clear()
    
    def load_emotion_to_ui(self, emotion_name, emotion_config):
        """加载音色配置到UI控件"""
        if not emotion_config:
            return
        
        # 设置音色名称
        self.emotion_name.setText(emotion_name)
        
        # 设置参考音频
        self.ref_audio.setCurrentText(emotion_config.get("ref_audio", ""))
        
        # 设置参考文本
        self.prompt_text.setText(emotion_config.get("prompt_text", ""))
        
        # 更新辅助参考音频列表
        self.update_aux_ref_list(emotion_config.get("aux_refs", []))
    
    def get_current_config(self):
        """获取当前配置"""
        config = {
            "version": "v3",
            "gpt_path": self.gpt_path.currentText(),
            "sovits_path": self.sovits_path.currentText(),
            "text_lang": self.text_lang.currentText(),
            "prompt_lang": self.prompt_lang.currentText(),
            "speed": self.speed.value(),
            "ref_free": self.ref_free.isChecked(),
            "if_sr": self.if_sr.isChecked(),
            "top_k": self.top_k.value(),
            "top_p": self.top_p.value(),
            "temperature": self.temperature.value(),
            "sample_steps": self.sample_steps.value(),
            "pause_second": self.pause_second.value(),
            "description": self.description.toPlainText(),
            "emotions": {}  # 音色配置需要单独处理
        }
        return config
    
    def get_current_emotion_config(self):
        """获取当前音色配置"""
        emotion_config = {
            "ref_audio": self.ref_audio.currentText(),
            "prompt_text": self.prompt_text.toPlainText(),
            "aux_refs": []
        }
        
        # 获取辅助参考音频列表
        for i in range(self.aux_ref_list.count()):
            emotion_config["aux_refs"].append(self.aux_ref_list.item(i).text())
        
        return emotion_config
    
    # 事件处理方法
    def on_role_selection_changed(self, current, previous):
        """角色选择改变事件"""
        if current:
            self.role_selected.emit(current.text())
    
    def on_emotion_selection_changed(self, current, previous):
        """音色选择改变事件"""
        if current:
            self.emotion_selected.emit(current.text())
    
    def on_add_role_clicked(self):
        """添加角色按钮点击事件"""
        role_name, ok = QInputDialog.getText(self, "新建角色", "请输入角色名称:")
        if ok and role_name:
            self.role_add.emit(role_name)
    
    def on_del_role_clicked(self):
        """删除角色按钮点击事件"""
        current_item = self.role_list.currentItem()
        if not current_item:
            self.show_message("错误", "请先选择一个角色!", QMessageBox.Warning)
            return
        
        role_name = current_item.text()
        reply = QMessageBox.question(self, "确认删除", 
                                    f"确定要删除角色 '{role_name}' 吗?",
                                    QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.role_delete.emit(role_name)
    
    def on_import_role_clicked(self):
        """导入角色按钮点击事件"""
        file_path, _ = QFileDialog.getOpenFileName(self, "导入角色", "", "ZIP文件 (*.zip)")
        if file_path:
            self.role_import.emit(file_path)
    
    def on_export_role_clicked(self):
        """导出角色按钮点击事件"""
        current_item = self.role_list.currentItem()
        if not current_item:
            self.show_message("错误", "请先选择一个角色!", QMessageBox.Warning)
            return
        
        role_name = current_item.text()
        file_path, _ = QFileDialog.getSaveFileName(self, "导出角色", 
                                                f"{role_name}.zip", "ZIP文件 (*.zip)")
        if file_path:
            self.role_export.emit(role_name, file_path)
    
    def on_add_emotion_clicked(self):
        """添加音色按钮点击事件"""
        # 获取当前音色名称
        emotion_name = self.emotion_name.text().strip()
        if not emotion_name:
            self.show_message("错误", "请输入音色名称", QMessageBox.Warning)
            return
        
        # 检查GPT和SoVITS模型是否设置
        gpt_path = self.gpt_path.currentText().strip()
        sovits_path = self.sovits_path.currentText().strip()
        if not gpt_path:
            self.show_message("错误", "请选择GPT模型", QMessageBox.Warning)
            return
        if not sovits_path:
            self.show_message("错误", "请选择SoVITS模型", QMessageBox.Warning)
            return
        
        # 检查参考音频是否设置
        ref_audio = self.ref_audio.currentText().strip()
        if not ref_audio:
            reply = QMessageBox.question(
                self,
                "缺少参考音频",
                "未设置参考音频，是否先选择参考音频?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.on_browse_ref_audio()
                if not self.ref_audio.currentText().strip():
                    return
            elif not self.ref_free.isChecked():  # 非无参考模式需要参考音频
                self.show_message("错误", "未启用无参考模式，必须设置参考音频", QMessageBox.Warning)
                return
        
        # 检查参考文本是否设置
        prompt_text = self.prompt_text.toPlainText().strip()
        if not prompt_text and not self.ref_free.isChecked():
            reply = QMessageBox.question(
                self,
                "缺少参考文本",
                "未设置参考文本，是否继续?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
        
        # 检查该音色是否已存在
        for i in range(self.emotion_list.count()):
            if self.emotion_list.item(i).text() == emotion_name:
                # 如果已存在，询问是否覆盖
                reply = QMessageBox.question(
                    self, 
                    "音色已存在", 
                    f"音色'{emotion_name}'已存在，是否覆盖?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return
                break
        
        # 发送添加音色信号
        self.emotion_add.emit(emotion_name)
    
    def on_del_emotion_clicked(self):
        """删除音色按钮点击事件"""
        current_item = self.emotion_list.currentItem()
        if not current_item:
            self.show_message("错误", "请先选择一个音色!", QMessageBox.Warning)
            return
        
        emotion_name = current_item.text()
        reply = QMessageBox.question(self, "确认删除", 
                                   f"确定要删除音色 '{emotion_name}' 吗?",
                                   QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.emotion_delete.emit(emotion_name)
    
    def on_add_aux_ref_clicked(self):
        """添加辅助参考音频按钮点击事件"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择辅助参考音频", "", "音频文件 (*.wav *.mp3)")
        if file_path:
            # 检查是否已添加
            found = False
            for i in range(self.aux_ref_list.count()):
                if self.aux_ref_list.item(i).text() == file_path:
                    found = True
                    break
                    
            if not found:
                self.aux_ref_list.addItem(file_path)
                self.aux_ref_add.emit(file_path)
    
    def on_del_aux_ref_clicked(self):
        """删除辅助参考音频按钮点击事件"""
        current_item = self.aux_ref_list.currentItem()
        if not current_item:
            self.show_message("错误", "请先选择一个辅助参考音频!", QMessageBox.Warning)
            return
        
        ref_path = current_item.text()
        self.aux_ref_list.takeItem(self.aux_ref_list.row(current_item))
        self.aux_ref_delete.emit(ref_path)
    
    def on_save_config_clicked(self):
        """保存配置按钮点击事件"""
        config = self.get_current_config()
        self.config_save.emit(config)
    
    def on_browse_gpt_model(self):
        """浏览GPT模型按钮点击事件"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择GPT模型", "", "模型文件 (*.ckpt *.pth)")
        if file_path:
            self.gpt_path.setCurrentText(file_path)
    
    def on_browse_sovits_model(self):
        """浏览SoVITS模型按钮点击事件"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择SoVITS模型", "", "模型文件 (*.pth)")
        if file_path:
            self.sovits_path.setCurrentText(file_path)
    
    def on_browse_ref_audio(self):
        """浏览参考音频按钮点击事件"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择参考音频", "", "音频文件 (*.wav *.mp3)")
        if file_path:
            # 设置参考音频路径，但不复制文件（由控制器处理）
            self.ref_audio.setCurrentText(file_path)
            
            # 如果参考文本为空，尝试使用文件名作为参考文本
            if not self.prompt_text.toPlainText().strip():
                base_name = os.path.basename(file_path)
                name_without_ext = os.path.splitext(base_name)[0]
                self.prompt_text.setText(name_without_ext)
    
    def load_model_paths(self):
        """加载模型路径到下拉框"""
        # 加载GPT模型
        gpt_dirs = ["GPT_weights_v2", "GPT_weights_v3"]
        gpt_models = []
        
        # 查找项目根目录中的GPT模型文件
        for gpt_dir in gpt_dirs:
            if os.path.exists(gpt_dir):
                for file in os.listdir(gpt_dir):
                    if file.endswith(".ckpt") or file.endswith(".pth"):
                        gpt_models.append(os.path.join(gpt_dir, file))
        
        # 添加到GPT模型下拉框
        self.gpt_path.clear()
        self.gpt_path.addItems(gpt_models)
        
        # 加载SoVITS模型
        sovits_dirs = ["SoVITS_weights_v2", "SoVITS_weights_v3"]
        sovits_models = []
        
        # 查找项目根目录中的SoVITS模型文件
        for sovits_dir in sovits_dirs:
            if os.path.exists(sovits_dir):
                for file in os.listdir(sovits_dir):
                    if file.endswith(".pth"):
                        sovits_models.append(os.path.join(sovits_dir, file))
        
        # 添加到SoVITS模型下拉框
        self.sovits_path.clear()
        self.sovits_path.addItems(sovits_models) 