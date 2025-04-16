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
        
        # 左侧角色列表区域
        role_group = QGroupBox("角色列表")
        role_layout = QVBoxLayout()
        
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
        
        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # 基本配置
        basic_group = QGroupBox("基本配置")
        basic_form = QFormLayout()
        
        # 模型选择
        self.gpt_path = QComboBox()
        self.gpt_path.setEditable(True)
        self.gpt_path_browse = QPushButton("浏览")
        self.gpt_path_browse.clicked.connect(self.on_browse_gpt_model)
        gpt_layout = QHBoxLayout()
        gpt_layout.addWidget(self.gpt_path)
        gpt_layout.addWidget(self.gpt_path_browse)
        
        self.sovits_path = QComboBox()
        self.sovits_path.setEditable(True)
        self.sovits_path_browse = QPushButton("浏览")
        self.sovits_path_browse.clicked.connect(self.on_browse_sovits_model)
        sovits_layout = QHBoxLayout()
        sovits_layout.addWidget(self.sovits_path)
        sovits_layout.addWidget(self.sovits_path_browse)
        
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
        ref_layout = QVBoxLayout()
        
        # 音色列表
        self.emotion_list = QListWidget()
        self.emotion_list.currentItemChanged.connect(self.on_emotion_selection_changed)
        
        # 音色操作按钮
        emotion_btn_layout = QHBoxLayout()
        self.add_emotion_btn = QPushButton("添加音色")
        self.del_emotion_btn = QPushButton("删除音色")
        self.add_emotion_btn.clicked.connect(self.on_add_emotion_clicked)
        self.del_emotion_btn.clicked.connect(self.on_del_emotion_clicked)
        
        emotion_btn_layout.addWidget(self.add_emotion_btn)
        emotion_btn_layout.addWidget(self.del_emotion_btn)
        
        # 音色编辑区
        emotion_edit_group = QGroupBox("音色编辑")
        emotion_edit_form = QFormLayout()
        
        self.emotion_name = QLineEdit()
        
        self.ref_audio = QComboBox()
        self.ref_audio.setEditable(True)
        self.ref_audio_browse = QPushButton("浏览")
        self.ref_audio_browse.clicked.connect(self.on_browse_ref_audio)
        ref_audio_layout = QHBoxLayout()
        ref_audio_layout.addWidget(self.ref_audio)
        ref_audio_layout.addWidget(self.ref_audio_browse)
        
        self.prompt_text = QTextEdit()
        self.prompt_text.setMaximumHeight(80)
        
        emotion_edit_form.addRow("音色名称:", self.emotion_name)
        emotion_edit_form.addRow("参考音频:", ref_audio_layout)
        emotion_edit_form.addRow("参考文本:", self.prompt_text)
        emotion_edit_group.setLayout(emotion_edit_form)
        
        # 辅助参考音频列表
        aux_group = QGroupBox("辅助参考音频")
        aux_layout = QVBoxLayout()
        
        self.aux_ref_list = QListWidget()
        
        aux_btn_layout = QHBoxLayout()
        self.add_aux_btn = QPushButton("添加")
        self.del_aux_btn = QPushButton("删除")
        self.add_aux_btn.clicked.connect(self.on_add_aux_ref_clicked)
        self.del_aux_btn.clicked.connect(self.on_del_aux_ref_clicked)
        
        aux_btn_layout.addWidget(self.add_aux_btn)
        aux_btn_layout.addWidget(self.del_aux_btn)
        
        aux_layout.addWidget(self.aux_ref_list)
        aux_layout.addLayout(aux_btn_layout)
        aux_group.setLayout(aux_layout)
        
        # 组合音色编辑区域
        ref_layout.addWidget(self.emotion_list)
        ref_layout.addLayout(emotion_btn_layout)
        ref_layout.addWidget(emotion_edit_group)
        ref_layout.addWidget(aux_group)
        ref_group.setLayout(ref_layout)
        
        # 参数配置
        params_group = QGroupBox("参数配置")
        params_form = QFormLayout()
        
        self.speed = QDoubleSpinBox()
        self.speed.setRange(0.1, 5.0)
        self.speed.setSingleStep(0.1)
        self.speed.setValue(1.0)
        
        self.ref_free = QCheckBox("无参考推理")
        
        self.if_sr = QCheckBox("使用超分辨率")
        
        self.top_k = QSpinBox()
        self.top_k.setRange(1, 100)
        self.top_k.setValue(15)
        
        self.top_p = QDoubleSpinBox()
        self.top_p.setRange(0.1, 1.0)
        self.top_p.setSingleStep(0.05)
        self.top_p.setValue(1.0)
        
        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0.1, 1.0)
        self.temperature.setSingleStep(0.05)
        self.temperature.setValue(1.0)
        
        self.sample_steps = QSpinBox()
        self.sample_steps.setRange(1, 100)
        self.sample_steps.setValue(32)
        
        self.pause_second = QDoubleSpinBox()
        self.pause_second.setRange(0.1, 2.0)
        self.pause_second.setSingleStep(0.1)
        self.pause_second.setValue(0.3)
        
        self.description = QTextEdit()
        self.description.setMaximumHeight(80)
        
        params_form.addRow("语速:", self.speed)
        params_form.addRow("", self.ref_free)
        params_form.addRow("", self.if_sr)
        params_form.addRow("Top K:", self.top_k)
        params_form.addRow("Top P:", self.top_p)
        params_form.addRow("Temperature:", self.temperature)
        params_form.addRow("采样步数:", self.sample_steps)
        params_form.addRow("停顿时长:", self.pause_second)
        params_form.addRow("角色描述:", self.description)
        params_group.setLayout(params_form)
        
        # 保存按钮
        self.save_btn = QPushButton("保存配置")
        self.save_btn.clicked.connect(self.on_save_config_clicked)
        
        # 添加所有组件到滚动区域
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
        QMessageBox.information(self, title, message, icon)
    
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
        emotion_name, ok = QInputDialog.getText(self, "添加音色", "请输入音色名称:")
        if ok and emotion_name:
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
        file_path, _ = QFileDialog.getOpenFileName(self, "选择辅助参考音频", "", "音频文件 (*.wav)")
        if file_path:
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
        file_path, _ = QFileDialog.getOpenFileName(self, "选择参考音频", "", "音频文件 (*.wav)")
        if file_path:
            self.ref_audio.setCurrentText(file_path)
    
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