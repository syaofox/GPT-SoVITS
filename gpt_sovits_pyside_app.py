import os
import sys
import json
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QComboBox, QPushButton, QListWidget, QGridLayout,
    QFileDialog, QMessageBox, QScrollArea, QFrame, QGroupBox, QFormLayout,
    QSlider, QCheckBox, QSpinBox, QDoubleSpinBox, QTextEdit, QSplitter, QInputDialog
)
from PySide6.QtCore import Qt, Signal, Slot, QSize
from PySide6.QtGui import QIcon, QFont

# 导入gpt_sovits_lib库
from gpt_sovits_lib import GPTSoVITS, GPTSoVITSConfig

class RoleConfigTab(QWidget):
    """角色配置选项卡"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.role_dir = Path("configs/roles")
        self.current_role = None
        self.current_config = {}
        self.init_ui()
        self.load_roles()
    
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
        self.role_list.currentItemChanged.connect(self.on_role_selected)
        
        # 角色操作按钮
        role_btn_layout = QHBoxLayout()
        self.add_role_btn = QPushButton("新建")
        self.del_role_btn = QPushButton("删除")
        self.import_role_btn = QPushButton("导入")
        self.export_role_btn = QPushButton("导出")
        
        self.add_role_btn.clicked.connect(self.on_add_role)
        self.del_role_btn.clicked.connect(self.on_del_role)
        self.import_role_btn.clicked.connect(self.on_import_role)
        self.export_role_btn.clicked.connect(self.on_export_role)
        
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
        self.gpt_path_browse.clicked.connect(lambda: self.browse_file(self.gpt_path, "GPT 模型文件 (*.ckpt *.pth)"))
        gpt_layout = QHBoxLayout()
        gpt_layout.addWidget(self.gpt_path)
        gpt_layout.addWidget(self.gpt_path_browse)
        
        self.sovits_path = QComboBox()
        self.sovits_path.setEditable(True)
        self.sovits_path_browse = QPushButton("浏览")
        self.sovits_path_browse.clicked.connect(lambda: self.browse_file(self.sovits_path, "SoVITS 模型文件 (*.pth)"))
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
        self.emotion_list.currentItemChanged.connect(self.on_emotion_selected)
        
        # 音色操作按钮
        emotion_btn_layout = QHBoxLayout()
        self.add_emotion_btn = QPushButton("添加音色")
        self.del_emotion_btn = QPushButton("删除音色")
        self.add_emotion_btn.clicked.connect(self.on_add_emotion)
        self.del_emotion_btn.clicked.connect(self.on_del_emotion)
        
        emotion_btn_layout.addWidget(self.add_emotion_btn)
        emotion_btn_layout.addWidget(self.del_emotion_btn)
        
        # 音色编辑区
        emotion_edit_group = QGroupBox("音色编辑")
        emotion_edit_form = QFormLayout()
        
        self.emotion_name = QLineEdit()
        
        self.ref_audio = QComboBox()
        self.ref_audio.setEditable(True)
        self.ref_audio_browse = QPushButton("浏览")
        self.ref_audio_browse.clicked.connect(lambda: self.browse_file(self.ref_audio, "音频文件 (*.wav)"))
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
        self.add_aux_btn.clicked.connect(self.on_add_aux_ref)
        self.del_aux_btn.clicked.connect(self.on_del_aux_ref)
        
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
        self.save_btn.clicked.connect(self.save_current_config)
        
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
    
    def load_roles(self):
        """加载所有角色到列表"""
        self.role_list.clear()
        
        # 确保角色目录存在
        if not self.role_dir.exists():
            self.role_dir.mkdir(parents=True)
            return
        
        # 加载所有角色目录
        for role_path in self.role_dir.iterdir():
            if role_path.is_dir():
                self.role_list.addItem(role_path.name)
    
    def browse_file(self, target_widget, file_filter):
        """打开文件选择对话框"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", file_filter)
        if file_path:
            target_widget.setCurrentText(file_path)
    
    def on_role_selected(self, current, previous):
        """角色选择事件处理"""
        if current is None:
            self.set_config_widgets_enabled(False)
            self.current_role = None
            self.current_config = {}
            return
        
        # 获取选中的角色名
        role_name = current.text()
        self.current_role = role_name
        
        # 加载角色配置
        config_path = self.role_dir / role_name / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    self.current_config = json.load(f)
                self.load_config_to_ui()
                self.set_config_widgets_enabled(True)
            except Exception as e:
                QMessageBox.warning(self, "加载失败", f"无法加载角色配置: {str(e)}")
                self.set_config_widgets_enabled(False)
        else:
            QMessageBox.information(self, "新角色", "这是一个新角色，请配置并保存。")
            self.current_config = {
                "version": "v3",
                "emotions": {},
                "text_lang": "中文",
                "prompt_lang": "中文",
                "gpt_path": "",
                "sovits_path": "",
                "speed": 1.0,
                "ref_free": False,
                "if_sr": False,
                "top_k": 15,
                "top_p": 1.0,
                "temperature": 1.0,
                "sample_steps": 32,
                "pause_second": 0.3,
                "description": ""
            }
            self.load_config_to_ui()
            self.set_config_widgets_enabled(True)
    
    def load_config_to_ui(self):
        """将配置加载到UI控件"""
        # 基本配置
        self.gpt_path.setCurrentText(self.current_config.get("gpt_path", ""))
        self.sovits_path.setCurrentText(self.current_config.get("sovits_path", ""))
        
        # 设置语言选择
        text_lang = self.current_config.get("text_lang", "中文")
        prompt_lang = self.current_config.get("prompt_lang", "中文")
        
        index = self.text_lang.findText(text_lang)
        if index >= 0:
            self.text_lang.setCurrentIndex(index)
        
        index = self.prompt_lang.findText(prompt_lang)
        if index >= 0:
            self.prompt_lang.setCurrentIndex(index)
        
        # 加载音色列表
        self.emotion_list.clear()
        emotions = self.current_config.get("emotions", {})
        for emotion_name in emotions.keys():
            self.emotion_list.addItem(emotion_name)
        
        # 参数配置
        self.speed.setValue(self.current_config.get("speed", 1.0))
        self.ref_free.setChecked(self.current_config.get("ref_free", False))
        self.if_sr.setChecked(self.current_config.get("if_sr", False))
        self.top_k.setValue(self.current_config.get("top_k", 15))
        self.top_p.setValue(self.current_config.get("top_p", 1.0))
        self.temperature.setValue(self.current_config.get("temperature", 1.0))
        self.sample_steps.setValue(self.current_config.get("sample_steps", 32))
        self.pause_second.setValue(self.current_config.get("pause_second", 0.3))
        self.description.setText(self.current_config.get("description", ""))
        
        # 清空音色编辑区
        self.emotion_name.clear()
        self.ref_audio.clear()
        self.prompt_text.clear()
        self.aux_ref_list.clear()
    
    def on_emotion_selected(self, current, previous):
        """音色选择事件处理"""
        if current is None:
            self.emotion_name.clear()
            self.ref_audio.clear()
            self.prompt_text.clear()
            self.aux_ref_list.clear()
            return
        
        # 获取选中的音色名
        emotion_name = current.text()
        
        # 获取音色配置
        emotions = self.current_config.get("emotions", {})
        emotion_config = emotions.get(emotion_name, {})
        
        # 加载到UI
        self.emotion_name.setText(emotion_name)
        self.ref_audio.setCurrentText(emotion_config.get("ref_audio", ""))
        self.prompt_text.setText(emotion_config.get("prompt_text", ""))
        
        # 加载辅助参考音频
        self.aux_ref_list.clear()
        aux_refs = emotion_config.get("aux_refs", [])
        for aux_ref in aux_refs:
            self.aux_ref_list.addItem(aux_ref)
    
    def on_add_role(self):
        """添加新角色"""
        role_name, ok = QInputDialog.getText(self, "新建角色", "请输入角色名称:")
        if ok and role_name:
            # 检查角色是否已存在
            role_path = self.role_dir / role_name
            if role_path.exists():
                QMessageBox.warning(self, "错误", f"角色 '{role_name}' 已存在!")
                return
            
            # 创建角色目录
            role_path.mkdir(parents=True)
            
            # 添加到列表并选中
            self.role_list.addItem(role_name)
            for i in range(self.role_list.count()):
                if self.role_list.item(i).text() == role_name:
                    self.role_list.setCurrentRow(i)
                    break
    
    def on_del_role(self):
        """删除角色"""
        current_item = self.role_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "错误", "请先选择一个角色!")
            return
        
        role_name = current_item.text()
        reply = QMessageBox.question(self, "确认删除", 
                                    f"确定要删除角色 '{role_name}' 吗?",
                                    QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            # 这里我们不实际删除文件，而是将角色移动到 del_roles 目录
            del_dir = Path("configs/del_roles")
            if not del_dir.exists():
                del_dir.mkdir(parents=True)
            
            from shutil import move
            src_path = self.role_dir / role_name
            dst_path = del_dir / role_name
            
            try:
                # 如果目标存在，先删除
                if dst_path.exists():
                    import shutil
                    shutil.rmtree(dst_path)
                
                # 移动目录
                move(str(src_path), str(dst_path))
                
                # 从列表中移除
                row = self.role_list.row(current_item)
                self.role_list.takeItem(row)
                
                QMessageBox.information(self, "成功", f"角色 '{role_name}' 已移动到回收站!")
            except Exception as e:
                QMessageBox.warning(self, "错误", f"删除失败: {str(e)}")
    
    def on_import_role(self):
        """导入角色"""
        file_path, _ = QFileDialog.getOpenFileName(self, "导入角色", "", "ZIP文件 (*.zip)")
        if not file_path:
            return
        
        import zipfile
        import tempfile
        
        try:
            # 创建临时目录
            with tempfile.TemporaryDirectory() as temp_dir:
                # 解压文件
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # 查找config.json
                temp_path = Path(temp_dir)
                config_files = list(temp_path.glob("**/config.json"))
                
                if not config_files:
                    QMessageBox.warning(self, "导入失败", "ZIP文件中未找到config.json")
                    return
                
                # 获取角色名称
                config_path = config_files[0]
                role_name = config_path.parent.name
                
                # 检查角色是否已存在
                target_path = self.role_dir / role_name
                if target_path.exists():
                    reply = QMessageBox.question(self, "角色已存在", 
                                               f"角色 '{role_name}' 已存在，是否覆盖?",
                                               QMessageBox.Yes | QMessageBox.No)
                    if reply == QMessageBox.No:
                        return
                
                # 复制文件
                import shutil
                if target_path.exists():
                    shutil.rmtree(target_path)
                shutil.copytree(config_path.parent, target_path)
                
                # 刷新角色列表
                self.load_roles()
                
                # 选中导入的角色
                for i in range(self.role_list.count()):
                    if self.role_list.item(i).text() == role_name:
                        self.role_list.setCurrentRow(i)
                        break
                
                QMessageBox.information(self, "导入成功", f"角色 '{role_name}' 导入成功!")
        except Exception as e:
            QMessageBox.warning(self, "导入失败", f"导入角色时出错: {str(e)}")
    
    def on_export_role(self):
        """导出角色"""
        current_item = self.role_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "错误", "请先选择一个角色!")
            return
        
        role_name = current_item.text()
        file_path, _ = QFileDialog.getSaveFileName(self, "导出角色", 
                                                  f"{role_name}.zip", "ZIP文件 (*.zip)")
        if not file_path:
            return
        
        try:
            import zipfile
            import os
            
            role_path = self.role_dir / role_name
            
            with zipfile.ZipFile(file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(role_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, self.role_dir)
                        zipf.write(file_path, arcname)
            
            QMessageBox.information(self, "导出成功", f"角色 '{role_name}' 导出成功!")
        except Exception as e:
            QMessageBox.warning(self, "导出失败", f"导出角色时出错: {str(e)}")
    
    def on_add_emotion(self):
        """添加新音色"""
        if not self.current_role:
            QMessageBox.warning(self, "错误", "请先选择一个角色!")
            return
        
        emotion_name, ok = QInputDialog.getText(self, "添加音色", "请输入音色名称:")
        if ok and emotion_name:
            # 检查音色是否已存在
            emotions = self.current_config.get("emotions", {})
            if emotion_name in emotions:
                QMessageBox.warning(self, "错误", f"音色 '{emotion_name}' 已存在!")
                return
            
            # 添加新音色
            emotions[emotion_name] = {
                "ref_audio": "",
                "prompt_text": "",
                "aux_refs": []
            }
            self.current_config["emotions"] = emotions
            
            # 添加到列表并选中
            self.emotion_list.addItem(emotion_name)
            for i in range(self.emotion_list.count()):
                if self.emotion_list.item(i).text() == emotion_name:
                    self.emotion_list.setCurrentRow(i)
                    break
    
    def on_del_emotion(self):
        """删除音色"""
        current_item = self.emotion_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "错误", "请先选择一个音色!")
            return
        
        emotion_name = current_item.text()
        reply = QMessageBox.question(self, "确认删除", 
                                    f"确定要删除音色 '{emotion_name}' 吗?",
                                    QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            # 从配置中删除
            emotions = self.current_config.get("emotions", {})
            if emotion_name in emotions:
                del emotions[emotion_name]
                self.current_config["emotions"] = emotions
            
            # 从列表中移除
            row = self.emotion_list.row(current_item)
            self.emotion_list.takeItem(row)
            
            # 清空音色编辑区
            self.emotion_name.clear()
            self.ref_audio.clear()
            self.prompt_text.clear()
            self.aux_ref_list.clear()
    
    def on_add_aux_ref(self):
        """添加辅助参考音频"""
        current_item = self.emotion_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "错误", "请先选择一个音色!")
            return
        
        file_path, _ = QFileDialog.getOpenFileName(self, "选择辅助参考音频", "", "音频文件 (*.wav)")
        if not file_path:
            return
        
        # 添加到列表
        self.aux_ref_list.addItem(file_path)
        
        # 更新配置
        self.update_emotion_config()
    
    def on_del_aux_ref(self):
        """删除辅助参考音频"""
        current_item = self.aux_ref_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "错误", "请先选择一个辅助参考音频!")
            return
        
        # 从列表中移除
        row = self.aux_ref_list.row(current_item)
        self.aux_ref_list.takeItem(row)
        
        # 更新配置
        self.update_emotion_config()
    
    def update_emotion_config(self):
        """更新当前音色配置"""
        current_item = self.emotion_list.currentItem()
        if not current_item:
            return
        
        old_emotion_name = current_item.text()
        new_emotion_name = self.emotion_name.text()
        
        # 获取音色配置
        emotions = self.current_config.get("emotions", {})
        
        # 如果音色名称已更改，需要重命名
        if old_emotion_name != new_emotion_name and new_emotion_name:
            if new_emotion_name in emotions and old_emotion_name != new_emotion_name:
                QMessageBox.warning(self, "错误", f"音色 '{new_emotion_name}' 已存在!")
                self.emotion_name.setText(old_emotion_name)
                return
            
            # 复制配置到新名称，然后删除旧配置
            emotions[new_emotion_name] = emotions.get(old_emotion_name, {})
            if old_emotion_name in emotions:
                del emotions[old_emotion_name]
            
            # 更新列表项
            current_item.setText(new_emotion_name)
        
        # 更新音色配置
        emotion_config = {
            "ref_audio": self.ref_audio.currentText(),
            "prompt_text": self.prompt_text.toPlainText(),
            "aux_refs": []
        }
        
        # 添加辅助参考音频
        for i in range(self.aux_ref_list.count()):
            emotion_config["aux_refs"].append(self.aux_ref_list.item(i).text())
        
        emotions[new_emotion_name] = emotion_config
        self.current_config["emotions"] = emotions
    
    def save_current_config(self):
        """保存当前配置"""
        if not self.current_role:
            QMessageBox.warning(self, "错误", "请先选择一个角色!")
            return
        
        # 更新当前音色配置
        self.update_emotion_config()
        
        # 更新基本配置
        self.current_config["gpt_path"] = self.gpt_path.currentText()
        self.current_config["sovits_path"] = self.sovits_path.currentText()
        self.current_config["text_lang"] = self.text_lang.currentText()
        self.current_config["prompt_lang"] = self.prompt_lang.currentText()
        self.current_config["speed"] = self.speed.value()
        self.current_config["ref_free"] = self.ref_free.isChecked()
        self.current_config["if_sr"] = self.if_sr.isChecked()
        self.current_config["top_k"] = self.top_k.value()
        self.current_config["top_p"] = self.top_p.value()
        self.current_config["temperature"] = self.temperature.value()
        self.current_config["sample_steps"] = self.sample_steps.value()
        self.current_config["pause_second"] = self.pause_second.value()
        self.current_config["description"] = self.description.toPlainText()
        
        # 保存配置到文件
        role_path = self.role_dir / self.current_role
        if not role_path.exists():
            role_path.mkdir(parents=True)
        
        config_path = role_path / "config.json"
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(self.current_config, f, ensure_ascii=False, indent=4)
            QMessageBox.information(self, "保存成功", "角色配置已保存!")
        except Exception as e:
            QMessageBox.warning(self, "保存失败", f"保存配置时出错: {str(e)}")

class InferenceTab(QWidget):
    """推理选项卡"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # 占位，将在下一步实现
        label = QLabel("推理页面（待实现）")
        layout = QVBoxLayout(self)
        layout.addWidget(label)

class MainWindow(QMainWindow):
    """主窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GPT-SoVITS 语音合成")
        self.setMinimumSize(1024, 768)
        
        # 创建标签页
        self.tabs = QTabWidget()
        self.role_config_tab = RoleConfigTab()
        self.inference_tab = InferenceTab()
        
        self.tabs.addTab(self.role_config_tab, "角色配置")
        self.tabs.addTab(self.inference_tab, "音频推理")
        
        self.setCentralWidget(self.tabs)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 