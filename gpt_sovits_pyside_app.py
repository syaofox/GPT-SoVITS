import sys
import os
import json
import re
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QComboBox, QPushButton, QListWidget, QGridLayout,
    QFileDialog, QMessageBox, QScrollArea, QFrame, QGroupBox, QFormLayout,
    QSlider, QCheckBox, QSpinBox, QDoubleSpinBox, QTextEdit, QSplitter, QInputDialog,
    QProgressBar, QListWidgetItem
)
from PySide6.QtCore import Qt, Signal, Slot, QSize, QTimer, QThread, QUrl
from PySide6.QtGui import QIcon, QFont, QPainter, QColor
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

import numpy as np
from datetime import datetime
import soundfile as sf
try:
    import librosa
except ImportError:
    print("警告: librosa 未安装，波形图显示功能将不可用")

# 导入gpt_sovits_lib库
from gpt_sovits_lib import GPTSoVITS, GPTSoVITSConfig

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtWidgets import QSizePolicy

class WaveformCanvas(FigureCanvas):
    """波形图画布类"""
    
    # 添加自定义信号用于点击跳转
    playback_position_changed = Signal(float)
    
    def __init__(self, parent=None, width=5, height=2, dpi=100):
        try:
            self.fig = plt.Figure(figsize=(width, height), dpi=dpi)
            self.axes = self.fig.add_subplot(111)
            self.axes.set_facecolor('#282828')
            self.fig.patch.set_facecolor('#282828')
            self.axes.set_ylim([-1.1, 1.1])
            self.axes.set_yticks([])
            self.axes.set_xticks([])
            
            FigureCanvas.__init__(self, self.fig)
            self.setParent(parent)
            
            FigureCanvas.setSizePolicy(self,
                                      QSizePolicy.Expanding,
                                      QSizePolicy.Expanding)
            FigureCanvas.updateGeometry(self)
            
            # 存储音频信息
            self.audio_data = None
            self.audio_sr = None
            self.audio_duration = 0
            
            # 添加鼠标点击事件
            self.mpl_connect('button_press_event', self.on_click)
            
            # 添加当前位置标记线
            self.position_line = None
            
        except Exception as e:
            print(f"初始化波形图失败: {str(e)}")
            # 创建一个替代的占位符
            self.setParent(parent)
            self.setMinimumHeight(80)
            self.setStyleSheet("background-color: #282828;")
    
    def on_click(self, event):
        """鼠标点击事件处理"""
        try:
            if event.xdata is not None and self.audio_duration > 0:
                # 计算点击位置对应的时间(秒)
                pos_ratio = event.xdata / self.audio_duration
                # 发出信号
                self.playback_position_changed.emit(pos_ratio)
                # 更新位置标记线
                self.update_position_line(event.xdata)
        except Exception as e:
            print(f"处理波形图点击事件出错: {str(e)}")
    
    def update_position_line(self, position):
        """更新位置标记线"""
        try:
            # 移除旧线
            if self.position_line:
                self.position_line.remove()
            
            # 添加新线
            self.position_line = self.axes.axvline(x=position, color='red', linewidth=1)
            self.draw()
        except Exception as e:
            print(f"更新位置标记线出错: {str(e)}")
    
    def set_playback_position(self, position_ms):
        """更新播放位置标记线（由播放器进度更新触发）"""
        try:
            if self.audio_duration > 0:
                position_sec = position_ms / 1000.0
                if 0 <= position_sec <= self.audio_duration:
                    self.update_position_line(position_sec)
        except Exception as e:
            print(f"设置播放位置出错: {str(e)}")
    
    def plot_waveform(self, audio_path):
        """绘制波形图"""
        try:
            if not os.path.exists(audio_path):
                return
            
            # 检查是否有librosa库，如果没有则返回
            if 'librosa' not in sys.modules:
                print("缺少librosa库，无法绘制波形图")
                return
                
            # 清除当前图形
            self.axes.clear()
            self.axes.set_facecolor('#282828')
            self.axes.set_ylim([-1.1, 1.1])
            self.axes.set_yticks([])
            self.axes.set_xticks([])
            
            # 加载音频
            self.audio_data, self.audio_sr = librosa.load(audio_path, sr=None)
            
            # 计算时间轴和音频时长
            self.audio_duration = len(self.audio_data) / self.audio_sr
            time = np.arange(0, len(self.audio_data)) / self.audio_sr
            
            # 绘制波形
            self.axes.plot(time, self.audio_data, color='#00BFFF', linewidth=0.5)
            
            # 添加时间轴标签（每隔几秒一个）
            if self.audio_duration > 0:
                num_ticks = min(5, int(self.audio_duration) + 1)
                tick_positions = np.linspace(0, self.audio_duration, num_ticks)
                tick_labels = [f"{int(t//60):02d}:{int(t%60):02d}" for t in tick_positions]
                self.axes.set_xticks(tick_positions)
                self.axes.set_xticklabels(tick_labels, fontsize=8, color='white')
            
            # 重置位置标记线
            self.position_line = None
            
            # 刷新图形
            self.fig.tight_layout()
            self.draw()
        except Exception as e:
            print(f"绘制波形图出错: {str(e)}")

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
    
    def on_emotion_selected(self, index):
        """音色选择事件处理"""
        if index < 0:
            return
        
        # 获取选中的音色名
        emotion_name = self.emotion_list.item(index).text()
        self.current_emotion = emotion_name
        
        # 加载音色配置
        emotions = self.current_config.get("emotions", {})
        emotion_config = emotions.get(emotion_name, {})
        
        # 更新参考音频信息
        ref_audio = emotion_config.get("ref_audio", "")
        self.ref_audio_label.setText(ref_audio)
        
        # 更新参考文本
        prompt_text = emotion_config.get("prompt_text", "")
        self.prompt_text.setText(prompt_text)
        
        # 加载辅助参考音频
        self.aux_ref_list.clear()
        aux_refs = emotion_config.get("aux_refs", [])
        for aux_ref in aux_refs:
            self.aux_ref_list.addItem(aux_ref)
        
        # 加载参数设置
        self.speed.setValue(self.current_config.get("speed", 1.0))
        self.ref_free.setChecked(self.current_config.get("ref_free", False))
        self.if_sr.setChecked(self.current_config.get("if_sr", False))
        self.top_k.setValue(self.current_config.get("top_k", 15))
        self.top_p.setValue(self.current_config.get("top_p", 1.0))
        self.temperature.setValue(self.current_config.get("temperature", 1.0))
        self.sample_steps.setValue(self.current_config.get("sample_steps", 32))
        self.pause_second.setValue(self.current_config.get("pause_second", 0.3))
        
        # 加载参考音频
        if ref_audio:
            ref_path = self.role_dir / self.current_role / ref_audio
            if ref_path.exists():
                # 设置参考音频路径
                self.ref_player.setSource(QUrl.fromLocalFile(str(ref_path)))
                # 绘制波形图
                self.ref_waveform.plot_waveform(str(ref_path))
    
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
        # 创建变量
        self.role_dir = Path("configs/roles")
        self.output_dir = Path("output")
        self.current_role = None
        self.current_emotion = None
        self.current_config = {}
        self.gpt_sovits = None
        self.ref_player = None
        self.result_player = None
        self.is_inferring = False
        
        # 初始化界面
        self.init_ui()
        
        # 加载角色列表
        self.load_roles()
        
        # 确保输出目录存在
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
    
    def init_ui(self):
        """初始化用户界面"""
        # 主布局 - 水平三列布局
        main_layout = QHBoxLayout(self)
        
        # ===== 左侧面板（固定宽度）=====
        left_panel = QVBoxLayout()
        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        left_widget.setFixedWidth(300)  # 设置固定宽度
        
        # 角色选择区域
        role_group = QGroupBox("角色选择")
        role_layout = QVBoxLayout()
        
        # 角色列表
        self.role_list = QListWidget()
        self.role_list.setMinimumWidth(150)
        self.role_list.currentItemChanged.connect(self.on_role_selected)
        
        # 音色列表
        emotion_label = QLabel("音色选择:")
        self.emotion_combo = QComboBox()
        self.emotion_combo.currentIndexChanged.connect(self.on_emotion_selected)
        
        # 添加到布局
        role_layout.addWidget(self.role_list)
        role_layout.addWidget(emotion_label)
        role_layout.addWidget(self.emotion_combo)
        role_group.setLayout(role_layout)
        
        # 参考音频区域
        ref_group = QGroupBox("参考音频")
        ref_layout = QVBoxLayout()
        
        # 参考音频信息
        ref_info_layout = QFormLayout()
        self.ref_audio_label = QLabel("")
        ref_info_layout.addRow("音频:", self.ref_audio_label)
        
        # 参考文本
        self.prompt_text = QTextEdit()
        self.prompt_text.setReadOnly(True)
        self.prompt_text.setMaximumHeight(60)
        ref_info_layout.addRow("参考文本:", self.prompt_text)
        
        # 波形图
        self.ref_waveform = WaveformCanvas(self, width=5, height=1.5, dpi=100)
        self.ref_waveform.setMinimumHeight(80)
        self.ref_waveform.playback_position_changed.connect(self.on_ref_waveform_clicked)
        
        # 播放按钮
        self.ref_play_btn = QPushButton("播放")
        self.ref_play_btn.clicked.connect(self.play_ref_audio)
        
        # 添加到布局
        ref_layout.addLayout(ref_info_layout)
        ref_layout.addWidget(self.ref_waveform)
        ref_layout.addWidget(self.ref_play_btn)
        ref_group.setLayout(ref_layout)
        
        # 辅助参考音频区域
        aux_group = QGroupBox("辅助参考音频")
        aux_layout = QVBoxLayout()
        
        # 辅助参考音频列表
        self.aux_ref_list = QListWidget()
        self.aux_ref_list.itemDoubleClicked.connect(self.play_aux_audio)
        
        # 添加到布局
        aux_layout.addWidget(self.aux_ref_list)
        aux_group.setLayout(aux_layout)
        
        # 参数区域
        params_group = QGroupBox("合成参数")
        params_layout = QFormLayout()
        
        # 语速
        self.speed = QDoubleSpinBox()
        self.speed.setRange(0.1, 5.0)
        self.speed.setSingleStep(0.1)
        self.speed.setValue(1.0)
        
        # 采样控制
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
        
        # 其他参数
        self.sample_steps = QSpinBox()
        self.sample_steps.setRange(1, 100)
        self.sample_steps.setValue(32)
        
        self.pause_second = QDoubleSpinBox()
        self.pause_second.setRange(0.1, 2.0)
        self.pause_second.setSingleStep(0.1)
        self.pause_second.setValue(0.3)
        
        # 切分标点
        self.cut_punc = QLineEdit(",.，。")
        
        # 选项
        options_layout = QVBoxLayout()
        self.ref_free = QCheckBox("无参考推理")
        self.if_sr = QCheckBox("使用超分辨率")
        options_layout.addWidget(self.ref_free)
        options_layout.addWidget(self.if_sr)
        
        # 添加到布局
        params_layout.addRow("语速:", self.speed)
        params_layout.addRow("Top K:", self.top_k)
        params_layout.addRow("Top P:", self.top_p)
        params_layout.addRow("Temperature:", self.temperature)
        params_layout.addRow("采样步数:", self.sample_steps)
        params_layout.addRow("停顿时长:", self.pause_second)
        params_layout.addRow("切分标点:", self.cut_punc)
        params_layout.addRow("选项:", options_layout)
        params_group.setLayout(params_layout)
        
        # 添加所有组件到左侧面板
        left_panel.addWidget(role_group)
        left_panel.addWidget(ref_group)
        left_panel.addWidget(aux_group)
        left_panel.addWidget(params_group)
        
        # ===== 中间面板（自适应宽度）=====
        middle_panel = QVBoxLayout()
        middle_widget = QWidget()
        middle_widget.setLayout(middle_panel)
        
        # 推理文本区域
        text_group = QGroupBox("待推理文本")
        text_layout = QVBoxLayout()
        
        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText("请在此输入需要转换为语音的文本...")
        
        # 添加到布局
        text_layout.addWidget(self.input_text)
        text_group.setLayout(text_layout)
        
        # 推理控制区域
        ctrl_layout = QHBoxLayout()
        
        self.infer_btn = QPushButton("开始推理")
        self.infer_btn.clicked.connect(self.start_inference)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        ctrl_layout.addWidget(self.infer_btn)
        ctrl_layout.addWidget(self.progress_bar)
        
        # 添加到中间面板
        middle_panel.addWidget(text_group)
        middle_panel.addLayout(ctrl_layout)
        
        # ===== 右侧面板（固定宽度）=====
        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)  # 减少组件间距
        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        right_widget.setFixedWidth(300)  # 设置固定宽度
        
        # 结果音频区域
        result_group = QGroupBox("结果音频")
        result_layout = QVBoxLayout()
        result_layout.setContentsMargins(5, 5, 5, 5)  # 减少内边距
        result_layout.setSpacing(5)  # 减少内部组件间距
        
        # 结果音频波形图
        self.result_waveform = WaveformCanvas(self, width=5, height=1.2, dpi=100)
        self.result_waveform.setMinimumHeight(80)  # 设置固定高度与参考音频一致
        self.result_waveform.setMaximumHeight(120)  # 设置最大高度限制
        self.result_waveform.playback_position_changed.connect(self.on_result_waveform_clicked)
        
        # 播放控制
        result_ctrl_layout = QHBoxLayout()
        self.result_play_btn = QPushButton("播放")
        self.result_play_btn.clicked.connect(self.play_result_audio)
        
        self.result_save_btn = QPushButton("另存为")
        self.result_save_btn.clicked.connect(self.save_result_audio)
        
        result_ctrl_layout.addWidget(self.result_play_btn)
        result_ctrl_layout.addWidget(self.result_save_btn)
        
        # 添加到布局
        result_layout.addWidget(self.result_waveform)
        result_layout.addLayout(result_ctrl_layout)
        result_group.setLayout(result_layout)
        result_group.setMaximumHeight(180)  # 限制结果音频区域的最大高度
        
        # 历史结果区域
        history_group = QGroupBox("历史结果")
        history_layout = QVBoxLayout()
        history_layout.setContentsMargins(5, 5, 5, 5)  # 减少内边距
        
        self.history_list = QListWidget()
        self.history_list.itemDoubleClicked.connect(self.on_history_item_clicked)
        
        history_btn_layout = QHBoxLayout()
        self.refresh_history_btn = QPushButton("刷新")
        self.refresh_history_btn.clicked.connect(self.load_history)
        self.clear_history_btn = QPushButton("清空")
        self.clear_history_btn.clicked.connect(self.clear_history)
        
        history_btn_layout.addWidget(self.refresh_history_btn)
        history_btn_layout.addWidget(self.clear_history_btn)
        
        history_layout.addWidget(self.history_list)
        history_layout.addLayout(history_btn_layout)
        history_group.setLayout(history_layout)
        
        # 添加所有组件到右侧面板，设置伸展因子，让历史结果区域占据更多空间
        right_panel.addWidget(result_group, 1)  # 结果音频区域占比较小
        right_panel.addWidget(history_group, 3)  # 历史结果区域占比较大
        
        # 添加三个面板到主布局
        main_layout.addWidget(left_widget)
        main_layout.addWidget(middle_widget, 1)  # 设置伸展因子为1，使中间面板自适应
        main_layout.addWidget(right_widget)
        
        # 初始化播放器
        self.init_player()
        
        # 禁用控件，直到选择角色
        self.set_inference_widgets_enabled(False)
        
        # 加载历史结果
        self.load_history()
    
    def init_player(self):
        """初始化音频播放器"""
        # 参考音频播放器
        self.ref_player = QMediaPlayer()
        self.ref_audio_output = QAudioOutput()
        self.ref_player.setAudioOutput(self.ref_audio_output)
        self.ref_player.playbackStateChanged.connect(self.on_ref_playback_state_changed)
        self.ref_player.positionChanged.connect(self.on_ref_position_changed)
        self.ref_player.durationChanged.connect(self.on_ref_duration_changed)
        
        # 结果音频播放器
        self.result_player = QMediaPlayer()
        self.result_audio_output = QAudioOutput()
        self.result_player.setAudioOutput(self.result_audio_output)
        self.result_player.playbackStateChanged.connect(self.on_result_playback_state_changed)
        self.result_player.positionChanged.connect(self.on_result_position_changed)
        self.result_player.durationChanged.connect(self.on_result_duration_changed)
    
    def on_ref_position_changed(self, position):
        """参考音频播放位置变化事件"""
        self.ref_waveform.set_playback_position(position)
    
    def on_ref_duration_changed(self, duration):
        """参考音频时长变化事件"""
        # 只是为了记录持续时间，不需要额外处理
        pass
    
    def on_result_position_changed(self, position):
        """结果音频播放位置变化事件"""
        self.result_waveform.set_playback_position(position)
    
    def on_result_duration_changed(self, duration):
        """结果音频时长变化事件"""
        # 只是为了记录持续时间，不需要额外处理
        pass
    
    def on_ref_waveform_clicked(self, position_ratio):
        """参考音频波形图点击事件"""
        if self.ref_player.duration() > 0:
            position = int(position_ratio * self.ref_player.duration())
            self.ref_player.setPosition(position)
            # 如果当前不在播放状态，点击后开始播放
            if self.ref_player.playbackState() != QMediaPlayer.PlayingState:
                self.ref_player.play()
    
    def on_result_waveform_clicked(self, position_ratio):
        """结果音频波形图点击事件"""
        if self.result_player.duration() > 0:
            position = int(position_ratio * self.result_player.duration())
            self.result_player.setPosition(position)
            # 如果当前不在播放状态，点击后开始播放
            if self.result_player.playbackState() != QMediaPlayer.PlayingState:
                self.result_player.play()
    
    def set_inference_widgets_enabled(self, enabled):
        """启用或禁用推理控件"""
        for widget in [
            self.emotion_combo, self.ref_play_btn, self.aux_ref_list,
            self.speed, self.top_k, self.top_p, self.temperature,
            self.sample_steps, self.pause_second, self.cut_punc,
            self.ref_free, self.if_sr, self.input_text, self.infer_btn,
            self.result_play_btn, self.result_save_btn
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
                config_path = role_path / "config.json"
                if config_path.exists():  # 只加载有配置文件的角色
                    self.role_list.addItem(role_path.name)
    
    def on_role_selected(self, current, previous):
        """角色选择事件处理"""
        if current is None:
            self.set_inference_widgets_enabled(False)
            self.current_role = None
            self.current_config = {}
            self.emotion_combo.clear()
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
                
                # 加载音色列表
                self.emotion_combo.clear()
                emotions = self.current_config.get("emotions", {})
                if emotions:
                    self.emotion_combo.addItems(emotions.keys())
                    self.set_inference_widgets_enabled(True)
                else:
                    QMessageBox.warning(self, "警告", f"角色 '{role_name}' 没有配置音色!")
                    self.set_inference_widgets_enabled(False)
            except Exception as e:
                QMessageBox.warning(self, "加载失败", f"无法加载角色配置: {str(e)}")
                self.set_inference_widgets_enabled(False)
        else:
            QMessageBox.warning(self, "错误", f"角色 '{role_name}' 配置文件不存在!")
            self.set_inference_widgets_enabled(False)
    
    def on_emotion_selected(self, index):
        """音色选择事件处理"""
        if index < 0:
            return
        
        # 获取选中的音色名
        emotion_name = self.emotion_combo.currentText()
        self.current_emotion = emotion_name
        
        # 加载音色配置
        emotions = self.current_config.get("emotions", {})
        emotion_config = emotions.get(emotion_name, {})
        
        # 更新参考音频信息
        ref_audio = emotion_config.get("ref_audio", "")
        self.ref_audio_label.setText(ref_audio)
        
        # 更新参考文本
        prompt_text = emotion_config.get("prompt_text", "")
        self.prompt_text.setText(prompt_text)
        
        # 加载辅助参考音频
        self.aux_ref_list.clear()
        aux_refs = emotion_config.get("aux_refs", [])
        for aux_ref in aux_refs:
            self.aux_ref_list.addItem(aux_ref)
        
        # 加载参数设置
        self.speed.setValue(self.current_config.get("speed", 1.0))
        self.ref_free.setChecked(self.current_config.get("ref_free", False))
        self.if_sr.setChecked(self.current_config.get("if_sr", False))
        self.top_k.setValue(self.current_config.get("top_k", 15))
        self.top_p.setValue(self.current_config.get("top_p", 1.0))
        self.temperature.setValue(self.current_config.get("temperature", 1.0))
        self.sample_steps.setValue(self.current_config.get("sample_steps", 32))
        self.pause_second.setValue(self.current_config.get("pause_second", 0.3))
        
        # 加载参考音频
        if ref_audio:
            ref_path = self.role_dir / self.current_role / ref_audio
            if ref_path.exists():
                # 设置参考音频路径
                self.ref_player.setSource(QUrl.fromLocalFile(str(ref_path)))
                # 绘制波形图
                self.ref_waveform.plot_waveform(str(ref_path))
    
    def play_ref_audio(self):
        """播放参考音频"""
        if self.ref_player.playbackState() == QMediaPlayer.PlayingState:
            self.ref_player.pause()
        else:
            self.ref_player.play()
    
    def play_aux_audio(self, item):
        """播放辅助参考音频"""
        audio_path = item.text()
        path = self.role_dir / self.current_role / audio_path
        if not path.exists() and Path(audio_path).exists():
            path = Path(audio_path)
            
        if path.exists():
            self.ref_player.setSource(QUrl.fromLocalFile(str(path)))
            self.ref_player.play()
            # 绘制波形图
            self.ref_waveform.plot_waveform(str(path))
        else:
            QMessageBox.warning(self, "错误", f"音频文件 '{audio_path}' 不存在!")
    
    def play_result_audio(self):
        """播放结果音频"""
        if self.result_player.playbackState() == QMediaPlayer.PlayingState:
            self.result_player.pause()
        else:
            self.result_player.play()
    
    def on_ref_playback_state_changed(self, state):
        """参考音频播放状态改变事件"""
        if state == QMediaPlayer.PlayingState:
            self.ref_play_btn.setText("暂停")
        else:
            self.ref_play_btn.setText("播放")
    
    def on_result_playback_state_changed(self, state):
        """结果音频播放状态改变事件"""
        if state == QMediaPlayer.PlayingState:
            self.result_play_btn.setText("暂停")
        else:
            self.result_play_btn.setText("播放")
    
    def save_result_audio(self):
        """另存为结果音频"""
        if not hasattr(self, 'last_result_path') or not self.last_result_path:
            QMessageBox.warning(self, "错误", "没有可保存的结果音频!")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存音频", "", "WAV文件 (*.wav)"
        )
        if file_path:
            try:
                import shutil
                shutil.copy2(self.last_result_path, file_path)
                QMessageBox.information(self, "保存成功", f"音频已保存到: {file_path}")
            except Exception as e:
                QMessageBox.warning(self, "保存失败", f"保存音频时出错: {str(e)}")
    
    def load_history(self):
        """加载历史结果"""
        self.history_list.clear()
        
        # 确保输出目录存在
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
            return
        
        # 加载所有WAV文件
        wav_files = list(self.output_dir.glob("*.wav"))
        wav_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        for wav_file in wav_files:
            item = QListWidgetItem(wav_file.name)
            item.setData(Qt.UserRole, str(wav_file))
            self.history_list.addItem(item)
    
    def clear_history(self):
        """清空历史结果"""
        reply = QMessageBox.question(
            self, "确认清空", 
            "确定要清空历史记录吗? 这将删除output目录中的所有WAV文件!",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                for wav_file in self.output_dir.glob("*.wav"):
                    wav_file.unlink()
                self.history_list.clear()
                QMessageBox.information(self, "清空成功", "历史记录已清空!")
            except Exception as e:
                QMessageBox.warning(self, "清空失败", f"清空历史记录时出错: {str(e)}")
    
    def on_history_item_clicked(self, item):
        """历史结果项点击事件"""
        file_path = item.data(Qt.UserRole)
        if file_path and Path(file_path).exists():
            # 设置音频路径
            self.result_player.setSource(QUrl.fromLocalFile(file_path))
            self.result_player.play()
            
            # 绘制波形图
            self.result_waveform.plot_waveform(file_path)
            
            # 保存当前结果路径
            self.last_result_path = file_path
    
    def start_inference(self):
        """开始推理"""
        if self.is_inferring:
            self.stop_inference()
            return
        
        # 检查输入
        if not self.current_role or not self.current_emotion:
            QMessageBox.warning(self, "错误", "请先选择角色和音色!")
            return
        
        text = self.input_text.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "错误", "请输入待推理文本!")
            return
        
        # 获取参数
        emotions = self.current_config.get("emotions", {})
        emotion_config = emotions.get(self.current_emotion, {})
        
        ref_audio = emotion_config.get("ref_audio", "")
        if not ref_audio and not self.ref_free.isChecked():
            QMessageBox.warning(self, "错误", "未设置参考音频且未启用无参考推理!")
            return
        
        prompt_text = emotion_config.get("prompt_text", "")
        if not prompt_text and not self.ref_free.isChecked():
            QMessageBox.warning(self, "错误", "未设置参考文本且未启用无参考推理!")
            return
        
        # 构建参数字典
        ref_audio_path = str(self.role_dir / self.current_role / ref_audio) if ref_audio else None
        
        # 辅助参考音频列表
        aux_refs = []
        for i in range(self.aux_ref_list.count()):
            aux_path = self.aux_ref_list.item(i).text()
            full_path = self.role_dir / self.current_role / aux_path
            if not full_path.exists() and Path(aux_path).exists():
                full_path = Path(aux_path)
            if full_path.exists():
                aux_refs.append(str(full_path))
        
        # 准备推理参数
        params = {
            "ref_wav_path": ref_audio_path,
            "prompt_text": prompt_text,
            "prompt_language": self.current_config.get("prompt_lang", "中文"),
            "text": text,
            "text_language": self.current_config.get("text_lang", "中文"),
            "top_k": self.top_k.value(),
            "top_p": self.top_p.value(),
            "temperature": self.temperature.value(),
            "speed": self.speed.value(),
            "inp_refs": aux_refs if aux_refs else None,
            "sample_steps": self.sample_steps.value(),
            "if_sr": self.if_sr.isChecked(),
            "pause_second": self.pause_second.value(),
            "cut_punc": self.cut_punc.text(),
            "audio_format": "wav",
            "bit_depth": "int16",
        }
        
        # 初始化推理线程
        self.inference_thread = InferenceThread(
            self.current_config.get("gpt_path", ""),
            self.current_config.get("sovits_path", ""),
            params
        )
        self.inference_thread.progress_update.connect(self.update_progress)
        self.inference_thread.inference_complete.connect(self.on_inference_complete)
        self.inference_thread.inference_error.connect(self.on_inference_error)
        
        # 更新UI状态
        self.is_inferring = True
        self.infer_btn.setText("取消")
        self.progress_bar.setValue(0)
        self.set_inference_widgets_enabled(False)
        self.infer_btn.setEnabled(True)
        
        # 开始推理
        self.inference_thread.start()
    
    def stop_inference(self):
        """停止推理"""
        if hasattr(self, 'inference_thread') and self.inference_thread.isRunning():
            self.inference_thread.terminate()
            self.inference_thread.wait()
        
        self.is_inferring = False
        self.infer_btn.setText("开始推理")
        self.progress_bar.setValue(0)
        self.set_inference_widgets_enabled(True)
    
    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)
    
    def on_inference_complete(self, output_path):
        """推理完成事件"""
        self.is_inferring = False
        self.infer_btn.setText("开始推理")
        self.progress_bar.setValue(100)
        self.set_inference_widgets_enabled(True)
        
        QMessageBox.information(self, "推理完成", f"音频已生成: {output_path}")
        
        # 设置音频路径
        self.result_player.setSource(QUrl.fromLocalFile(str(output_path)))
        self.result_player.play()
        
        # 绘制波形图
        self.result_waveform.plot_waveform(str(output_path))
        
        # 保存当前结果路径
        self.last_result_path = str(output_path)
        
        # 刷新历史记录
        self.load_history()
    
    def on_inference_error(self, error_msg):
        """推理错误事件"""
        self.is_inferring = False
        self.infer_btn.setText("开始推理")
        self.progress_bar.setValue(0)
        self.set_inference_widgets_enabled(True)
        
        QMessageBox.critical(self, "推理失败", error_msg)


class InferenceThread(QThread):
    """推理线程"""
    
    progress_update = Signal(int)
    inference_complete = Signal(str)
    inference_error = Signal(str)
    
    def __init__(self, gpt_path, sovits_path, params):
        super().__init__()
        self.gpt_path = gpt_path
        self.sovits_path = sovits_path
        self.params = params
    
    def run(self):
        """运行推理"""
        try:
            # 模拟进度更新
            self.progress_update.emit(10)
            
            # 初始化GPTSoVITS
            config = GPTSoVITSConfig()
            config.gpt_path = self.gpt_path
            config.sovits_path = self.sovits_path
            
            self.progress_update.emit(20)
            
            gpt_sovits = GPTSoVITS(config)
            gpt_sovits.load_models()
            
            self.progress_update.emit(30)
            
            # 生成输出文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            text_prefix = self.params["text"][:10].replace(' ', '_')
            output_path = Path("output") / f"{text_prefix}_{timestamp}.wav"
            
            self.progress_update.emit(40)
            
            # 修改参数并执行推理
            # 移除spk参数或检查其是否有效，避免传递无效的spk
            tts_params = self.params.copy()
            if "spk" in tts_params:
                # 如果需要，可以移除这个参数或提供默认值
                # del tts_params["spk"]  # 移除参数
                # 或者确保它不是None
                if tts_params["spk"] is None:
                    tts_params["spk"] = "default"
            
            # 执行推理，使用处理后的参数
            result_audio_bytes = gpt_sovits.tts(
                ref_wav_path=tts_params["ref_wav_path"],
                prompt_text=tts_params["prompt_text"],
                prompt_language=tts_params["prompt_language"],
                text=tts_params["text"],
                text_language=tts_params["text_language"],
                top_k=tts_params["top_k"],
                top_p=tts_params["top_p"],
                temperature=tts_params["temperature"],
                speed=tts_params["speed"],
                inp_refs=tts_params["inp_refs"],
                sample_steps=tts_params["sample_steps"],
                if_sr=tts_params["if_sr"],
                # spk参数可能是造成问题的原因，暂时注释掉
                # spk=tts_params["spk"],
                pause_second=tts_params["pause_second"],
                cut_punc=tts_params["cut_punc"],
                audio_format=tts_params["audio_format"],
                bit_depth=tts_params["bit_depth"],
            )
            
            self.progress_update.emit(80)
            
            # 保存音频字节流到WAV文件
            with open(str(output_path), 'wb') as f:
                f.write(result_audio_bytes)
            
            self.progress_update.emit(100)
            
            # 发送完成信号
            self.inference_complete.emit(str(output_path))
            
        except Exception as e:
            self.inference_error.emit(f"推理失败: {str(e)}")


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