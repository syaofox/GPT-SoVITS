"""
推理视图
负责语音合成界面的显示
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QLabel, QLineEdit, QComboBox, QPushButton, QListWidget, 
    QTextEdit, QCheckBox, QSpinBox, QDoubleSpinBox, QProgressBar,
    QMessageBox, QListWidgetItem, QFileDialog
)
from PySide6.QtCore import Qt, Signal, Slot, QUrl
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

from gui.views.common.waveform_canvas import WaveformCanvas

class InferenceView(QWidget):
    """推理视图类"""
    
    # 定义信号
    role_selected = Signal(str)  # 角色选择信号
    emotion_selected = Signal(str)  # 音色选择信号
    infer_start = Signal(dict)  # 开始推理信号
    infer_stop = Signal()  # 停止推理信号
    history_selected = Signal(str)  # 历史选择信号
    history_clear = Signal()  # 清空历史信号
    history_refresh = Signal()  # 刷新历史信号
    aux_ref_play = Signal(str)  # 播放辅助参考音频信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.init_player()
    
    def init_ui(self):
        """初始化用户界面"""
        # 主布局
        main_layout = QHBoxLayout(self)
        
        # 左侧面板
        left_panel = QVBoxLayout()
        
        # 角色选择区域
        role_group = QGroupBox("角色选择")
        role_layout = QVBoxLayout()
        
        # 角色列表
        self.role_list = QListWidget()
        self.role_list.setMinimumWidth(150)
        self.role_list.currentItemChanged.connect(self.on_role_selection_changed)
        
        # 音色列表
        emotion_label = QLabel("音色选择:")
        self.emotion_combo = QComboBox()
        self.emotion_combo.currentIndexChanged.connect(self.on_emotion_selection_changed)
        
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
        self.ref_play_btn.clicked.connect(self.on_ref_play_clicked)
        
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
        self.aux_ref_list.itemDoubleClicked.connect(self.on_aux_ref_double_clicked)
        
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
        self.infer_btn.clicked.connect(self.on_infer_clicked)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        ctrl_layout.addWidget(self.infer_btn)
        ctrl_layout.addWidget(self.progress_bar)
        
        # 添加所有组件到左侧面板
        left_panel.addWidget(role_group)
        left_panel.addWidget(ref_group)
        left_panel.addWidget(aux_group)
        left_panel.addWidget(params_group)
        left_panel.addWidget(text_group)
        left_panel.addLayout(ctrl_layout)
        
        # 右侧面板
        right_panel = QVBoxLayout()
        
        # 结果音频区域
        result_group = QGroupBox("结果音频")
        result_layout = QVBoxLayout()
        
        # 结果音频波形图
        self.result_waveform = WaveformCanvas(self, width=5, height=2, dpi=100)
        self.result_waveform.setMinimumHeight(120)
        self.result_waveform.playback_position_changed.connect(self.on_result_waveform_clicked)
        
        # 播放控制
        result_ctrl_layout = QHBoxLayout()
        self.result_play_btn = QPushButton("播放")
        self.result_play_btn.clicked.connect(self.on_result_play_clicked)
        
        self.result_save_btn = QPushButton("另存为")
        self.result_save_btn.clicked.connect(self.on_result_save_clicked)
        
        result_ctrl_layout.addWidget(self.result_play_btn)
        result_ctrl_layout.addWidget(self.result_save_btn)
        
        # 添加到布局
        result_layout.addWidget(self.result_waveform)
        result_layout.addLayout(result_ctrl_layout)
        result_group.setLayout(result_layout)
        
        # 历史结果区域
        history_group = QGroupBox("历史结果")
        history_layout = QVBoxLayout()
        
        self.history_list = QListWidget()
        self.history_list.itemDoubleClicked.connect(self.on_history_item_double_clicked)
        
        history_btn_layout = QHBoxLayout()
        self.refresh_history_btn = QPushButton("刷新")
        self.refresh_history_btn.clicked.connect(self.on_refresh_history_clicked)
        self.clear_history_btn = QPushButton("清空")
        self.clear_history_btn.clicked.connect(self.on_clear_history_clicked)
        
        history_btn_layout.addWidget(self.refresh_history_btn)
        history_btn_layout.addWidget(self.clear_history_btn)
        
        history_layout.addWidget(self.history_list)
        history_layout.addLayout(history_btn_layout)
        history_group.setLayout(history_layout)
        
        # 添加所有组件到右侧面板
        right_panel.addWidget(result_group)
        right_panel.addWidget(history_group)
        
        # 添加左右面板到主布局
        main_layout.addLayout(left_panel, 3)
        main_layout.addLayout(right_panel, 2)
        
        # 禁用控件，直到选择角色
        self.set_inference_widgets_enabled(False)
    
    def init_player(self):
        """初始化音频播放器"""
        # 参考音频播放器
        self.ref_player = QMediaPlayer()
        self.ref_audio_output = QAudioOutput()
        self.ref_player.setAudioOutput(self.ref_audio_output)
        self.ref_player.playbackStateChanged.connect(self.on_ref_playback_state_changed)
        self.ref_player.positionChanged.connect(self.on_ref_position_changed)
        
        # 结果音频播放器
        self.result_player = QMediaPlayer()
        self.result_audio_output = QAudioOutput()
        self.result_player.setAudioOutput(self.result_audio_output)
        self.result_player.playbackStateChanged.connect(self.on_result_playback_state_changed)
        self.result_player.positionChanged.connect(self.on_result_position_changed)
    
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
    
    def update_role_list(self, roles):
        """更新角色列表"""
        self.role_list.clear()
        for role in roles:
            self.role_list.addItem(role)
    
    def update_emotion_list(self, emotions):
        """更新音色列表"""
        self.emotion_combo.clear()
        for emotion in emotions:
            self.emotion_combo.addItem(emotion)
    
    def update_aux_ref_list(self, refs):
        """更新辅助参考音频列表"""
        self.aux_ref_list.clear()
        for ref in refs:
            self.aux_ref_list.addItem(ref)
    
    def update_history_list(self, history_items):
        """更新历史列表"""
        self.history_list.clear()
        for item in history_items:
            list_item = QListWidgetItem(item["name"])
            list_item.setData(Qt.UserRole, item["path"])
            self.history_list.addItem(list_item)
    
    def show_message(self, title, message, icon=QMessageBox.Information):
        """显示消息对话框"""
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setIcon(icon)
        msg_box.exec()
    
    def load_reference_info(self, ref_audio, prompt_text):
        """加载参考音频信息"""
        self.ref_audio_label.setText(ref_audio if ref_audio else "")
        self.prompt_text.setText(prompt_text if prompt_text else "")
    
    def load_parameters(self, params):
        """加载参数设置"""
        self.speed.setValue(params.get("speed", 1.0))
        self.top_k.setValue(params.get("top_k", 15))
        self.top_p.setValue(params.get("top_p", 1.0))
        self.temperature.setValue(params.get("temperature", 1.0))
        self.sample_steps.setValue(params.get("sample_steps", 32))
        self.pause_second.setValue(params.get("pause_second", 0.3))
        self.ref_free.setChecked(params.get("ref_free", False))
        self.if_sr.setChecked(params.get("if_sr", False))
    
    def set_ref_audio(self, audio_path):
        """设置参考音频"""
        if audio_path:
            self.ref_player.setSource(QUrl.fromLocalFile(audio_path))
            self.ref_waveform.plot_waveform(audio_path)
    
    def set_result_audio(self, audio_path):
        """设置结果音频"""
        if audio_path:
            self.result_player.setSource(QUrl.fromLocalFile(audio_path))
            self.result_waveform.plot_waveform(audio_path)
            self.last_result_path = audio_path
    
    def get_inference_params(self):
        """获取推理参数"""
        params = {
            "speed": self.speed.value(),
            "top_k": self.top_k.value(),
            "top_p": self.top_p.value(),
            "temperature": self.temperature.value(),
            "sample_steps": self.sample_steps.value(),
            "pause_second": self.pause_second.value(),
            "cut_punc": self.cut_punc.text(),
            "ref_free": self.ref_free.isChecked(),
            "if_sr": self.if_sr.isChecked(),
            "text": self.input_text.toPlainText()
        }
        return params
    
    def set_progress(self, value):
        """设置进度条值"""
        self.progress_bar.setValue(value)
    
    def set_inferring_state(self, is_inferring):
        """设置推理状态"""
        self.is_inferring = is_inferring
        self.infer_btn.setText("取消" if is_inferring else "开始推理")
        self.set_inference_widgets_enabled(not is_inferring)
        self.infer_btn.setEnabled(True)
    
    # 事件处理方法
    def on_role_selection_changed(self, current, previous):
        """角色选择改变事件"""
        if current:
            self.role_selected.emit(current.text())
    
    def on_emotion_selection_changed(self, index):
        """音色选择改变事件"""
        if index >= 0:
            self.emotion_selected.emit(self.emotion_combo.currentText())
    
    def on_ref_play_clicked(self):
        """参考音频播放按钮点击事件"""
        if self.ref_player.playbackState() == QMediaPlayer.PlayingState:
            self.ref_player.pause()
        else:
            self.ref_player.play()
    
    def on_aux_ref_double_clicked(self, item):
        """辅助参考音频双击事件"""
        ref_path = item.text()
        self.aux_ref_play.emit(ref_path)
    
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
    
    def on_result_play_clicked(self):
        """结果音频播放按钮点击事件"""
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
    
    def on_ref_position_changed(self, position):
        """参考音频播放位置改变事件"""
        self.ref_waveform.set_playback_position(position)
    
    def on_result_position_changed(self, position):
        """结果音频播放位置改变事件"""
        self.result_waveform.set_playback_position(position)
    
    def on_result_save_clicked(self):
        """结果音频保存按钮点击事件"""
        if not hasattr(self, 'last_result_path') or not self.last_result_path:
            self.show_message("错误", "没有可保存的结果音频!", QMessageBox.Warning)
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存音频", "", "WAV文件 (*.wav)"
        )
        if file_path:
            try:
                import shutil
                shutil.copy2(self.last_result_path, file_path)
                self.show_message("保存成功", f"音频已保存到: {file_path}")
            except Exception as e:
                self.show_message("保存失败", f"保存音频时出错: {str(e)}", QMessageBox.Warning)
    
    def on_history_item_double_clicked(self, item):
        """历史项目双击事件"""
        file_path = item.data(Qt.UserRole)
        if file_path:
            # 向控制器发送信号
            self.history_selected.emit(file_path)
            
            # 立即设置并播放音频
            self.set_result_audio(file_path)
            self.result_player.play()
    
    def on_refresh_history_clicked(self):
        """刷新历史按钮点击事件"""
        self.history_refresh.emit()
    
    def on_clear_history_clicked(self):
        """清空历史按钮点击事件"""
        reply = QMessageBox.question(
            self, "确认清空", 
            "确定要清空历史记录吗? 这将删除output目录中的所有WAV文件!",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.history_clear.emit()
    
    def on_infer_clicked(self):
        """推理按钮点击事件"""
        if hasattr(self, 'is_inferring') and self.is_inferring:
            self.infer_stop.emit()
        else:
            # 检查输入
            text = self.input_text.toPlainText().strip()
            if not text:
                self.show_message("错误", "请输入待推理文本!", QMessageBox.Warning)
                return
            
            # 获取参数
            params = self.get_inference_params()
            self.infer_start.emit(params) 