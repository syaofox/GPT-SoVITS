"""
推理视图
负责语音合成界面的显示
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QLabel, QLineEdit, QComboBox, QPushButton, QListWidget, 
    QTextEdit, QCheckBox, QSpinBox, QDoubleSpinBox, QProgressBar,
    QMessageBox, QListWidgetItem, QFileDialog, QSplitter, QSizePolicy,
    QSlider
)
from PySide6.QtCore import Qt, Signal, Slot, QUrl, QEvent
from PySide6.QtGui import QFont, QTextOption
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput, QMediaDevices, QAudioDevice

from gui.views.common.waveform_canvas import WaveformCanvas

# 添加自定义的纯文本编辑框类
class PlainTextEdit(QTextEdit):
    """纯文本编辑框，禁用富文本功能并自动换行"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # 设置为纯文本模式
        self.setAcceptRichText(False)
        # 设置自动换行模式
        self.setLineWrapMode(QTextEdit.WidgetWidth)
        # 设置字体
        self.setFont(QFont("SimSun", 10))
        # 设置样式，禁用横向滚动条
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # 设置文档默认格式，启用自动换行
        text_option = QTextOption()
        text_option.setWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.document().setDefaultTextOption(text_option)
        # 禁用文本阴影效果
        self.setStyleSheet("QTextEdit { background-color: white; color: black; }")
    
    def insertFromMimeData(self, source):
        """重写此方法以确保粘贴的是纯文本"""
        if source.hasText():
            # 清除缓冲区中可能存在的任何残留文本格式
            self.document().clear()
            # 只插入纯文本内容
            self.insertPlainText(source.text())
        # 不调用父类方法，避免富文本格式插入
    
    def keyPressEvent(self, event):
        """重写按键事件，确保文本输入正确渲染"""
        # 使用原生文本输入处理
        super().keyPressEvent(event)
        # 如果需要额外的处理，可以在这里添加
        
    def paintEvent(self, event):
        """重写绘制事件，确保渲染清晰无残影"""
        # 先清除当前视图
        self.viewport().update()
        # 调用父类方法进行正常绘制
        super().paintEvent(event)

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
    role_refresh = Signal()  # 刷新角色列表信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.init_player()
        # 设置最小宽度，使整体界面宽度增加30%
        self.setMinimumWidth(1200)  # 设置一个合理的宽度值，确保推理文本区域有足够空间
    
    def init_ui(self):
        """初始化用户界面"""
        # 设置扁平化按钮样式
        self.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #c0c0c0;
                border-radius: 3px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
            QPushButton:pressed {
                background-color: #d0d0d0;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #e0e0e0;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #4a86e8;
                border: none;
                width: 14px;
                height: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            QSlider::sub-page:horizontal {
                background: #4a86e8;
                border-radius: 3px;
            }
        """)
        
        # 主布局 - 四栏水平布局
        main_layout = QHBoxLayout(self)
        
        # 第1块 - 角色选择列表 (固定宽度)
        panel1 = QVBoxLayout()
        panel1_widget = QWidget()
        panel1_widget.setLayout(panel1)
        panel1_widget.setFixedWidth(200)  # 固定宽度
        
        # 设置第1块在垂直方向上的大小策略为自适应
        size_policy = panel1_widget.sizePolicy()
        size_policy.setVerticalPolicy(QSizePolicy.Expanding)
        panel1_widget.setSizePolicy(size_policy)
        
        # 第2块 - 参考音频、辅助参考音频、合成参数 (固定宽度)
        panel2 = QVBoxLayout()
        panel2_widget = QWidget()
        panel2_widget.setLayout(panel2)
        panel2_widget.setFixedWidth(250)  # 固定宽度
        
        # 设置第2块在垂直方向上的大小策略为自适应
        size_policy = panel2_widget.sizePolicy()
        size_policy.setVerticalPolicy(QSizePolicy.Expanding)
        panel2_widget.setSizePolicy(size_policy)
        
        # 第3块 - 待推理文本、开始推理-进度 (自适应宽度)
        panel3 = QVBoxLayout()
        panel3_widget = QWidget()
        panel3_widget.setLayout(panel3)
        
        # 设置第3块在垂直方向上的大小策略为自适应
        size_policy = panel3_widget.sizePolicy()
        size_policy.setVerticalPolicy(QSizePolicy.Expanding)
        size_policy.setHorizontalPolicy(QSizePolicy.Expanding)  # 水平方向也设为自适应
        panel3_widget.setSizePolicy(size_policy)
        
        # 第4块 - 历史结果 (固定宽度)
        panel4 = QVBoxLayout()
        panel4_widget = QWidget()
        panel4_widget.setLayout(panel4)
        panel4_widget.setFixedWidth(300)  # 减小宽度，只包含历史列表
        
        # 设置第4块在垂直方向上的大小策略为自适应
        size_policy = panel4_widget.sizePolicy()
        size_policy.setVerticalPolicy(QSizePolicy.Expanding)
        panel4_widget.setSizePolicy(size_policy)
        
        # === 第1块内容 - 角色选择列表 ===
        
        # 角色选择区域
        role_group = QGroupBox("角色选择")
        role_layout = QVBoxLayout()
        role_layout.setContentsMargins(5, 5, 5, 5)  # 减小内边距
        role_layout.setSpacing(5)  # 减小间距
        
        # 角色列表
        self.role_list = QListWidget()
        self.role_list.setMinimumWidth(150)
        self.role_list.currentItemChanged.connect(self.on_role_selection_changed)
        
        # 添加角色刷新按钮
        self.refresh_role_btn = QPushButton("刷新角色")
        self.refresh_role_btn.clicked.connect(self.on_refresh_role_clicked)
        
        # 音色列表
        emotion_label = QLabel("音色选择:")
        self.emotion_combo = QComboBox()
        self.emotion_combo.currentIndexChanged.connect(self.on_emotion_selection_changed)
        
        # 添加到布局
        role_layout.addWidget(self.role_list)
        role_layout.addWidget(self.refresh_role_btn)  # 添加刷新角色按钮
        role_layout.addWidget(emotion_label)
        role_layout.addWidget(self.emotion_combo)
        role_group.setLayout(role_layout)
        
        # 添加所有组件到第1块 - 角色选择列表占满高度
        panel1.addWidget(role_group, 1)  # 使用权重1让角色选择列表占满高度
        
        # === 第2块内容 - 参考音频、辅助参考音频、合成参数 ===
        
        # 参考音频区域
        ref_group = QGroupBox("参考音频")
        ref_layout = QVBoxLayout()
        ref_layout.setContentsMargins(5, 5, 5, 5)  # 减小内边距
        
        # 参考音频信息
        ref_info_layout = QFormLayout()
        ref_info_layout.setContentsMargins(0, 0, 0, 0)  # 减小内边距
        ref_info_layout.setSpacing(5)  # 减小间距
        self.ref_audio_label = QLabel("")
        ref_info_layout.addRow("音频:", self.ref_audio_label)
        
        # 参考文本
        self.prompt_text = QTextEdit()
        self.prompt_text.setReadOnly(True)
        self.prompt_text.setMaximumHeight(100)
        ref_info_layout.addRow("参考文本:", self.prompt_text)
        
        # 波形图
        self.ref_waveform = WaveformCanvas(self, width=5, height=1.5, dpi=100, max_points=2000)
        self.ref_waveform.setMinimumHeight(80)
        self.ref_waveform.setFixedHeight(80)  # 设置固定高度
        self.ref_waveform.playback_position_changed.connect(self.on_ref_waveform_clicked)
        
        # 播放控制和音量滑块放在同一行
        control_layout = QHBoxLayout()
        control_layout.setContentsMargins(0, 0, 0, 0)  # 减小内边距
        control_layout.setSpacing(10)  # 控件之间的间距
        
        # 播放按钮
        self.ref_play_btn = QPushButton("播放")
        self.ref_play_btn.clicked.connect(self.on_ref_play_clicked)
        self.ref_play_btn.setMinimumWidth(60)  # 设置最小宽度
        
        # 音量控制标签和滑块
        volume_label = QLabel("音量:")
        self.ref_volume_slider = QSlider(Qt.Horizontal)
        self.ref_volume_slider.setRange(0, 100)
        self.ref_volume_slider.setValue(100)  
        self.ref_volume_slider.valueChanged.connect(self.on_ref_volume_changed)
        
        # 添加到水平布局
        control_layout.addWidget(self.ref_play_btn)
        control_layout.addWidget(volume_label)
        control_layout.addWidget(self.ref_volume_slider, 1)  # 音量滑块占据剩余空间
        
        # 添加到布局
        ref_layout.addLayout(ref_info_layout)
        ref_layout.addWidget(self.ref_waveform)
        ref_layout.addLayout(control_layout)  # 使用新的控制布局
        ref_group.setLayout(ref_layout)
        
        # 辅助参考音频区域
        aux_group = QGroupBox("辅助参考音频")
        aux_layout = QVBoxLayout()
        aux_layout.setContentsMargins(5, 5, 5, 5)  # 减小内边距
        
        # 辅助参考音频列表
        self.aux_ref_list = QListWidget()
        self.aux_ref_list.itemDoubleClicked.connect(self.on_aux_ref_double_clicked)
        
        # 添加到布局
        aux_layout.addWidget(self.aux_ref_list)
        aux_group.setLayout(aux_layout)
        
        # 音频设备设置
        audio_device_group = QGroupBox("音频设备")
        audio_device_layout = QFormLayout()
        audio_device_layout.setContentsMargins(5, 5, 5, 5)
        audio_device_layout.setSpacing(5)
        
        # 音频设备选择下拉框
        self.audio_device_combo = QComboBox()
        self.audio_device_combo.currentIndexChanged.connect(self.on_audio_device_changed)
        
        # 添加刷新按钮
        refresh_device_layout = QHBoxLayout()
        refresh_device_layout.addWidget(self.audio_device_combo, 1)  # 1表示可伸缩
        
        self.refresh_device_btn = QPushButton("刷新")
        self.refresh_device_btn.clicked.connect(self.on_refresh_audio_devices)
        refresh_device_layout.addWidget(self.refresh_device_btn)
        
        # 音量控制放在设备选择下方
        self.global_volume_slider = QSlider(Qt.Horizontal)
        self.global_volume_slider.setRange(0, 150)  # 扩大音量范围到150%
        self.global_volume_slider.setValue(100)
        self.global_volume_slider.valueChanged.connect(self.on_global_volume_changed)
        
        # 添加到布局
        audio_device_layout.addRow("输出设备:", refresh_device_layout)
        audio_device_layout.addRow("主音量:", self.global_volume_slider)
        audio_device_group.setLayout(audio_device_layout)
        
        # 参数区域
        params_group = QGroupBox("合成参数")
        params_layout = QFormLayout()
        params_layout.setContentsMargins(5, 5, 5, 5)  # 减小内边距
        params_layout.setSpacing(5)  # 减小间距
        
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
        
        # 禁用所有数值输入控件的滚轮事件
        self.disable_wheel_event(self.speed)
        self.disable_wheel_event(self.top_k)
        self.disable_wheel_event(self.top_p)
        self.disable_wheel_event(self.temperature)
        self.disable_wheel_event(self.sample_steps)
        self.disable_wheel_event(self.pause_second)
        
        # 切分标点
        self.cut_punc = QLineEdit("。！？.!?")
        
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
        
        # 添加所有组件到第2块
        panel2.addWidget(ref_group, 0)  # 参考音频区域固定高度
        panel2.addWidget(aux_group, 1)  # 辅助参考音频区域自适应填充
        panel2.addWidget(audio_device_group, 0)  # 音频设备设置固定大小
        panel2.addWidget(params_group, 0)  # 参数区域固定高度
        
        # === 第3块内容 - 待推理文本、开始推理-进度 ===
        
        # 推理文本区域
        text_group = QGroupBox("待推理文本")
        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(5, 5, 5, 5)  # 减小内边距
        
        # 使用自定义的纯文本编辑框替代QTextEdit
        self.input_text = PlainTextEdit()
        self.input_text.setPlaceholderText("请在此输入需要转换为语音的文本...")
        # 优化文本编辑性能设置
        self.input_text.setUndoRedoEnabled(True)
        self.input_text.setCursorWidth(2)  # 增加光标宽度提高可见性
        self.input_text.setTabChangesFocus(True)  # Tab键切换焦点而非插入制表符
        # 设置文本提示和自动调整大小
        self.input_text.setMinimumHeight(100)
        # 禁用复杂的文本渲染特性，减少残影
        self.input_text.document().setDocumentMargin(6)
        self.input_text.setAutoFormatting(QTextEdit.AutoNone)
        
        
        
        # 结果音频区域
        result_group = QGroupBox("结果音频")
        result_layout = QVBoxLayout()
        result_layout.setContentsMargins(5, 5, 5, 5)  # 减小内边距
        
        # 结果音频波形图
        self.result_waveform = WaveformCanvas(self, width=5, height=1.5, dpi=100, max_points=2000)
        self.result_waveform.setMinimumHeight(80)
        self.result_waveform.setFixedHeight(80)  # 设置与参考音频相同的固定高度
        self.result_waveform.playback_position_changed.connect(self.on_result_waveform_clicked)
        
        # 播放控制和音量滑块放在同一行
        control_layout = QHBoxLayout()
        control_layout.setContentsMargins(0, 0, 0, 0)  # 减小内边距
        control_layout.setSpacing(10)  # 控件之间的间距
        
        # 播放按钮
        self.result_play_btn = QPushButton("播放")
        self.result_play_btn.clicked.connect(self.on_result_play_clicked)
        self.result_play_btn.setMinimumWidth(60)  # 设置最小宽度
        
        # 保存按钮
        self.result_save_btn = QPushButton("另存为")
        self.result_save_btn.clicked.connect(self.on_result_save_clicked)
        self.result_save_btn.setMinimumWidth(60)  # 设置最小宽度
        
        # 音量控制标签和滑块
        volume_label = QLabel("音量:")
        self.result_volume_slider = QSlider(Qt.Horizontal)
        self.result_volume_slider.setRange(0, 100)
        self.result_volume_slider.setValue(100) 
        self.result_volume_slider.valueChanged.connect(self.on_result_volume_changed)
        
        # 添加到水平布局
        control_layout.addWidget(self.result_play_btn)
        control_layout.addWidget(self.result_save_btn)
        control_layout.addWidget(volume_label)
        control_layout.addWidget(self.result_volume_slider, 1)  # 音量滑块占据剩余空间
        
        # 添加到布局
        result_layout.addWidget(self.result_waveform)
        result_layout.addLayout(control_layout)  # 使用新的控制布局
        result_group.setLayout(result_layout)
        
        # 将结果音频区域添加到第3块
        panel3.addWidget(result_group, 0)  # 结果音频区域固定高度


        # 添加到布局
        text_layout.addWidget(self.input_text)
        text_group.setLayout(text_layout)
        
        # 推理控制区域
        ctrl_group = QGroupBox("推理控制")
        ctrl_layout = QHBoxLayout()
        ctrl_layout.setContentsMargins(5, 5, 5, 5)  # 减小内边距
        ctrl_layout.setSpacing(5)  # 减小间距
        
        self.infer_btn = QPushButton("开始推理")
        self.infer_btn.clicked.connect(self.on_infer_clicked)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        ctrl_layout.addWidget(self.infer_btn)
        ctrl_layout.addWidget(self.progress_bar)
        ctrl_group.setLayout(ctrl_layout)
        
        # 添加所有组件到第3块
        panel3.addWidget(text_group, 1)  # 文本区域占据更多空间
        panel3.addWidget(ctrl_group, 0)  # 控制区域固定高度
        
        # === 第4块内容 - 历史结果 ===
        
        # 历史结果区域
        history_group = QGroupBox("历史结果")
        history_layout = QVBoxLayout()
        history_layout.setContentsMargins(5, 5, 5, 5)  # 减小内边距
        
        self.history_list = QListWidget()
        self.history_list.itemDoubleClicked.connect(self.on_history_item_double_clicked)
        
        history_btn_layout = QHBoxLayout()
        history_btn_layout.setContentsMargins(0, 0, 0, 0)  # 减小内边距
        history_btn_layout.setSpacing(5)  # 减小间距
        self.refresh_history_btn = QPushButton("刷新")
        self.refresh_history_btn.clicked.connect(self.on_refresh_history_clicked)
        self.clear_history_btn = QPushButton("清空")
        self.clear_history_btn.clicked.connect(self.on_clear_history_clicked)
        
        history_btn_layout.addWidget(self.refresh_history_btn)
        history_btn_layout.addWidget(self.clear_history_btn)
        
        history_layout.addWidget(self.history_list)
        history_layout.addLayout(history_btn_layout)
        history_group.setLayout(history_layout)
        
        # 将历史结果添加到第4块
        panel4.addWidget(history_group, 1)  # 历史结果区域占满高度
        
        # 添加四个面板到主布局
        main_layout.addWidget(panel1_widget)
        main_layout.addWidget(panel2_widget)
        main_layout.addWidget(panel3_widget, 1)  # 第3块自适应宽度
        main_layout.addWidget(panel4_widget)
        
        # 禁用控件，直到选择角色
        self.set_inference_widgets_enabled(False)
    
    def disable_wheel_event(self, widget):
        """禁用控件的滚轮事件"""
        widget.wheelEvent = lambda event: event.ignore()
    
    def init_player(self):
        """初始化音频播放器"""
        # 参考音频播放器
        self.ref_player = QMediaPlayer()
        self.ref_audio_output = QAudioOutput()
        self.ref_player.setAudioOutput(self.ref_audio_output)
        self.ref_player.playbackStateChanged.connect(self.on_ref_playback_state_changed)
        self.ref_player.positionChanged.connect(self.on_ref_position_changed)
        # 设置默认音量为80%
        self.ref_audio_output.setVolume(0.8)
        
        # 结果音频播放器
        self.result_player = QMediaPlayer()
        self.result_audio_output = QAudioOutput()
        self.result_player.setAudioOutput(self.result_audio_output)
        self.result_player.playbackStateChanged.connect(self.on_result_playback_state_changed)
        self.result_player.positionChanged.connect(self.on_result_position_changed)
        # 设置默认音量为80%
        self.result_audio_output.setVolume(0.8)
        
        # 初始化音频设备列表
        self.refresh_audio_devices()
    
    def set_inference_widgets_enabled(self, enabled):
        """启用或禁用推理控件"""
        for widget in [
            self.emotion_combo, self.ref_play_btn, self.aux_ref_list,
            self.speed, self.top_k, self.top_p, self.temperature,
            self.sample_steps, self.pause_second, self.cut_punc,
            self.ref_free, self.if_sr, self.input_text, self.infer_btn,
            self.result_play_btn, self.result_save_btn,
            self.ref_volume_slider, self.result_volume_slider,
            self.audio_device_combo, self.refresh_device_btn,  # 添加音频设备相关控件
            self.global_volume_slider  # 添加全局音量滑块
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
            # 对于参考音频，通常较短，可以全部加载
            self.ref_waveform.plot_waveform(audio_path)
    
    def set_result_audio(self, audio_path):
        """设置结果音频"""
        if audio_path:
            self.result_player.setSource(QUrl.fromLocalFile(audio_path))
            # 显示完整音频波形图，不限制时长
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
    
    def set_progress(self, value, segment_info=None):
        """设置进度条值
        
        Args:
            value: 进度值（0-100）
            segment_info: 段落信息字典，包含 current_segment 和 total_segments
        """
        self.progress_bar.setValue(value)
        
        # 如果提供了段落信息，更新进度条格式
        if segment_info and 'current_segment' in segment_info and 'total_segments' in segment_info:
            current = segment_info['current_segment'] + 1  # 转为从1开始计数
            total = segment_info['total_segments']
            # 设置进度条格式文本
            self.progress_bar.setFormat(f"{value}% (第{current}/{total}段)")
        else:
            # 恢复默认格式
            self.progress_bar.setFormat("%p%")
    
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
    
    def on_refresh_role_clicked(self):
        """刷新角色按钮点击事件"""
        self.role_refresh.emit()
    
    def on_ref_volume_changed(self, value):
        """参考音频音量改变事件"""
        self.ref_audio_output.setVolume(value / 100.0)
        # 更新全局音量滑块（如果不是由全局音量改变触发的）
        if self.global_volume_slider.value() != value:
            self.global_volume_slider.setValue(value)
    
    def on_result_volume_changed(self, value):
        """结果音频音量改变事件"""
        self.result_audio_output.setVolume(value / 100.0)
        # 更新全局音量滑块（如果不是由全局音量改变触发的）
        if self.global_volume_slider.value() != value:
            self.global_volume_slider.setValue(value)
    
    def refresh_audio_devices(self):
        """刷新音频设备列表"""
        self.audio_device_combo.clear()
        
        # 获取所有可用的音频输出设备
        output_devices = QMediaDevices.audioOutputs()
        
        # 添加设备到下拉框
        for device in output_devices:
            self.audio_device_combo.addItem(device.description(), device)
        
        # 添加默认设备选项
        self.audio_device_combo.insertItem(0, "默认系统设备", None)
        self.audio_device_combo.setCurrentIndex(0)  # 默认选择系统默认设备
    
    def on_refresh_audio_devices(self):
        """刷新音频设备按钮点击事件"""
        self.refresh_audio_devices()
        self.show_message("提示", "已刷新音频设备列表")
    
    def on_audio_device_changed(self, index):
        """音频设备选择改变事件"""
        if index >= 0:
            device = self.audio_device_combo.currentData()
            self.set_audio_device(device)
            
            # # 如果选择了具体设备，显示提示
            # if device:
            #     self.show_message("设备切换", f"已切换到: {self.audio_device_combo.currentText()}")
    
    def set_audio_device(self, device):
        """设置音频设备"""
        if device:
            # 设置两个播放器的音频设备
            self.ref_audio_output.setDevice(device)
            self.result_audio_output.setDevice(device)
        else:
            # 使用系统默认设备
            default_device = QMediaDevices.defaultAudioOutput()
            self.ref_audio_output.setDevice(default_device)
            self.result_audio_output.setDevice(default_device)
    
    def on_global_volume_changed(self, value):
        """全局音量改变事件"""
        # 将0-150的值转换为0-1.5的浮点数，允许放大音量到150%
        volume = value / 100.0
        
        # 设置两个播放器的音量
        self.ref_audio_output.setVolume(volume)
        self.result_audio_output.setVolume(volume)
        
        # 同步更新两个音量滑块
        self.ref_volume_slider.setValue(value)
        self.result_volume_slider.setValue(value) 