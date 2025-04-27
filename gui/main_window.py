"""
主窗口模块

应用程序的主窗口界面
"""

import os
import shutil
import tempfile
from pathlib import Path
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QMessageBox, QSplitter, QLabel, QPushButton, QProgressBar, QGroupBox, QGridLayout, QComboBox
from PySide6.QtCore import Qt

from gui.controllers import RoleController, InferenceController
from gui.views import ExperimentTab, RoleTab, ReplaceTab
from gui.components.audio_player import AudioPlayer
from gui.components.history_list import HistoryList
from gui.models import ModelManager, TextProcessor, ProgressManager, ConfigApplier, InferenceRequestHandler


class MainWindow(QMainWindow):
    """主窗口"""
    
    def __init__(self):
        super().__init__()
        
        # 初始化模型管理器和文本处理器
        self.model_manager = ModelManager()
        self.text_processor = TextProcessor()
        
        # 初始化控制器
        self.role_controller = RoleController()
        self.inference_controller = InferenceController()
        
        # 扫描模型文件
        self.gpt_models = self.model_manager.scan_models(["GPT_weights", "GPT_weights_v2", "GPT_weights_v3", "GPT_weights_v4"])
        self.sovits_models = self.model_manager.scan_models(["SoVITS_weights", "SoVITS_weights_v2", "SoVITS_weights_v3", "SoVITS_weights_v4"])
        
        # 当前角色和情感
        self.current_role = ""
        self.current_emotion = ""
        
        # 初始化UI
        self.init_ui()
        
        # 初始化进度管理器
        self.progress_manager = ProgressManager(
            self.progress_bar, 
            self.progress_label, 
            self.status_bar,
            self.generate_button
        )
        # 设置主窗口引用，使进度管理器能够控制UI元素
        self.progress_manager.set_main_window(self)
        
        # 初始化配置应用器
        self.config_applier = ConfigApplier(
            self.experiment_tab,
            self.gpt_models,
            self.sovits_models
        )
        
        # 初始化推理请求处理器
        self.inference_handler = InferenceRequestHandler(
            self.inference_controller,
            self.text_processor,
            self
        )
        
        # 连接信号槽
        self.connect_signals()
        
        # 加载模型到下拉框
        self.experiment_tab.load_gpt_models(self.gpt_models)
        self.experiment_tab.load_sovits_models(self.sovits_models)
        
        # 刷新角色列表
        self.refresh_roles()
    
    def init_ui(self):
        """初始化界面"""
        self.setWindowTitle("GPT-SoVITS 语音合成")
        self.setMinimumSize(1100, 700)
        
        # 中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧部分
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # 选项卡
        self.tab_widget = QTabWidget()
        # self.tab_widget.currentChanged.connect(self.on_tab_changed)
        
        # 角色选项卡 - 创建为共享控件版本，不包含角色选择
        self.role_tab = RoleTab(self.role_controller, self.inference_controller, shared_controls=True)
        self.tab_widget.addTab(self.role_tab, "角色推理")
        
        # 实验选项卡 - 创建为共享控件版本
        self.experiment_tab = ExperimentTab(self.role_controller, self.inference_controller, shared_controls=True)
        self.tab_widget.addTab(self.experiment_tab, "试听配置")
        
        # 替换规则编辑标签页
        self.replace_tab = ReplaceTab()
        self.tab_widget.addTab(self.replace_tab, "替换规则")
        
        left_layout.addWidget(self.tab_widget)
        
        # 右侧部分 - 共享控件
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # 公共角色选择部分（放在右侧最上方）
        role_group = QGroupBox("角色选择")
        role_layout = QGridLayout(role_group)
        
        role_layout.addWidget(QLabel("角色:"), 0, 0)
        self.role_combo = QComboBox()
        self.role_combo.currentIndexChanged.connect(self.on_role_changed)
        role_layout.addWidget(self.role_combo, 0, 1)
        
        role_layout.addWidget(QLabel("情感:"), 1, 0)
        self.emotion_combo = QComboBox()
        self.emotion_combo.currentIndexChanged.connect(self.on_emotion_changed)
        role_layout.addWidget(self.emotion_combo, 1, 1)
        
        self.refresh_button = QPushButton("刷新列表")
        self.refresh_button.clicked.connect(self.refresh_roles)
        role_layout.addWidget(self.refresh_button, 2, 0, 1, 2)
        
        right_layout.addWidget(role_group)
        
        # 进度显示区域
        progress_layout = QVBoxLayout()
        
        # 状态文字显示
        self.progress_label = QLabel("就绪")
        self.progress_label.setAlignment(Qt.AlignLeft)
        progress_layout.addWidget(self.progress_label)
        
        # 添加进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)  # 不在进度条上显示文字
        progress_layout.addWidget(self.progress_bar)
        
        right_layout.addLayout(progress_layout)
        
        # 生成语音按钮
        self.generate_button = QPushButton("生成语音")
        self.generate_button.clicked.connect(self.generate_speech)
        right_layout.addWidget(self.generate_button)
        
        # 音频播放器
        self.audio_player = AudioPlayer()
        right_layout.addWidget(self.audio_player)
        
        # 历史记录
        history_group_layout = QVBoxLayout()
        self.history_list = HistoryList()
        history_group_layout.addWidget(self.history_list)
        right_layout.addLayout(history_group_layout)
        
        # 添加到主布局的分割器
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([600, 400])
        
        main_layout.addWidget(splitter)
        
        # 状态栏
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("就绪")
    
    def connect_signals(self):
        """连接信号槽"""
        # 控制器信号
        self.role_controller.error_occurred.connect(self.show_error)
        self.inference_controller.error_occurred.connect(self.show_error)
        
        # 使用进度管理器处理推理信号
        self.inference_controller.inference_started.connect(self.progress_manager.on_inference_started)
        self.inference_controller.inference_completed.connect(self.progress_manager.on_inference_completed)
        self.inference_controller.inference_failed.connect(self.progress_manager.on_inference_failed)
        self.inference_controller.inference_stopped.connect(self.progress_manager.on_inference_stopped)
        self.inference_controller.progress_updated.connect(self.progress_manager.update_progress)
        
        # 推理完成信号连接到音频生成处理
        self.inference_controller.inference_completed.connect(self.on_new_audio_generated)
        
        # 历史列表和音频播放器连接
        self.history_list.audio_selected.connect(self.audio_player.load_audio)
        
        # 标签页生成按钮信号
        self.experiment_tab.generate_requested.connect(self.inference_handler.on_generate_requested)
        self.role_tab.generate_requested.connect(self.inference_handler.on_generate_requested)
        
        # 连接替换规则更新信号
        self.replace_tab.rules_updated.connect(self.update_replace_rules)
    
    def refresh_roles(self):
        """刷新角色列表"""
        self.role_controller.refresh_roles()
        roles = self.role_controller.get_role_names()
        
        # 保存当前选择
        current_role = self.role_combo.currentText() if self.role_combo.count() > 0 else ""
        
        # 更新角色下拉框
        self.role_combo.clear()
        self.role_combo.addItems(roles)
        
        # 恢复之前的选择
        if current_role and current_role in roles:
            index = self.role_combo.findText(current_role)
            self.role_combo.setCurrentIndex(index)
    
    def on_role_changed(self, index):
        """角色改变回调"""
        if index < 0:
            return
            
        self.current_role = self.role_combo.currentText()
        self.update_emotions()
        
        # 同步到角色标签页
        if hasattr(self.role_tab, 'role_combo'):
            index = self.role_tab.role_combo.findText(self.current_role)
            if index >= 0:
                self.role_tab.role_combo.setCurrentIndex(index)
    
    def update_emotions(self):
        """更新情感列表"""
        if not self.current_role:
            self.emotion_combo.clear()
            return
            
        emotions = self.role_controller.get_emotion_names(self.current_role)
        
        # 保存当前选择
        current_emotion = self.emotion_combo.currentText() if self.emotion_combo.count() > 0 else ""
        
        # 更新情感下拉框
        self.emotion_combo.clear()
        self.emotion_combo.addItems(emotions)
        
        # 恢复之前的选择
        if current_emotion and current_emotion in emotions:
            index = self.emotion_combo.findText(current_emotion)
            self.emotion_combo.setCurrentIndex(index)
        elif emotions:
            # 默认选择第一个情感并更新配置
            self.emotion_combo.setCurrentIndex(0)
        
        # 情感更新后获取当前配置
        self.get_current_emotion_config()
    
    def on_emotion_changed(self, index):
        """情感改变回调"""
        if index < 0:
            return
            
        self.current_emotion = self.emotion_combo.currentText()
        # 情感改变时获取当前配置
        self.get_current_emotion_config()
        
        # 同步到角色标签页
        if hasattr(self.role_tab, 'emotion_combo'):
            index = self.role_tab.emotion_combo.findText(self.current_emotion)
            if index >= 0:
                self.role_tab.emotion_combo.setCurrentIndex(index)
    
    def get_current_emotion_config(self):
        """获取当前选中角色和情感的配置，并发射信号更新试听配置"""
        if not self.current_role or not self.current_emotion:
            return
            
        # 获取当前情感配置
        emotion_config = self.role_controller.get_emotion_config(self.current_role, self.current_emotion)
        if emotion_config:
            # 打印调试信息以便追踪
            print(f"获取角色配置: {self.current_role}/{self.current_emotion}")
            print(f"配置中的模型: gpt={emotion_config.get('gpt_path', '未设置')}, sovits={emotion_config.get('sovits_path', '未设置')}")
            
            # 应用配置到试听配置标签页
            self.config_applier.apply_role_config(emotion_config)
            self.config_applier.apply_role_info(self.current_role, self.current_emotion)
    
    def on_new_audio_generated(self, file_path: str):
        """新音频生成后刷新历史列表"""
        self.history_list.load_output_files()
        
        # 将音频文件复制到临时目录，然后加载播放
        try:
            # 创建临时文件，保留原始文件扩展名
            file_ext = os.path.splitext(file_path)[1]
            temp_dir = tempfile.gettempdir()
            temp_file = os.path.join(temp_dir, f"gpt_sovits_temp_{os.path.basename(file_path)}")
            
            # 复制文件到临时目录
            shutil.copy2(file_path, temp_file)
            
            # 加载临时文件到播放器
            self.audio_player.load_audio(temp_file)
            
            # 更新状态信息
            self.status_bar.showMessage(f"语音已生成: {os.path.basename(file_path)}")
        except Exception as e:
            self.status_bar.showMessage(f"播放音频时出错: {str(e)}")
    
    def show_error(self, message: str):
        """显示错误信息"""
        QMessageBox.critical(self, "错误", message)
    
    def update_replace_rules(self, new_rules):
        """更新替换规则"""
        count = self.text_processor.update_rules(new_rules)
        self.status_bar.showMessage(f"已更新替换规则，共 {count} 条")
    
    def generate_speech(self):
        """生成语音或停止生成"""
        # 根据当前状态决定是生成语音还是停止生成
        if self.progress_manager.is_running():
            # 正在推理中，点击按钮应该停止推理
            self.inference_controller.stop_inference()
        else:
            # 未在推理中，点击按钮应该开始推理
            # 根据当前选中的标签页决定使用哪种生成方法
            current_index = self.tab_widget.currentIndex()
            
            if current_index == 1:  # 实验选项卡
                config = self.experiment_tab.get_inference_config()
                text = self.experiment_tab.text_edit.toPlainText()
                self.inference_handler.handle_experiment_request(config, text)
                    
            elif current_index == 0:  # 角色选项卡
                text = self.role_tab.text_edit.toPlainText()
                
                # 从公共角色选择获取当前角色和情感
                role_name = self.current_role
                emotion_name = self.current_emotion
                
                # 获取角色配置
                config = self.role_controller.get_emotion_config(role_name, emotion_name)
                if config:
                    # 添加文本和角色信息
                    config["text"] = text
                    config["role_name"] = role_name
                    config["emotion_name"] = emotion_name
                    
                    self.inference_handler.handle_role_request(role_name, emotion_name, text, config)
    
    # def on_tab_changed(self, index):
    #     """处理标签页切换"""
    #     # 切换到任何标签页时，都确保角色配置已加载
    #     if self.current_role and self.current_emotion:
    #         self.get_current_emotion_config()
    
    def closeEvent(self, event):
        """窗口关闭事件处理"""
        # InferenceController已经通过atexit注册了程序退出时的保存函数，不需要重复调用
        # 直接继续正常关闭
        event.accept() 