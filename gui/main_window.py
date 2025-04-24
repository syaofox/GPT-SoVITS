"""
主窗口模块

应用程序的主窗口界面
"""

import os
from pathlib import Path
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QMessageBox, QSplitter, QLabel, QPushButton, QProgressBar
from PySide6.QtCore import Qt

from gui.controllers import RoleController, InferenceController
from gui.tabs import ExperimentTab, RoleTab, ReplaceTab
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
    
    def init_ui(self):
        """初始化界面"""
        self.setWindowTitle("GPT-SoVITS 语音合成")
        self.setMinimumSize(1100, 700)
        
        # 中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧部分 - 选项卡
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # 选项卡
        self.tab_widget = QTabWidget()
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        
        # 角色选项卡 - 创建为共享控件版本，移至第一位
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
        
        # 连接角色选择信号到配置应用器
        self.role_tab.role_config_selected.connect(self.config_applier.apply_role_config)
        self.role_tab.role_info_selected.connect(self.config_applier.apply_role_info)
        
        # 连接角色更新信号到角色选项卡刷新方法
        self.experiment_tab.role_updated.connect(self.role_tab.refresh_roles)
        
        # 连接替换规则更新信号
        self.replace_tab.rules_updated.connect(self.update_replace_rules)
    
    def on_new_audio_generated(self, file_path: str):
        """新音频生成后刷新历史列表"""
        self.history_list.load_output_files()
        # 不再自动加载到播放器，避免占用文件
        # self.audio_player.load_audio(file_path)
        self.status_bar.showMessage(f"语音已生成: {os.path.basename(file_path)}")
    
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
                role_name = self.role_tab.current_role
                emotion_name = self.role_tab.current_emotion
                text = self.role_tab.text_edit.toPlainText()
                config = self.role_tab.get_inference_config()
                self.inference_handler.handle_role_request(role_name, emotion_name, text, config)
    
    def on_tab_changed(self, index):
        """处理标签页切换"""
        if index == 0:  # 切换到角色选项卡
            # 如果当前角色和情感有选择，则触发配置加载
            if self.role_tab.current_role and self.role_tab.current_emotion:
                self.role_tab.get_current_emotion_config()
    
    def closeEvent(self, event):
        """窗口关闭事件处理"""
        # InferenceController已经通过atexit注册了程序退出时的保存函数，不需要重复调用
        # 直接继续正常关闭
        event.accept() 