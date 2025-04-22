"""
主窗口模块

应用程序的主窗口界面
"""

import os
import glob
from pathlib import Path
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QMessageBox, QSplitter, QLabel, QPushButton
from PySide6.QtCore import Qt, QSize

from gui.controllers import RoleController, InferenceController
from gui.tabs import ExperimentTab, RoleTab
from gui.components.audio_player import AudioPlayer
from gui.components.history_list import HistoryList


class MainWindow(QMainWindow):
    """主窗口"""
    
    def __init__(self):
        super().__init__()
        
        # 初始化控制器
        self.role_controller = RoleController()
        self.inference_controller = InferenceController()
        
        # 扫描模型文件
        self.gpt_models = self.scan_models(["GPT_weights", "GPT_weights_v2", "GPT_weights_v3", "GPT_weights_v4"])
        self.sovits_models = self.scan_models(["SoVITS_weights", "SoVITS_weights_v2", "SoVITS_weights_v3", "SoVITS_weights_v4"])
        
        self.init_ui()
        self.connect_signals()
        
        # 加载模型到下拉框
        self.experiment_tab.load_gpt_models(self.gpt_models)
        self.experiment_tab.load_sovits_models(self.sovits_models)
        
        # 加载并显示历史记录
        self.update_history_display()
    
    def scan_models(self, model_dirs):
        """扫描模型文件夹，返回模型名称和路径的字典"""
        models_dict = {}
        
        # 获取项目根目录
        root_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        
        # 扫描每个模型目录
        for model_dir in model_dirs:
            dir_path = root_dir / model_dir
            if not dir_path.exists():
                continue
                
            # 查找所有模型文件 (.pth, .ckpt)
            for model_file in glob.glob(str(dir_path / "*.pth")) + glob.glob(str(dir_path / "*.ckpt")):
                model_path = Path(model_file)
                # 使用相对路径作为显示名称
                display_name = f"{model_dir}/{model_path.name}"
                # 使用绝对路径作为实际值
                models_dict[display_name] = str(model_path.absolute())
        
        return models_dict
    
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
        
        # 实验选项卡 - 创建为共享控件版本
        self.experiment_tab = ExperimentTab(self.role_controller, self.inference_controller, shared_controls=True)
        self.tab_widget.addTab(self.experiment_tab, "试听配置")
        
        # 角色选项卡 - 创建为共享控件版本
        self.role_tab = RoleTab(self.role_controller, self.inference_controller, shared_controls=True)
        self.tab_widget.addTab(self.role_tab, "角色推理")
        
        left_layout.addWidget(self.tab_widget)
        
        # 右侧部分 - 共享控件
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # 状态显示
        self.progress_label = QLabel("就绪")
        right_layout.addWidget(self.progress_label)
        
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
        
        self.inference_controller.inference_started.connect(self.on_inference_started)
        self.inference_controller.inference_completed.connect(self.on_inference_completed)
        self.inference_controller.inference_failed.connect(self.on_inference_failed)
        self.inference_controller.progress_updated.connect(self.update_progress)
        
        # 共享组件信号
        self.inference_controller.history_updated.connect(self.update_history_display)
        self.history_list.audio_selected.connect(self.audio_player.load_audio)
        self.history_list.history_cleared.connect(self.inference_controller.clear_history)
        
        # 标签页生成按钮信号
        self.experiment_tab.generate_requested.connect(self.on_generate_requested)
        self.role_tab.generate_requested.connect(self.on_generate_requested)
    
    def update_history_display(self):
        """更新历史记录显示"""
        history = self.inference_controller.get_history()
        self.history_list.update_history(history)
    
    def on_inference_started(self):
        """推理开始回调"""
        self.status_bar.showMessage("正在生成语音...")
        self.generate_button.setEnabled(False)
        self.progress_label.setText("正在处理...")
    
    def on_inference_completed(self, file_path: str):
        """推理完成回调"""
        self.status_bar.showMessage("生成完成")
        self.generate_button.setEnabled(True)
        self.progress_label.setText("就绪")
        self.audio_player.load_audio(file_path)
    
    def on_inference_failed(self, error_msg: str):
        """推理失败回调"""
        self.status_bar.showMessage(f"生成失败: {error_msg}")
        self.generate_button.setEnabled(True)
        self.progress_label.setText(f"失败: {error_msg}")
    
    def update_progress(self, message: str):
        """更新进度信息"""
        self.status_bar.showMessage(message)
        self.progress_label.setText(message)
    
    def on_generate_requested(self, config=None, text="", is_role=False):
        """处理来自标签页的生成请求"""
        if is_role:
            # 角色推理
            if config and text:
                config["text"] = text
                self.inference_controller.generate_speech_async(config)
        else:
            # 实验模式推理
            if config:
                self.inference_controller.generate_speech_async(config)
    
    def show_error(self, message: str):
        """显示错误信息"""
        QMessageBox.critical(self, "错误", message)
    
    def closeEvent(self, event):
        """窗口关闭事件处理"""
        # 保存历史记录
        self.inference_controller.save_history()
        # 继续正常关闭
        event.accept()

    def generate_speech(self):
        """生成语音"""
        # 根据当前选中的标签页决定使用哪种生成方法
        current_index = self.tab_widget.currentIndex()
        
        if current_index == 0:  # 实验选项卡
            config = self.experiment_tab.get_current_config()
            text = self.experiment_tab.text_edit.toPlainText()
            
            # 验证必要参数
            if not text:
                self.show_error("请输入要合成的文本")
                return
                
            if not config["gpt_model"]:
                self.show_error("请选择GPT模型")
                return
                
            if not config["sovits_model"]:
                self.show_error("请选择SoVITS模型")
                return
                
            if not config["ref_free"] and not config["ref_audio"]:
                self.show_error("请选择参考音频文件或勾选无参考文本")
                return
                
            if not config["ref_free"] and not os.path.exists(config["ref_audio"]):
                self.show_error(f"参考音频文件不存在: {config['ref_audio']}")
                return
                
            # 调用推理控制器
            self.inference_controller.generate_speech_async(config)
            
        elif current_index == 1:  # 角色选项卡
            role_name = self.role_tab.current_role
            emotion_name = self.role_tab.current_emotion
            text = self.role_tab.text_edit.toPlainText()
            
            if not role_name or not emotion_name:
                self.show_error("请先选择角色和情感")
                return
                
            if not text:
                self.show_error("请输入要合成的文本")
                return
                
            # 获取角色配置
            config = self.role_controller.get_emotion_config(role_name, emotion_name)
            if not config:
                self.show_error("无法获取角色配置")
                return
                
            # 检查参考音频路径是否存在
            ref_audio = config.get("ref_audio", "")
            if ref_audio and not os.path.exists(ref_audio):
                self.show_error(f"参考音频文件不存在: {ref_audio}")
                return
                
            # 添加当前文本到配置中
            config["text"] = text
                
            # 调用推理控制器
            self.inference_controller.generate_speech_async(config) 