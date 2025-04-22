"""
主窗口模块

应用程序的主窗口界面
"""

import os
import glob
from pathlib import Path
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QTabWidget, QMessageBox
from PySide6.QtCore import QSize

from gui.controllers import RoleController, InferenceController
from gui.tabs import ExperimentTab, RoleTab


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
        self.setMinimumSize(900, 600)
        
        # 中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 选项卡
        self.tab_widget = QTabWidget()
        
        # 实验选项卡
        self.experiment_tab = ExperimentTab(self.role_controller, self.inference_controller)
        self.tab_widget.addTab(self.experiment_tab, "试听配置")
        
        # 角色选项卡
        self.role_tab = RoleTab(self.role_controller, self.inference_controller)
        self.tab_widget.addTab(self.role_tab, "角色推理")
        
        main_layout.addWidget(self.tab_widget)
        
        # 状态栏
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("就绪")
    
    def connect_signals(self):
        """连接信号槽"""
        self.role_controller.error_occurred.connect(self.show_error)
        self.inference_controller.error_occurred.connect(self.show_error)
        
        self.inference_controller.inference_started.connect(lambda: self.status_bar.showMessage("正在生成语音..."))
        self.inference_controller.inference_completed.connect(lambda _: self.status_bar.showMessage("生成完成"))
        self.inference_controller.inference_failed.connect(lambda err: self.status_bar.showMessage(f"生成失败: {err}"))
        self.inference_controller.progress_updated.connect(self.status_bar.showMessage)
        
        # 连接历史记录更新信号
        self.inference_controller.history_updated.connect(self.update_history_display)
    
    def update_history_display(self):
        """更新并显示历史记录"""
        # 更新实验选项卡的历史记录
        self.experiment_tab.update_history()
        # 更新角色选项卡的历史记录
        self.role_tab.update_history()
    
    def show_error(self, message: str):
        """显示错误信息"""
        QMessageBox.critical(self, "错误", message)
    
    def closeEvent(self, event):
        """窗口关闭事件处理"""
        # 保存历史记录
        self.inference_controller.save_history()
        # 继续正常关闭
        event.accept() 