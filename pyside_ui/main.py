#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPT-SoVITS 基于PySide6的UI界面
实现与原Tkinter UI类似的功能
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget
from PySide6.QtCore import Qt

from pyside_ui.tabs.text_file_tab import TextFileTab
from pyside_ui.tabs.config_tab import ConfigTab
from pyside_ui.tabs.role_management_tab import RoleManagementTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GPT-SoVITS 文本转语音 (PySide6版)")
        self.resize(1200, 800)
        
        # 创建中央部件
        self.central_widget = QTabWidget()
        self.setCentralWidget(self.central_widget)
        
        # 创建各个标签页
        self.text_file_tab = TextFileTab()
        self.role_management_tab = RoleManagementTab()
        self.config_tab = ConfigTab()
        
        # 添加标签页
        self.central_widget.addTab(self.text_file_tab, "文本文件")
        self.central_widget.addTab(self.role_management_tab, "角色管理")
        self.central_widget.addTab(self.config_tab, "配置替换音")


if __name__ == "__main__":
    # 确保必要的目录存在
    os.makedirs("configs/roles", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 