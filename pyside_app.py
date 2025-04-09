#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPT-SoVITS PySide6应用程序启动脚本
"""

import os
import sys

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pyside_ui.main import MainWindow
from PySide6.QtWidgets import QApplication


if __name__ == "__main__":
    # 确保必要的目录存在
    os.makedirs("configs/roles", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 