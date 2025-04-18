#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPT-SoVITS GUI应用
使用MVC模式组织代码结构
"""
import sys
import os
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from gui.main_window import MainWindow

def main():
    """程序入口"""
    # 添加环境变量配置，优化Qt渲染
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"  # 启用高DPI缩放
    os.environ["QT_SCALE_FACTOR_ROUNDING_POLICY"] = "PassThrough"  # 不进行四舍五入
    # 设置渲染属性，避免文字残影
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    
    app = QApplication(sys.argv)
    # 设置全局样式表，解决文本渲染问题
    app.setStyleSheet("""
        QTextEdit {
            background-color: white;
            color: black;
            selection-background-color: #4a86e8;
            selection-color: white;
        }
    """)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 