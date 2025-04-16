#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPT-SoVITS GUI应用
使用MVC模式组织代码结构
"""
import sys
from PySide6.QtWidgets import QApplication
from gui.main_window import MainWindow

def main():
    """程序入口"""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 