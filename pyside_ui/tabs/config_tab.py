#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置标签页UI
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QTextEdit, QMessageBox
)

from pyside_ui.controllers.config_controller import ConfigController


class ConfigTab(QWidget):
    def __init__(self):
        super().__init__()
        
        # 创建控制器
        self.controller = ConfigController()
        
        # 初始化UI
        self.init_ui()
        
        # 连接信号
        self.connect_signals()
    
    def init_ui(self):
        """初始化UI"""
        # 主布局
        main_layout = QVBoxLayout(self)
        
        # 词语替换配置区域
        config_group = QLabel("词语替换配置")
        main_layout.addWidget(config_group)
        
        self.word_replace_text = QTextEdit()
        main_layout.addWidget(self.word_replace_text)
        
        # 尝试加载词语替换配置
        try:
            config_text = self.controller.load_word_replace_config()
            self.word_replace_text.setPlainText(config_text)
        except Exception as e:
            self.word_replace_text.setPlainText(f"加载配置失败: {str(e)}\n\n# 格式：\n# 替换前 替换后\n# 例如：\ntest 测试")
        
        # 按钮区域
        button_layout = QHBoxLayout()
        
        save_btn = QPushButton("保存词语替换配置")
        button_layout.addWidget(save_btn)
        
        refresh_btn = QPushButton("刷新")
        button_layout.addWidget(refresh_btn)
        
        main_layout.addLayout(button_layout)
        
        # 保存按钮引用
        self.save_btn = save_btn
        self.refresh_btn = refresh_btn
    
    def connect_signals(self):
        """连接信号"""
        # 保存按钮
        self.save_btn.clicked.connect(self.save_word_replace)
        
        # 刷新按钮
        self.refresh_btn.clicked.connect(self.refresh_word_replace)
    
    def save_word_replace(self):
        """保存词语替换配置"""
        config_text = self.word_replace_text.toPlainText()
        try:
            result = self.controller.save_word_replace_config(config_text)
            QMessageBox.information(self, "保存结果", result)
        except Exception as e:
            QMessageBox.critical(self, "保存失败", str(e))
    
    def refresh_word_replace(self):
        """刷新词语替换配置"""
        try:
            config_text = self.controller.load_word_replace_config()
            self.word_replace_text.setPlainText(config_text)
            QMessageBox.information(self, "刷新成功", "词语替换配置已刷新")
        except Exception as e:
            QMessageBox.critical(self, "刷新失败", str(e)) 