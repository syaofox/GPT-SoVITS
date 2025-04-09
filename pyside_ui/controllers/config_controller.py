#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置控制器
"""

from ui.utils import load_word_replace_config, save_word_replace_config


class ConfigController:
    def __init__(self):
        """初始化控制器"""
        pass
    
    def load_word_replace_config(self):
        """加载词语替换配置"""
        return load_word_replace_config()
    
    def save_word_replace_config(self, text):
        """保存词语替换配置"""
        return save_word_replace_config(text) 