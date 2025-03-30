#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPT-SoVITS API界面入口

该文件是重构后的UI入口，调用ui目录下的模块化实现
"""

from ui.main import create_ui

if __name__ == "__main__":
    app = create_ui()
    app.launch(server_name="0.0.0.0", server_port=7878, inbrowser=True) 