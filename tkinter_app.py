#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPT-SoVITS Tkinter UI界面入口

该文件是使用Tkinter实现的UI入口
"""

import os
import sys
from pathlib import Path
import tkinter as tk

# 导入tkinter UI实现
from tkinter_ui import TkinterApp

if __name__ == "__main__":
    # 检查是否需要迁移角色配置
    roles_dir = Path("configs/roles")
    if not roles_dir.exists():
        os.makedirs(roles_dir, exist_ok=True)
        print("创建角色配置目录:", roles_dir)
    
    # 检查输出目录
    output_dir = Path("output")
    if not output_dir.exists():
        os.makedirs(output_dir, exist_ok=True)
        print("创建输出目录:", output_dir)
    
    # 启动Tkinter应用
    root = tk.Tk()
    app = TkinterApp(root)
    root.mainloop() 