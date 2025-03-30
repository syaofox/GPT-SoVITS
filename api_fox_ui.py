#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPT-SoVITS API界面入口

该文件是重构后的UI入口，调用ui目录下的模块化实现
"""

import os
from pathlib import Path
from ui.main import create_ui

if __name__ == "__main__":
    # 检查是否需要迁移角色配置
    roles_dir = Path("configs/roles")
    needs_migration = False
    
    # 如果目录存在且有.json文件但没有子目录，则需要迁移
    if roles_dir.exists():
        has_json_files = any(roles_dir.glob("*.json"))
        has_dirs = any(item.is_dir() for item in roles_dir.iterdir() if item.name != "__pycache__")
        
        if has_json_files and not has_dirs:
            needs_migration = True
    
    if needs_migration:
        print("检测到旧版角色配置格式，正在进行迁移...")
        try:
            from ui.migrate_roles import migrate_roles
            migrated = migrate_roles()
            if migrated:
                print(f"成功迁移 {len(migrated)} 个角色到新的目录结构")
            else:
                print("没有角色需要迁移")
        except Exception as e:
            print(f"迁移过程中发生错误: {str(e)}")
            print("请手动迁移角色配置，或继续使用旧配置格式")
    
    app = create_ui()
    app.launch(server_name="0.0.0.0", server_port=7878, inbrowser=True) 