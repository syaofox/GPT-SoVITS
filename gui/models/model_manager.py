"""
模型管理器模块

负责扫描、加载和管理模型文件
"""

import os
import glob
from pathlib import Path


class ModelManager:
    """模型管理器，负责扫描、加载和管理模型文件"""
    
    @staticmethod
    def scan_models(model_dirs):
        """扫描模型文件夹，返回模型名称和路径的字典"""
        models_dict = {}
        
        # 获取项目根目录
        root_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
        
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