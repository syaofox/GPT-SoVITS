import re
from pathlib import Path
import gradio as gr


def preprocess_text(text_content: str, default_role: str, default_emotion: str) -> str:
    """预处理文本，提取双引号或「」内容作为对白，其他作为叙述
    
    Args:
        text_content: 原始文本内容
        default_role: 默认角色
        default_emotion: 默认情绪
        
    Returns:
        处理后的文本
    """
    if not default_role:
        raise gr.Error("请先选择默认角色")
    
    # 获取角色配置
    role_path = Path("configs/roles") / f"{default_role}.json"
    if not role_path.exists():
        raise gr.Error(f"找不到角色配置文件: {default_role}")
    
    # 获取情绪列表
    import json
    with open(role_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # 获取情绪列表
    if "emotions" not in config:
        raise gr.Error(f"角色 {default_role} 配置缺少情绪配置")
    
    emotions = list(config["emotions"].keys())
    
    # 如果没有指定默认情绪，使用第一个情绪
    if not default_emotion:
        default_emotion = emotions[0]
    
    # 处理文本
    lines = text_content.strip().split("\n")
    processed_lines = []
    
    for line in lines:
        if not line.strip():
            processed_lines.append("")
            continue
        
        # 先处理双引号包围的内容
        if '"' in line:
            # 查找所有双引号包围的内容
            dialogue_parts = re.findall(r'"([^"]*)"', line)
            
            if dialogue_parts:
                # 处理包含对白的行
                remaining = line
                for part in dialogue_parts:
                    # 分割成对白前、对白、对白后三部分
                    before, remaining = remaining.split(f'"{part}"', 1)
                    
                    # 处理对白前的叙述部分(如果有)，不添加角色标记
                    if before.strip():
                        processed_lines.append(f"{before.strip()}")
                    
                    # 处理对白部分，添加角色和情绪
                    processed_lines.append(f"({default_role}|{default_emotion})\"{part}\"")
                
                # 处理最后一个对白后的叙述部分(如果有)，不添加角色标记
                if remaining.strip():
                    processed_lines.append(f"{remaining.strip()}")
                
                continue  # 已处理完这一行，继续下一行
        
        # 处理「」包围的内容
        if '「' in line and '」' in line:
            # 查找所有「」包围的内容
            dialogue_parts = re.findall(r'「([^」]*)」', line)
            
            if dialogue_parts:
                # 处理包含对白的行
                remaining = line
                for part in dialogue_parts:
                    # 分割成对白前、对白、对白后三部分
                    before, remaining = remaining.split(f'「{part}」', 1)
                    
                    # 处理对白前的叙述部分(如果有)，不添加角色标记
                    if before.strip():
                        processed_lines.append(f"{before.strip()}")
                    
                    # 处理对白部分，添加角色和情绪，转换为双引号格式
                    processed_lines.append(f"({default_role}|{default_emotion})\"{part}\"")
                
                # 处理最后一个对白后的叙述部分(如果有)，不添加角色标记
                if remaining.strip():
                    processed_lines.append(f"{remaining.strip()}")
                
                continue  # 已处理完这一行，继续下一行

         # 处理"包围的内容
        if '"' in line and '"' in line:
            # 查找所有"包围的内容
            dialogue_parts = re.findall(r'"([^"]*)"', line)
            
            if dialogue_parts:
                # 处理包含对白的行
                remaining = line
                for part in dialogue_parts:
                    # 分割成对白前、对白、对白后三部分
                    before, remaining = remaining.split(f'"{part}"', 1)
                    
                    # 处理对白前的叙述部分(如果有)，不添加角色标记
                    if before.strip():
                        processed_lines.append(f"{before.strip()}")
                    
                    # 处理对白部分，添加角色和情绪，转换为双引号格式
                    processed_lines.append(f"({default_role}|{default_emotion})\"{part}\"")
                
                # 处理最后一个对白后的叙述部分(如果有)，不添加角色标记
                if remaining.strip():
                    processed_lines.append(f"{remaining.strip()}")
                
                continue  # 已处理完这一行，继续下一行
        
        # 没有对白，整行作为叙述，不添加任何标记
        processed_lines.append(f"{line.strip()}")
    
    return "\n".join(processed_lines) 