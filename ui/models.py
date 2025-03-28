import os
import json
from pathlib import Path
from typing import List, Tuple

import requests
import gradio as gr

from ui.utils import API_URL


def get_model_paths_from_role(role: str) -> Tuple[str, str]:
    """从角色配置文件中获取模型路径"""
    from ui.utils import g_config  # 避免循环导入
    
    role_path = Path("configs/roles") / f"{role}.json"
    if not role_path.exists():
        raise ValueError(f"找不到角色配置文件: {role}")

    with open(role_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 获取模型路径，优先使用角色配置中的路径
    gpt_path = config.get("gpt_path")
    sovits_path = config.get("sovits_path")

    # 如果角色配置中没有指定，则使用全局配置
    if not gpt_path:
        gpt_path = g_config.gpt_path or g_config.pretrained_gpt_path
    if not sovits_path:
        sovits_path = g_config.sovits_path or g_config.pretrained_sovits_path

    # 确保路径是绝对路径
    gpt_path = os.path.abspath(gpt_path)
    sovits_path = os.path.abspath(sovits_path)

    # 检查模型文件是否存在
    if not os.path.exists(gpt_path):
        raise ValueError(f"找不到GPT模型: {gpt_path}")
    if not os.path.exists(sovits_path):
        raise ValueError(f"找不到SoVITS模型: {sovits_path}")

    return gpt_path, sovits_path


def init_models(role: str) -> Tuple[str, str]:
    """初始化模型"""
    # 从角色配置获取模型路径
    gpt_path, sovits_path = get_model_paths_from_role(role)
    print(f"正在设置模型 - 角色: {role}")
    print(f"- GPT模型: {gpt_path}")
    print(f"- SoVITS模型: {sovits_path}")

    # 设置模型
    try:
        response = requests.post(
            f"{API_URL}/set_model",
            json={
                "gpt_model_path": gpt_path,
                "sovits_model_path": sovits_path,
            },
            timeout=30,
        )
        if response.status_code != 200:
            error_msg = response.text
            try:
                error_data = response.json()
                if "message" in error_data:
                    error_msg = error_data["message"]
            except Exception:
                pass
            raise gr.Error(f"设置模型失败: {error_msg}")
    except requests.exceptions.RequestException as e:
        raise gr.Error(f"API请求失败: {str(e)}")

    return gpt_path, sovits_path


def get_model_lists() -> Tuple[List[str], List[str]]:
    """获取GPT和SoVITS模型列表"""
    # GPT模型列表
    gpt_models = []
    gpt_roots = ["GPT_weights", "GPT_weights_v2", "GPT_weights_v3"]
    gpt_default_model = "GPT_SoVITS/pretrained_models/s1v3.ckpt"
    for root in gpt_roots:
        if os.path.exists(root):
            for name in os.listdir(root):
                if name.endswith(".ckpt"):
                    gpt_models.append(f"{root}/{name}")
    
    gpt_models.insert(0, gpt_default_model)

    # SoVITS模型列表
    sovits_models = []
    sovits_roots = ["SoVITS_weights", "SoVITS_weights_v2", "SoVITS_weights_v3"]
    sovits_default_model = "GPT_SoVITS/pretrained_models/s2Gv3.pth"
    for root in sovits_roots:
        if os.path.exists(root):
            for name in os.listdir(root):
                if name.endswith(".pth"):
                    sovits_models.append(f"{root}/{name}")

    sovits_models.insert(0, sovits_default_model)
    
    return sorted(gpt_models), sorted(sovits_models) 