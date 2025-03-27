import os
import gradio as gr

from ui.utils import g_default_role
from ui.models import init_models
from ui.tabs.text_file import create_text_file_tab
from ui.tabs.role_management import create_role_management_tab
from ui.tabs.config import create_config_tab


def create_ui():
    """创建UI界面"""
    with gr.Blocks(title="GPT-SoVITS API推理") as app:
        gr.Markdown("# GPT-SoVITS 文本转语音 (API版)")

        # 获取默认角色
        from ui.roles import list_roles
        roles = list_roles()

        # 确保默认角色在角色列表中，否则使用第一个角色
        global g_default_role
        g_default_role = roles[0] if g_default_role not in roles else g_default_role

        # 初始化默认角色的模型
        gpt_path, sovits_path = init_models(g_default_role)

        # 创建标签页
        with gr.Tab("文本文件"):
            text_file_components = create_text_file_tab()

        with gr.Tab("角色管理合成"):
            role_management_components = create_role_management_tab()

        with gr.Tab("配置替换音"):
            config_components = create_config_tab()

       
    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch(server_name="0.0.0.0", server_port=7878, inbrowser=True) 