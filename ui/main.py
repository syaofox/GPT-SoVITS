import os
import gradio as gr

from ui.utils import g_default_role
from ui.models import init_models
from ui.tabs.single_text import create_single_text_tab
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

        with gr.Tab("单条文本"):
            single_text_components = create_single_text_tab()

        with gr.Tab("角色管理合成"):
            role_management_components = create_role_management_tab()

        with gr.Tab("配置"):
            config_components = create_config_tab()

        # 添加使用说明
        gr.Markdown("""
        ## 使用说明
        1. **单条文本**：直接输入文本，选择角色和情绪，点击转换
        2. **文本文件**：可以选择以下两种方式之一：
           - 上传文本文件
           - 直接在文本框中输入多行文本
           支持以下格式：
           - `(角色)文本内容`
           - `(角色|情绪)文本内容`
           - 直接输入文本（需要设置默认角色）
        3. **强制角色**：忽略文本中的角色标记，全部使用指定角色
        4. **默认角色**：当文本没有指定角色时使用的角色
        5. **预处理文本**：将双引号内的文本作为对白（使用默认情绪），其他文本作为叙述
        """)

    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch(server_name="0.0.0.0", server_port=7878, inbrowser=True) 