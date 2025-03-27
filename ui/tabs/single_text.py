import gradio as gr

from ui.utils import LANGUAGE_OPTIONS, g_default_role
from ui.roles import list_roles, get_emotions, update_default_emotions
from ui.api_client import process_text


def create_single_text_tab():
    """创建单条文本标签页"""
    with gr.Row():
        text_input = gr.Textbox(
            label="输入文本", placeholder="请输入要转换的文本", lines=3
        )
        audio_output = gr.Audio(label="输出音频")

    with gr.Row():
        roles = list_roles()
        role = gr.Dropdown(
            label="选择角色",
            choices=[(role, role) for role in roles],
            value=g_default_role,
        )
        # 初始化情绪列表
        initial_emotions = get_emotions(g_default_role)
        emotion = gr.Dropdown(
            label="选择情绪",
            choices=[(e, e) for e in initial_emotions],
            value=initial_emotions[0] if initial_emotions else None,
        )
        text_lang = gr.Dropdown(
            label="文本语言",
            choices=LANGUAGE_OPTIONS,
            value="中文",
            type="value",
        )

        # 添加切分符号输入
        cut_punc_input = gr.Textbox(
            label="切分符号（可选）",
            placeholder="例如：,.。，…",
            value="",
        )

    convert_btn = gr.Button("转换")
    convert_btn.click(
        process_text,
        inputs=[text_input, role, emotion, text_lang, cut_punc_input],
        outputs=audio_output,
    )

    # 修改单行文本模式的角色切换事件
    role.change(
        update_default_emotions,
        inputs=[role],
        outputs=[emotion],
    )
    
    return {
        "role": role,
        "emotion": emotion,
        "text_lang": text_lang,
        "cut_punc": cut_punc_input
    } 