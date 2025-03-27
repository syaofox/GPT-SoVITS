import gradio as gr
import os

from ui.utils import LANGUAGE_OPTIONS, g_default_role
from ui.models import get_model_lists
from ui.roles import (
    list_roles, save_role_config, delete_role_config, 
    load_and_process_role_config
)
from ui.api_client import test_role_synthesis


def process_aux_refs(aux_refs_list):
    """处理辅助参考音频列表，提取文件路径"""
    if not aux_refs_list:
        return []
    
    processed_refs = []
    if isinstance(aux_refs_list, list):
        for item in aux_refs_list:
            if isinstance(item, dict) and "name" in item:
                processed_refs.append(item["name"])
            elif isinstance(item, str):
                processed_refs.append(item)
    elif isinstance(aux_refs_list, dict) and "name" in aux_refs_list:
        processed_refs.append(aux_refs_list["name"])
    elif isinstance(aux_refs_list, str):
        processed_refs.append(aux_refs_list)
    
    return processed_refs


def extract_text_from_filename(file_path):
    """从文件名自动提取参考文本"""
    if not file_path:
        return ""
    file_name = os.path.basename(file_path)
    name_without_ext = os.path.splitext(file_name)[0]
    return name_without_ext


def preview_audio_if_exists(file_path):
    """如果文件存在则返回用于预览的路径"""
    if file_path and os.path.isfile(file_path):
        return file_path
    return None


def create_role_management_tab():
    """创建角色管理标签页"""
    # 辅助函数 - 移到函数开始处
    def html_center(text, label='p'):
        return f"""<div style="text-align: center; margin: 0; padding: 0;">
                  <{label} style="margin: 0; padding: 0;">{text}</{label}>
                  </div>"""
    
    def html_left(text, label='p'):
        return f"""<div style="text-align: left; margin: 0; padding: 0;">
                  <{label} style="margin: 0; padding: 0;">{text}</{label}>
                  </div>"""
    
    gpt_models, sovits_models = get_model_lists()
    
    with gr.Blocks(title="GPT-SoVITS 角色管理"):
        
        # 模型选择区域
        with gr.Group():
            gr.Markdown(html_center("模型选择", 'h3'))
            with gr.Row():
                gpt_model = gr.Dropdown(
                    label="GPT模型列表",
                    choices=gpt_models,
                    value=gpt_models[0] if gpt_models else None,
                    interactive=True,
                    scale=14
                )
                sovits_model = gr.Dropdown(
                    label="SoVITS模型列表",
                    choices=sovits_models,
                    value=sovits_models[0] if sovits_models else None,
                    interactive=True,
                    scale=14
                )
                refresh_models_btn = gr.Button("刷新模型列表", variant="primary", scale=14)
            
            # 参考音频区域
            gr.Markdown(html_center("参考音频", 'h3'))
            with gr.Row():
                with gr.Column(scale=13):
                    ref_audio = gr.Textbox(
                        label="请输入3~10秒内参考音频的文件路径，超过会报错！",
                        placeholder="输入音频文件的完整路径",
                        type="text"
                    )
                    audio_preview = gr.Audio(
                        label="音频预览",
                        type="filepath",
                        visible=True,
                        interactive=False
                    )
                with gr.Column(scale=13):
                    ref_free = gr.Checkbox(
                        label="开启无参考文本模式。不填参考文本亦相当于开启。v3暂不支持该模式，使用了会报错。",
                        value=False,
                        interactive=True,
                        show_label=True,
                        scale=1
                    )
                    gr.Markdown(html_left("使用无参考文本模式时建议使用微调的GPT<br>听不清参考音频说的啥(不晓得写啥)可以开。开启后无视填写的参考文本。"))
                    prompt_text = gr.Textbox(
                        label="参考音频的文本",
                        value="",
                        lines=5,
                        max_lines=5,
                        scale=1
                    )
                with gr.Column(scale=14):
                    prompt_lang = gr.Dropdown(
                        label="参考音频的语种",
                        choices=LANGUAGE_OPTIONS,
                        value="中文",
                        scale=1
                    )
                    text_lang = gr.Dropdown(
                        label="目标合成的语种(建议选'中文')",
                        choices=LANGUAGE_OPTIONS,
                        value="中文",
                        scale=1
                    )
                    aux_refs = gr.File(
                        label="可选项：通过拖拽多个文件上传多个参考音频（建议同性），平均融合他们的音色。",
                        file_count="multiple",
                        file_types=["audio"],
                        type="filepath"
                    )
                    sample_steps = gr.Radio(
                        label="采样步数,如果觉得电,提高试试,如果觉得慢,降低试试",
                        value=32,
                        choices=[4, 8, 16, 32, 64, 128],
                        visible=True
                    )
                    if_sr = gr.Checkbox(
                        label="启用超采样提高语音质量(会增加延迟)",
                        value=False,
                        interactive=True
                    )
            
            # 合成区域
            gr.Markdown(html_center("参数区", 'h3'))
            with gr.Row():
                with gr.Column(scale=13):
                    target_text = gr.Textbox(
                        label="需要合成的文本",
                        value="",
                        lines=22,
                        max_lines=22
                    )
                with gr.Column(scale=7):
                    cut_punc = gr.Textbox(
                        label="断句符号",
                        placeholder="例如：,.。，",
                        value="。！？：.!?:"
                    )
                    speed = gr.Slider(
                        minimum=0.6,
                        maximum=1.65,
                        step=0.05,
                        label="语速",
                        value=1.0,
                        interactive=True
                    )
                    pause_second = gr.Slider(
                        minimum=0.1,
                        maximum=0.5,
                        step=0.01,
                        label="句间停顿秒数",
                        value=0.3,
                        interactive=True
                    )
                    
                    gr.Markdown(html_center("GPT推理参数(不懂就用默认)："))
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=100,
                        step=1,
                        label="top_k",
                        value=15,
                        interactive=True
                    )
                    top_p = gr.Slider(
                        minimum=0,
                        maximum=1,
                        step=0.05,
                        label="top_p",
                        value=1.0,
                        interactive=True
                    )
                    temperature = gr.Slider(
                        minimum=0,
                        maximum=1,
                        step=0.05,
                        label="temperature",
                        value=1.0,
                        interactive=True
                    )
            
            # 合成按钮与结果
            with gr.Row():
                synthesis_btn = gr.Button("合成语音", variant="primary", size="lg", scale=25)
                synthesis_output = gr.Audio(label="输出的语音", scale=14)
                
            
            # 角色管理区域
            gr.Markdown(html_center("角色管理", 'h3'))
            with gr.Row():
                with gr.Column(scale=1):
                    role_name = gr.Textbox(
                        label="角色名称",
                        placeholder="输入要保存的角色名称",
                        value=""
                    )
                    description = gr.Textbox(
                        label="角色描述(可选)",
                        placeholder="输入角色描述，如：性别，声音特点等",
                        lines=2
                    )
                    create_btn = gr.Button("新建/更新角色", variant="primary")
                
                with gr.Column(scale=1):
                    role_list = gr.Dropdown(
                        label="现有角色列表",
                        choices=[(role, role) for role in list_roles()],
                        value=g_default_role if g_default_role in list_roles() else None,
                        type="value"
                    )
                    with gr.Row():
                        load_role_btn = gr.Button("加载角色")
                        delete_role_btn = gr.Button("删除角色", variant="stop")
                        refresh_role_list_btn = gr.Button("刷新列表", variant="secondary")
                    with gr.Row():
                        status_text = gr.Markdown("")
    # 事件绑定
    
    # 刷新模型列表
    refresh_models_btn.click(
        fn=lambda: (
            gr.update(choices=get_model_lists()[0]),
            gr.update(choices=get_model_lists()[1])
        ),
        inputs=[],
        outputs=[gpt_model, sovits_model]
    )
    
    # 从文件名自动提取参考文本并预览音频
    ref_audio.change(
        fn=extract_text_from_filename,
        inputs=[ref_audio],
        outputs=[prompt_text]
    ).then(
        fn=preview_audio_if_exists,
        inputs=[ref_audio],
        outputs=[audio_preview]
    )
    
    # 保存角色配置
    create_btn.click(
        fn=lambda role_name, gpt_model, sovits_model, ref_audio, prompt_text, prompt_lang, text_lang, 
               speed, ref_free, if_sr, top_k, top_p, temperature, sample_steps, pause_second, 
               description, aux_refs: save_role_config(
            role_name, gpt_model, sovits_model, ref_audio, prompt_text, prompt_lang, text_lang,
            speed, ref_free, if_sr, top_k, top_p, temperature, sample_steps, pause_second, 
            description, process_aux_refs(aux_refs)
        ),
        inputs=[
            role_name, gpt_model, sovits_model, 
            ref_audio, prompt_text, prompt_lang, text_lang,
            speed, ref_free, if_sr, top_k, top_p, 
            temperature, sample_steps, pause_second, description, aux_refs
        ],
        outputs=[status_text]
    ).then(
        fn=lambda: (
            gr.update(choices=[(role, role) for role in list_roles()])
        ),
        inputs=[],
        outputs=[role_list]
    )
    
    # 合成测试
    synthesis_btn.click(
        fn=lambda target_text, gpt_model, sovits_model, ref_audio, prompt_text, 
                prompt_lang, text_lang, speed, ref_free, if_sr, top_k, top_p,
                temperature, sample_steps, cut_punc, role_name, aux_refs: test_role_synthesis(
            target_text, gpt_model, sovits_model, ref_audio, prompt_text, 
            prompt_lang, text_lang, speed, ref_free, if_sr, top_k, top_p,
            temperature, sample_steps, cut_punc, role_name, process_aux_refs(aux_refs)
        ),
        inputs=[
            target_text, gpt_model, sovits_model,
            ref_audio, prompt_text, prompt_lang, text_lang,
            speed, ref_free, if_sr, top_k, top_p,
            temperature, sample_steps, cut_punc, role_name, aux_refs
        ],
        outputs=[synthesis_output]
    )

    # 加载角色配置
    load_role_btn.click(
        fn=lambda role: load_and_process_role_config(role, process_aux_refs),
        inputs=[role_list],
        outputs=[
            gpt_model, sovits_model, ref_audio, prompt_text, aux_refs,
            prompt_lang, text_lang, speed, ref_free, if_sr,
            top_k, top_p, temperature, sample_steps, pause_second,
            description, status_text
        ]
    )
    
    # 删除角色配置
    delete_role_btn.click(
        fn=delete_role_config,
        inputs=[role_list],
        outputs=[status_text]
    ).then(
        fn=lambda: (
            gr.update(choices=[(role, role) for role in list_roles()])
        ),
        inputs=[],
        outputs=[role_list]
    )
    
    # 刷新角色列表
    refresh_role_list_btn.click(
        fn=lambda: (
            gr.update(choices=[(role, role) for role in list_roles()])
        ),
        inputs=[],
        outputs=[role_list]
    )
    
    return {
        "role_list": role_list,
        "gpt_model": gpt_model,
        "sovits_model": sovits_model,
        "ref_audio": ref_audio,
        "prompt_text": prompt_text,
        "aux_refs": aux_refs,
        "prompt_lang": prompt_lang,
        "text_lang": text_lang,
        "speed": speed,
        "ref_free": ref_free,
        "if_sr": if_sr,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "sample_steps": sample_steps,
        "pause_second": pause_second,
        "description": description,
        "role_name": role_name,
        "target_text": target_text,
        "cut_punc": cut_punc
    } 