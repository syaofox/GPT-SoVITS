import gradio as gr

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
    import os
    file_name = os.path.basename(file_path)
    name_without_ext = os.path.splitext(file_name)[0]
    return name_without_ext


def create_role_management_tab():
    """创建角色管理标签页"""
    gpt_models, sovits_models = get_model_lists()
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 模型选择")
            gpt_model = gr.Dropdown(
                label="GPT模型",
                choices=gpt_models,
                value=gpt_models[0] if gpt_models else None,
                interactive=True
            )
            sovits_model = gr.Dropdown(
                label="SoVITS模型",
                choices=sovits_models,
                value=sovits_models[0] if sovits_models else None,
                interactive=True
            )
            refresh_models_btn = gr.Button("刷新模型列表")
        
        with gr.Column(scale=1):
            gr.Markdown("### 参考音频")
            ref_audio = gr.Audio(
                label="参考音频",
                type="filepath",
                interactive=True
            )
            prompt_text = gr.Textbox(
                label="参考音频文本",
                placeholder="参考音频对应的文本内容",
                lines=2
            )
            aux_refs = gr.File(
                label="辅助参考音频(可选)",
                file_count="multiple", 
                file_types=["audio"],
                type="filepath",
                interactive=True
            )
            
        with gr.Column(scale=1):
            gr.Markdown("### 语言设置")
            prompt_lang = gr.Dropdown(
                label="参考音频语言",
                choices=LANGUAGE_OPTIONS,
                value="中文",
                type="value"
            )
            text_lang = gr.Dropdown(
                label="目标文本语言",
                choices=LANGUAGE_OPTIONS,
                value="中文",
                type="value"
            )
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 目标文本")
            target_text = gr.Textbox(
                label="目标文本",
                placeholder="输入要合成的文本内容",
                lines=4
            )
            cut_punc = gr.Textbox(
                label="切分符号（可选）",
                placeholder="例如：,.。，",
                value="。！？：.!?:"
            )
            
        with gr.Column(scale=1):
            gr.Markdown("### 参数配置")
            with gr.Row():
                with gr.Column(scale=1):
                    speed = gr.Slider(
                        label="语速",
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1
                    )
                    ref_free = gr.Checkbox(
                        label="参考自由模式",
                        value=False
                    )
                    if_sr = gr.Checkbox(
                        label="使用超分辨率",
                        value=False
                    )
                
                with gr.Column(scale=1):
                    top_k = gr.Slider(
                        label="Top-K",
                        minimum=1,
                        maximum=30,
                        value=15,
                        step=1
                    )
                    top_p = gr.Slider(
                        label="Top-P",
                        minimum=0.1,
                        maximum=1.0,
                        value=1.0,
                        step=0.05
                    )
                    temperature = gr.Slider(
                        label="温度",
                        minimum=0.1,
                        maximum=2.0,
                        value=1.0,
                        step=0.1
                    )
                    sample_steps = gr.Slider(
                        label="采样步数",
                        minimum=16,
                        maximum=64,
                        value=32,
                        step=8
                    )
                    pause_second = gr.Slider(
                        label="停顿秒数",
                        minimum=0.1,
                        maximum=1.0,
                        value=0.3,
                        step=0.1
                    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 角色信息")
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
            
        with gr.Column(scale=1):
            gr.Markdown("### 操作")
            create_btn = gr.Button("新建/更新角色", variant="primary")
            role_list = gr.Dropdown(
                label="现有角色列表",
                choices=[(role, role) for role in list_roles()],
                value=g_default_role if g_default_role in list_roles() else None,
                type="value"
            )
            with gr.Row():
                load_role_btn = gr.Button("加载角色")
                delete_role_btn = gr.Button("删除角色", variant="stop")
            refresh_role_list_btn = gr.Button("刷新角色列表", variant="secondary")
        
        with gr.Column(scale=1):
            gr.Markdown("### 合成结果")
            synthesis_btn = gr.Button("合成测试", variant="primary")
            synthesis_output = gr.Audio(label="合成结果")
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
    
    # 从文件名自动提取参考文本
    ref_audio.change(
        fn=extract_text_from_filename,
        inputs=[ref_audio],
        outputs=[prompt_text]
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