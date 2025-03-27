import gradio as gr

from ui.utils import load_word_replace_config, save_word_replace_config


def create_config_tab():
    """创建配置标签页"""
    with gr.Tab("词语替换"):
        word_replace_text = gr.TextArea(
            label="词语替换配置",
            value=load_word_replace_config(),
            lines=20,
            max_lines=50
        )
        
        with gr.Row():
            save_word_replace_btn = gr.Button("保存词语替换配置")
            refresh_word_replace_btn = gr.Button("刷新")
        
        save_word_replace_btn.click(
            fn=save_word_replace_config,
            inputs=[word_replace_text],
            outputs=[gr.Markdown("保存成功！")]
        )
        
        refresh_word_replace_btn.click(
            fn=load_word_replace_config,
            inputs=[],
            outputs=[word_replace_text]
        )
    
    return {
        "word_replace_text": word_replace_text
    } 