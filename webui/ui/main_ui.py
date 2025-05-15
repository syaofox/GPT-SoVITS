import gradio as gr

js_func = """
function set_dark_theme() {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""
PLACEHOLDER_TEXT = """请输入目标文本
支持多角色，选中的角色为默认角色，格式：

<角色名1>
文本内容段落1

<角色名2>
文本内容段落2
"""


class MainUI:
    def __init__(self):
        pass

    def build(self, event_handlers):
        with gr.Blocks(js=js_func) as demo:
            with gr.Tab("音频生成"):
                # 获取角色列表和默认值
                character_list = event_handlers.get_character_list()
                default_character = character_list[0] if character_list else None

                # 预加载默认角色的情绪列表
                emotion_list = []
                default_emotion = None
                default_ref_wav_path = None  # 先声明变量
                default_prompt_text = ""
                if default_character:
                    emotion_list = event_handlers.prompt_service.get_character_emotions(
                        default_character
                    )
                    default_emotion = emotion_list[0] if emotion_list else None

                    # 预加载默认情绪的参考音频和提示文本
                    if default_emotion:
                        prompt = event_handlers.prompt_service.get_prompt(
                            default_character, default_emotion
                        )
                        if prompt:
                            default_ref_wav_path = prompt.ref_wav_path
                            default_prompt_text = prompt.prompt_text

                with gr.Row():
                    with gr.Column():
                        prompt_audio = gr.Audio(
                            label="参考音频",
                            visible=True,
                            key="output_audio",
                            streaming=True,
                            value=default_ref_wav_path,
                        )

                        prompt_text_box = gr.Textbox(
                            label="参考文本",
                            interactive=False,
                            lines=3,
                            value=default_prompt_text,
                        )

                    with gr.Column():
                        character_dropdown = gr.Dropdown(
                            choices=character_list,
                            label="选择角色",
                            value=default_character,
                        )

                        prompt_dropdown = gr.Dropdown(
                            choices=emotion_list,
                            label="选择情绪",
                            value=default_emotion,
                            interactive=True,
                        )

                        refresh_button = gr.Button("刷新")

                input_text_single = gr.TextArea(
                    label="请输入目标文本",
                    key="input_text_single",
                    placeholder=PLACEHOLDER_TEXT,
                )

                gen_button = gr.Button(
                    "生成语音", key="gen_button", interactive=True, variant="primary"
                )
                output_audio = gr.Audio(
                    label="生成结果",
                    visible=True,
                    key="output_audio",
                    streaming=True,
                )

            # 下拉框选择事件（角色切换）
            character_dropdown.change(
                fn=event_handlers.character_dropdown_change,
                inputs=[character_dropdown],
                outputs=[prompt_dropdown, prompt_audio, prompt_text_box],
            )

            # 情绪选择事件
            prompt_dropdown.change(
                fn=event_handlers.prompt_dropdown_change,
                inputs=[prompt_dropdown, character_dropdown],
                outputs=[prompt_audio, prompt_text_box],
            )

            # 刷新按钮点击事件
            refresh_button.click(
                fn=event_handlers.refresh_characters,
                inputs=[],
                outputs=[
                    character_dropdown,
                    prompt_dropdown,
                    prompt_audio,
                    prompt_text_box,
                ],
            )

            # 生成语音按钮点击事件
            gen_button.click(
                fn=event_handlers.set_button_generating,
                inputs=[],
                outputs=[gen_button],
            ).then(
                fn=event_handlers.clear_audio,
                inputs=[],
                outputs=[output_audio],
            ).then(
                fn=event_handlers.gen_wavdata_togr,
                inputs=[
                    prompt_dropdown,
                    character_dropdown,
                    input_text_single,
                ],
                outputs=[output_audio, gen_button],
            )

        return demo
