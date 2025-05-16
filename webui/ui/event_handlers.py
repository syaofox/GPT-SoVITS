import gradio as gr

from models.logger import info, error
from services.prompt_service import PromptService
from models.text_utils import _split_text_by_speaker_and_lines
from models.audio_utils import normalize_audio, merge_audio, save_audio


class EventHandlers:
    def __init__(self, tts_service, prompt_service: PromptService):
        self.tts = tts_service
        self.prompt_service = prompt_service
        self.progress = gr.Progress()

    def get_character_list(self):
        return self.prompt_service.get_all_characters()

    def character_dropdown_change(self, selected_character):
        # 获取所选角色的所有情绪
        emotions = self.prompt_service.get_character_emotions(selected_character)
        # 设置默认值为第一个情绪，如果列表为空则为None
        default_emotion = emotions[0] if emotions else None

        # 获取该情绪的参考音频和提示文本
        default_ref_wav_path = None
        default_prompt_text = ""
        if default_emotion:
            prompt = self.prompt_service.get_prompt(selected_character, default_emotion)
            if prompt:
                default_ref_wav_path = prompt.ref_wav_path
                default_prompt_text = prompt.prompt_text

        # 返回更新
        return [
            gr.update(choices=emotions, value=default_emotion, interactive=True),
            gr.update(value=default_ref_wav_path),
            gr.update(value=default_prompt_text),
        ]

    def refresh_characters(self):
        """刷新角色列表和情绪列表"""
        # 重新加载所有提示词数据
        self.prompt_service.load_prompt_datas()
        # 获取角色列表
        characters = self.prompt_service.get_all_characters()
        # 设置默认值为第一个角色，如果列表为空则为None
        default_character = characters[0] if characters else None
        # 如果有默认角色，则获取该角色的情绪列表
        emotions = []
        default_emotion = None
        default_ref_wav_path = None
        default_prompt_text = ""
        if default_character:
            emotions = self.prompt_service.get_character_emotions(default_character)
            default_emotion = emotions[0] if emotions else None

            # 获取默认情绪的参考音频和提示文本
            if default_emotion:
                prompt = self.prompt_service.get_prompt(
                    default_character, default_emotion
                )
                if prompt:
                    default_ref_wav_path = prompt.ref_wav_path
                    default_prompt_text = prompt.prompt_text

        info(f"已刷新角色列表，共找到 {len(characters)} 个角色")
        if default_character:
            info(f"默认选择角色：{default_character}，共有 {len(emotions)} 种情绪")

        # 返回角色下拉框和情绪下拉框的更新
        return [
            gr.update(choices=characters, value=default_character, interactive=True),
            gr.update(choices=emotions, value=default_emotion, interactive=True),
            gr.update(value=default_ref_wav_path),
            gr.update(value=default_prompt_text),
        ]

    def prompt_dropdown_change(self, selected_emotion, selected_character):
        """情绪选择事件处理函数，加载选中情绪的参考音频和提示文本"""
        if not selected_character or not selected_emotion:
            info("未选择角色或情绪")
            return gr.update(), gr.update()

        prompt = self.prompt_service.get_prompt(selected_character, selected_emotion)
        if not prompt:
            info(f"未找到 {selected_character} 角色的 {selected_emotion} 情绪配置")
            return gr.update(), gr.update()

        # 返回参考音频路径和提示文本
        return [
            gr.update(value=prompt.ref_wav_path),
            gr.update(value=prompt.prompt_text),
        ]

    def clear_audio(self):
        """清空音频输出"""
        return gr.update(value=None, visible=True)

    def _set_progress(self, value, desc=""):
        if self.progress is not None:
            self.progress(value, desc=desc)

    def set_button_generating(self):
        """设置按钮为生成中状态"""
        return gr.update(value="生成中...", interactive=False)

    def gen_wavdata_togr(self, selected_emotion, selected_character, input_text_single):
        """生成语音数据并返回给 Gradio"""

        try:
            if not selected_character or not selected_emotion:
                info("未选择角色或情绪")
                return None, gr.update(value="生成语音", interactive=True)

            prompt = self.prompt_service.get_prompt(
                selected_character, selected_emotion
            )
            if not prompt:
                info(f"未找到 {selected_character} 角色的 {selected_emotion} 情绪配置")
                return None, gr.update(value="生成语音", interactive=True)

            # 根据文本内容分割成多个段落，支持多角色
            segments = _split_text_by_speaker_and_lines(
                input_text_single, selected_character, selected_emotion
            )
            if not segments:
                info("没有有效的文本内容")
                return None, gr.update(value="生成语音", interactive=True)

            # 计算总段落数
            total = len(segments)
            info(f"文本已分割为 {total} 个段落")

            # 创建音频段落列表
            audio_segments = []

            self._set_progress(0, "准备数据中...")

            # 处理每个段落
            for i, segment in enumerate(segments):
                # 获取当前段落的角色和情绪的提示配置
                current_prompt = self.prompt_service.get_prompt(
                    segment["speaker"], segment["emotion"]
                )
                if not current_prompt:
                    info(
                        f"未找到 {segment['speaker']} 角色的 {segment['emotion']} 情绪配置，使用默认配置"
                    )
                    current_prompt = prompt

                info(
                    f"处理 {segment['speaker']} 的 {segment['emotion']} 情绪文本: {segment['text']}"
                )

                # 将Prompt对象转换为字典
                prompt_dict = current_prompt.__dict__

                # 设置默认的文本语言为中文
                text_language = "中文"

                # 调用服务生成语音
                sr, audio_data = self.tts.generate_speech(
                    text=segment["text"],
                    text_language=text_language,
                    **prompt_dict,
                    process_callback=self._set_progress,
                    process_current_segment=i,
                    process_total_segment=total,
                )

                # 将音频数据和采样率存储在一个字典中
                audio_segments.append(
                    {
                        "audio": audio_data,
                        "sr": sr,
                        "speaker": segment["speaker"],
                        "emotion": segment["emotion"],
                    }
                )

            if not audio_segments:
                info("没有生成任何音频内容")
                return None, gr.update(value="生成语音", interactive=True)

            # 使用改进的归一化方法，传入音频段落字典列表
            normalized_segments = normalize_audio(audio_segments)
            # 合并所有段落，处理不同采样率
            combined_audio, final_sr = merge_audio(normalized_segments)
            info(f"合并后的音频采样率: {final_sr}")

            # 保存合并后的音频，使用合并后确定的采样率
            filename = save_audio(final_sr, combined_audio, segments)
            info(f"语音生成 -> {filename}")

            # 返回音频数据和重置按钮状态
            return (final_sr, combined_audio), gr.update(
                value="生成语音", interactive=True
            )

        except Exception as e:
            error(f"语音生成失败: {str(e)}")
            import traceback

            error(traceback.format_exc())
            return None, gr.update(value="生成语音", interactive=True)
