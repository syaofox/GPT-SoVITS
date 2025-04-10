import os
import time
import gradio as gr
import gc

import soundfile as sf
from ui.utils import LANGUAGE_OPTIONS, g_default_role
from ui.roles import list_roles, update_default_emotions, update_force_emotions, get_emotions
from ui.text_processing import preprocess_text
from ui.api_client import call_api, merge_audio_segments
from ui.utils import parse_line
from ui.models import init_models
from ui.roles import get_role_config


def get_formatted_filename(role_name: str, text_content: str) -> str:
    """生成格式化的文件名：角色名_时间戳_文本内容前20个字符
    
    Args:
        role_name: 角色名称
        text_content: 文本内容
    
    Returns:
        格式化的文件名
    """
    # 获取第一个非空行的前20个字符
    first_text = ""
    for line in text_content.strip().split("\n"):
        line = line.strip()
        if line:
            # 如果有角色标记，提取实际文本内容
            if line.startswith("(") and ")" in line:
                _, text = line.split(")", 1)
                first_text = text.strip()
            else:
                first_text = line
            break
    
    # 提取前20个字符，去除可能导致文件名问题的字符
    if first_text:
        short_text = first_text[:20].strip()
        # 替换不适合作为文件名的字符
        for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '\n', '\r', '\t']:
            short_text = short_text.replace(char, '_')
    else:
        short_text = "无文本"
        
    # 生成文件名
    return f"{role_name}_{int(time.time())}_{short_text}"


def process_text_content(
    text_content: str,
    force_role: str = "",
    default_role: str = "",
    force_emotion: str = "",
    default_emotion: str = "",
    text_lang: str = "中文",
    cut_punc: str = "",
    process_mode: str = "逐行处理",
    output_dir: str = "output",
) -> str:
    """处理文本内容"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 预先确定角色，用于文件名
    role_name = force_role if force_role and force_role != "无" else default_role
    
    # 使用新的文件名格式
    filename = get_formatted_filename(role_name, text_content)
    output_path = os.path.join(output_dir, f"{filename}.wav")

    try:
        # 如果是全文本处理或分段处理模式，使用默认角色和情绪处理整个文本
        if process_mode in ["全文本处理", "分段处理"]:
            if not default_role:
                raise gr.Error("全文本或分段处理模式必须设置默认角色")

            # 如果没有指定情绪，使用角色配置中的第一个情绪
            if not default_emotion or default_emotion == "无":
                import json
                from pathlib import Path
                role_path = Path("configs/roles") / f"{default_role}.json"
                with open(role_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                if "emotions" in config:
                    default_emotion = next(iter(config["emotions"].keys()))
                else:
                    raise gr.Error(f"角色 {default_role} 配置缺少情绪配置")

            print(f"{process_mode}模式 - 角色: {default_role} 情绪: {default_emotion}")
            
            # 处理文本，移除所有角色和情绪标记
            processed_text = ""
            lines = text_content.strip().split("\n")
            for line in lines:
                # 移除角色和情绪标记
                if line.startswith("(") and ")" in line:
                    _, text = line.split(")", 1)
                    processed_text += text.strip() + "\n"
                else:
                    processed_text += line.strip() + "\n"
            
            processed_text = processed_text.strip()
            
            # 初始化默认角色的模型
            init_models(default_role)
            
            # 获取配置
            role_config = get_role_config(default_role, default_emotion, text_lang)
            
            # 根据处理模式选择不同的处理方法
            if process_mode == "全文本处理":
                # 全文本处理模式，整个文本一次性处理
                audio_data = call_api(
                    processed_text, role_config, default_role, cut_punc=cut_punc
                )
                
                # 保存音频
                with open(output_path, "wb") as f:
                    f.write(audio_data)
                return output_path
            
            elif process_mode == "分段处理":
                # 分段处理模式，文本按段落分割，每段尽量不超过500字
                audio_segments = []
                
                # 保持段落完整性的分段处理
                paragraphs = processed_text.split("\n")
                current_segment = ""
                
                for paragraph in paragraphs:
                    # 如果当前段落加上当前片段会超过500字，先处理当前片段
                    if len(current_segment) + len(paragraph) > 500 and current_segment:
                        print(f"处理分段: {current_segment[:50]}... ({len(current_segment)}字)")
                        try:
                            audio_data = call_api(
                                current_segment, role_config, default_role, cut_punc=cut_punc
                            )
                            audio_segments.append(audio_data)
                        except Exception as e:
                            print(f"处理分段失败: {current_segment[:50]}..., 错误: {str(e)}")
                        current_segment = ""
                    
                    # 如果单个段落本身就超过500字，需要进一步分割
                    if len(paragraph) > 500:
                        # 尽量在标点符号处分割
                        sentence_endings = ["。", "！", "？", ".", "!", "?", ";", "；"]
                        last_cut = 0
                        for i in range(min(500, len(paragraph) - 1), 0, -1):
                            if i < len(paragraph) and paragraph[i] in sentence_endings:
                                last_cut = i + 1
                                break
                        
                        # 如果没有找到合适的分割点，就在500字处截断
                        if last_cut == 0:
                            last_cut = min(500, len(paragraph))
                        
                        # 处理分割的前半部分
                        if current_segment:
                            segment_to_process = current_segment + paragraph[:last_cut]
                        else:
                            segment_to_process = paragraph[:last_cut]
                            
                        print(f"处理分段: {segment_to_process[:50]}... ({len(segment_to_process)}字)")
                        try:
                            audio_data = call_api(
                                segment_to_process, role_config, default_role, cut_punc=cut_punc
                            )
                            audio_segments.append(audio_data)
                        except Exception as e:
                            print(f"处理分段失败: {segment_to_process[:50]}..., 错误: {str(e)}")
                        
                        # 将剩余部分作为新片段
                        current_segment = paragraph[last_cut:]
                    else:
                        # 将段落添加到当前片段
                        if current_segment:
                            current_segment += "\n" + paragraph
                        else:
                            current_segment = paragraph
                
                # 处理最后剩余的片段
                if current_segment:
                    print(f"处理最后分段: {current_segment[:50]}... ({len(current_segment)}字)")
                    try:
                        audio_data = call_api(
                            current_segment, role_config, default_role, cut_punc=cut_punc
                        )
                        audio_segments.append(audio_data)
                    except Exception as e:
                        print(f"处理分段失败: {current_segment[:50]}..., 错误: {str(e)}")
                
                if not audio_segments:
                    raise gr.Error("没有可处理的文本段落")
                
                # 合并音频段
                print("合并所有分段音频...")
                merged_audio, sample_rate = merge_audio_segments(audio_segments, add_silence=True)
                
                # 保存合并后的音频
                
                sf.write(output_path, merged_audio, sample_rate)
                del merged_audio, audio_segments  # 手动释放内存
                gc.collect()
                
                return output_path

        # 逐行处理模式（原有的处理逻辑）
        lines = text_content.strip().split("\n")
        audio_segments = []
        current_role = None
        last_role_name = None  # 记录最后使用的角色名

        for i, line in enumerate(lines):
            role_name, emotion, text = parse_line(line)

            # 使用强制角色或原始解析的角色或默认角色
            if force_role and force_role != "" and force_role != "无":
                role_name = force_role
            elif not role_name:
                role_name = default_role

            if not role_name:
                raise gr.Error("未指定角色且未设置默认角色")
                
            # 记录角色名，用于生成文件名
            last_role_name = role_name

            # 使用强制情绪或原始解析的情绪或默认情绪
            if force_emotion and force_emotion != "" and force_emotion != "无":
                emotion = force_emotion
            elif not emotion and default_emotion and default_emotion != "无":
                emotion = default_emotion

            # 如果角色改变，需要重新初始化模型
            if role_name != current_role:
                print(f"切换到角色: {role_name}")
                init_models(role_name)
                current_role = role_name

            if not text:  # 空行
                continue

            try:
                print(f"当前角色: {role_name} 当前情绪: {emotion} 当前文本: {text}")
                # 获取配置并调用API
                role_config = get_role_config(role_name, emotion, text_lang)
                audio_data = call_api(text, role_config, role_name, cut_punc=cut_punc)
                audio_segments.append(audio_data)
            except Exception as e:
                print(f"处理文本失败: {text}, 错误: {str(e)}")
                continue

        if not audio_segments:
            raise gr.Error("没有可处理的文本")
            
        # 更新文件名，使用最后使用的角色名
        if last_role_name and last_role_name != role_name:
            filename = get_formatted_filename(last_role_name, text_content)
            output_path = os.path.join(output_dir, f"{filename}.wav")

        # 合并音频
        merged_audio, sample_rate = merge_audio_segments(audio_segments, add_silence=(process_mode != "逐行处理"))

        # 保存合并后的音频
        sf.write(output_path, merged_audio, sample_rate)
        del merged_audio, audio_segments  # 手动释放内存
        gc.collect()

        return output_path
    except Exception as e:
        raise gr.Error(f"处理失败: {str(e)}")


def process_file(
    file: gr.components.File,
    force_role: str = "",
    default_role: str = "",
    force_emotion: str = "",
    default_emotion: str = "",
    text_lang: str = "中文",
    cut_punc: str = "",
    process_mode: str = "逐行处理",
    output_dir: str = "output",
) -> str:
    """处理文本文件"""
    if not file or not hasattr(file, "name"):
        raise gr.Error("请选择文件")

    with open(file.name, "r", encoding="utf-8") as f:
        text_content = f.read()

    return process_text_content(
        text_content,
        force_role,
        default_role,
        force_emotion,
        default_emotion,
        text_lang,
        cut_punc,
        process_mode,
        output_dir,
    )


def create_text_file_tab():
    """创建文本文件标签页"""
    with gr.Group():
        with gr.Row():
            with gr.Column(scale=2):
                text_content = gr.Textbox(
                    label="输入多行文本",
                    placeholder="每行一句话，支持以下格式：\n(角色)文本内容\n(角色|情绪)文本内容\n直接输入文本",
                    lines=15,
                    max_lines=15
                )
            with gr.Column(scale=1):
                file_input = gr.File(label="或上传文本文件", file_types=[".txt"])
                file_output = gr.Audio(label="输出音频",streaming=False)
    
    # 处理模式选择区域（放在上方突出显示）
    with gr.Group(elem_id="process_mode_group"):
        gr.Markdown("### 选择处理模式")
        with gr.Row():
            process_mode = gr.Radio(
                label="",
                choices=["逐行处理", "全文本处理", "分段处理"],
                value="全文本处理",
                elem_id="process_mode_radio"
            )
        
        with gr.Group(elem_id="process_mode_info"):
            mode_info = gr.Markdown("""
            **当前模式说明**：
            - **逐行处理**：逐行分析角色和情绪标记，适用于对话场景，每行文本单独处理
            - **全文本处理**：使用默认角色和情绪处理整个文本，适用于大段独白
            - **分段处理**：将文本分成多个段落（不超过500字），分别处理后合并，解决长文本问题
            """)

    with gr.Group():
        with gr.Row():
            roles = list_roles()  # 获取角色列表

            # 第一列：默认角色及其情绪
            with gr.Column():
                default_role = gr.Dropdown(
                    label="默认角色（必选）",
                    choices=[(role, role) for role in roles],
                    value=g_default_role,
                    type="value",
                )
                initial_emotions = get_emotions(g_default_role)
                default_emotion = gr.Dropdown(
                    label="默认情绪",
                    choices=[(e, e) for e in initial_emotions],
                    value=initial_emotions[0] if initial_emotions else None,
                    type="value",
                )

            # 第二列：强制角色及其情绪
            with gr.Column():
                force_role = gr.Dropdown(
                    label="强制使用角色（可选）",
                    choices=[("", "无")] + [(role, role) for role in roles],
                    value="",
                    type="value",
                )
                force_emotion = gr.Dropdown(
                    label="强制使用情绪（可选）",
                    choices=[("", "无")] + [(e, e) for e in initial_emotions],
                    value="",
                    type="value",
                )

            # 第三列：语言选择和切分符号
            with gr.Column():
                text_lang = gr.Dropdown(
                    label="文本语言",
                    choices=LANGUAGE_OPTIONS,
                    value="中文",
                    type="value",
                )
                cut_punc_input = gr.Textbox(
                    label="切分符号（可选）",
                    placeholder="例如：,.。，",
                    value="。！？：.!?:",
                )

    with gr.Group():
        with gr.Row():            
            process_text_btn = gr.Button("处理文本", variant="primary")
            preprocess_text_btn = gr.Button("预处理文本", variant="secondary")
            refresh_roles_btn = gr.Button("刷新角色列表", variant="secondary")
    
    # 添加使用说明
    
    gr.Markdown("""
    ## 使用说明
    1. **文本文件**：可以选择以下两种方式之一：
        - 上传文本文件
        - 直接在文本框中输入多行文本
    2. **处理模式**：
        - **逐行处理**：每行文本单独处理，支持行内角色和情绪标记
            - 格式：`(角色)文本内容` 或 `(角色|情绪)文本内容`
            - 适用场景：对话、剧本等多角色文本
        - **全文本处理**：使用默认角色和情绪处理整个文本
            - 适用场景：单一角色的大段文本，如小说、文章等
        - **分段处理**：将文本分段（每段不超过500字）处理后合并
            - 适用场景：大段文本，但想保持更好的语音连贯性
    3. **强制角色**：忽略文本中的角色标记，全部使用指定角色
    4. **默认角色**：当文本没有指定角色时使用的角色
    5. **预处理文本**：将双引号内的文本作为对白（使用默认情绪），其他文本作为叙述
    """)

    # 添加处理模式切换时的提示信息更新
    def update_mode_tips(mode):
        if mode == "逐行处理":
            return """
            **当前模式**：逐行处理
            - 每行文本独立处理，支持行内角色和情绪标记
            - 格式：`(角色)文本内容` 或 `(角色|情绪)文本内容`
            - 适用于对话、剧本等多角色文本
            - 提示：每行尽量保持在一句话以内，行与行之间会有停顿
            """
        elif mode == "全文本处理":
            return """
            **当前模式**：全文本处理
            - 使用默认角色和默认情绪处理整个文本
            - 系统会忽略文本中的角色和情绪标记
            - 适用于单一角色的大段独白文本
            - 提示：请确保已选择默认角色和情绪
            """
        else:  # 分段处理
            return """
            **当前模式**：分段处理
            - 将文本按段落分割，每段不超过500字
            - 使用默认角色和默认情绪处理
            - 系统会忽略文本中的角色和情绪标记
            - 适用于长文本处理，解决长文本处理问题
            - 提示：分段处理会保持段落完整性，并在自然停顿处分割
            """
    
    process_mode.change(
        fn=update_mode_tips,
        inputs=[process_mode],
        outputs=[mode_info]
    )

    process_text_btn.click(
        process_text_content,
        inputs=[
            text_content,
            force_role,
            default_role,
            force_emotion,
            default_emotion,
            text_lang,
            cut_punc_input,
            process_mode,
        ],
        outputs=file_output,
    )

    # 添加文件上传事件处理函数
    def load_file_content(file):
        if not file or not hasattr(file, "name"):
            return None
        try:
            with open(file.name, "r", encoding="utf-8") as f:
                content = f.read()
            return content
        except Exception as e:
            return f"文件读取错误：{str(e)}"

    file_input.change(
        load_file_content,
        inputs=[file_input],
        outputs=[text_content],
    )

    preprocess_text_btn.click(
        preprocess_text,
        inputs=[text_content, default_role, default_emotion],
        outputs=text_content,
    )

    # 添加刷新角色列表按钮的事件
    refresh_roles_btn.click(
        fn=lambda: gr.update(choices=[(role, role) for role in list_roles()]),
        inputs=[],
        outputs=[default_role]
    ).then(
        fn=update_default_emotions,
        inputs=[default_role],
        outputs=[default_emotion]
    ).then(
        fn=lambda: gr.update(choices=[("", "无")] + [(role, role) for role in list_roles()]),
        inputs=[],
        outputs=[force_role]
    ).then(
        fn=update_force_emotions,
        inputs=[force_role],
        outputs=[force_emotion]
    )
    
    return {
        "default_role": default_role,
        "default_emotion": default_emotion,
        "force_role": force_role,
        "force_emotion": force_emotion,
        "text_lang": text_lang,
        "cut_punc": cut_punc_input,
        "process_mode": process_mode,
        "text_content": text_content,
        "file_input": file_input
    } 