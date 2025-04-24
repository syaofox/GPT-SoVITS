import os
import sys

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))


import platform
import signal
from datetime import timedelta
from pathlib import Path
from subprocess import Popen

import gradio as gr
import psutil
import pysrt
from pydub import AudioSegment
from tqdm import tqdm

from config import is_share, python_exec, webui_port_subfix
from tools import my_utils
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto()

p_label = None


def kill_proc_tree(pid, including_parent=True):
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        # Process already terminated
        return

    children = parent.children(recursive=True)
    for child in children:
        try:
            os.kill(child.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass
    if including_parent:
        try:
            os.kill(parent.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass


system = platform.system()


def kill_process(pid):
    if system == "Windows":
        cmd = "taskkill /t /f /pid %s" % pid
        os.system(cmd)
    else:
        kill_proc_tree(pid)


def change_label(if_label, path_list):
    global p_label
    if if_label is True and p_label is None:
        path_list = my_utils.clean_path(path_list)
        cmd = (
            '"%s" tools/subfix_webui.py --load_list "%s" --webui_port %s --is_share %s'
            % (python_exec, path_list, webui_port_subfix, is_share)
        )
        yield i18n("打标工具WebUI已开启")
        print(cmd)
        p_label = Popen(cmd, shell=True)
    elif if_label is False and p_label is not None:
        kill_process(p_label.pid)
        p_label = None
        yield i18n("打标工具WebUI已关闭")


def asr_cut(raw_audio, input_srt, min_seconds, max_seconds, output_dir, languate="ZH"):
    # temp_path = Path(tempfile.mkdtemp(dir='TEMP'))
    raw_audio = my_utils.clean_path(raw_audio)
    input_srt = my_utils.clean_path(input_srt)
    output_dir = my_utils.clean_path(output_dir)

    main_audio = AudioSegment.from_file(raw_audio)
    main_audio = main_audio.set_channels(1)
    subs = pysrt.open(input_srt)

    output_path = Path(output_dir)
    asr_list_file = output_path / "raw_cut.list"
    raw_cut_path = output_path / "raw_cut"
    raw_cut_path.mkdir(parents=True, exist_ok=True)

    asr_list = []

    prefix_text = Path(raw_audio).stem

    for i, sub in tqdm(enumerate(subs)):
        start = sub.start.ordinal  # 字幕开始时间，单位为毫秒
        end = sub.end.ordinal  # 字幕结束时间，单位为毫秒
        text = sub.text
        # 从音频中提取对应的片段
        if end - start < min_seconds * 1000 or end - start > max_seconds * 1000:
            tqdm.write(f"skip:{text}")
            continue

        sub_wav = main_audio[start:end]
        sub_wav_fname = raw_cut_path / f"{prefix_text}_{i:09d}.wav"

        asr_list.append(
            f"{sub_wav_fname}|{raw_cut_path.stem}|{languate.upper()}|{text}"
        )

        sub_wav.export(sub_wav_fname, format="wav")

    with open(asr_list_file, "w", encoding="utf8") as f:
        for i in asr_list:
            f.write(i + "\n")

    return "切分"


def format_timedelta(td):
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = td.microseconds // 1000
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def merge_audio_with_subtitles(txt_file_path, output_dir):
    yield gr.update(visible=True, interactive=False)

    txt_file_path = my_utils.clean_path(txt_file_path)
    output_dir = my_utils.clean_path(output_dir)

    # 读取txt文件内容
    with open(txt_file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # 解析txt内容
    files_and_texts = [line.strip().split("|") for line in lines]

    name = files_and_texts[0][1]

    output_audio_path = os.path.join(output_dir, f"{name}.wav")

    output_srt_path = os.path.join(output_dir, f"{name}.srt")

    # 创建srt文件内容的列表
    srt_content = []
    start_time = timedelta()  # 使用timedelta来表示时间

    # 创建一个空的音频对象
    combined_audio = AudioSegment.empty()

    # 静音段，500毫秒
    silence = AudioSegment.silent(duration=500)

    # 处理每个wav文件
    for index, (file_path, speaker, language, text) in enumerate(files_and_texts):
        # 读取当前音频文件
        audio = AudioSegment.from_wav(file_path)
        combined_audio += audio
        duration = timedelta(milliseconds=len(audio))

        # 计算结束时间
        end_time = start_time + duration

        # 创建srt条目
        start_str = format_timedelta(start_time)
        end_str = format_timedelta(end_time)
        srt_entry = f"{index + 1}\n{start_str} --> {end_str}\n{text}\n"
        srt_content.append(srt_entry)
        srt_content.append("\n")
        print(srt_entry)

        # 更新起始时间：加上当前音频时长和500毫秒静音
        start_time = end_time + timedelta(milliseconds=500)
        combined_audio += silence

    # 保存合并后的音频文件
    combined_audio.export(output_audio_path, format="wav")

    # 保存srt文件
    with open(output_srt_path, "w", encoding="utf-8") as srt_file:
        srt_file.writelines(srt_content)

    yield gr.update(visible=True, interactive=True)


def generate_list_from_wav(wav_dir):
    wav_dir = my_utils.clean_path(wav_dir)
    wav_path = Path(wav_dir)
    
    if not wav_path.exists():
        return "目录不存在"
    
    list_file = wav_path.parent / f"{wav_path.stem}.list"
    list_content = []
    
    for wav_file in wav_path.glob("*.wav"):
        dir_name = wav_path.stem
        text = wav_file.stem  # 使用文件名作为文本
        list_content.append(f"{wav_file}|{dir_name}|ZH|{text}")
    
    with open(list_file, "w", encoding="utf8") as f:
        f.write("\n".join(list_content))
    
    return f"已生成list文件: {list_file}"


def rename_audio_and_update_list(list_file, old_path, new_path, line_number):
    try:
        # 确保路径是字符串
        old_path_str = str(old_path)
        new_path_str = str(new_path)
        
        # 先尝试重命名音频文件
        os.rename(old_path_str, new_path_str)
        
        # 只有重命名成功后才更新list文件
        with open(list_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 更新对应行的路径
        parts = lines[line_number].rstrip('\n').split('|')
        parts[0] = new_path_str
        lines[line_number] = '|'.join(parts) + '\n'
        
        with open(list_file, 'w', encoding='utf-8') as f:
            f.writelines(lines)
            
        return True
    except Exception as e:
        print(f"重命名失败: {e}")
        return False


def rename_wav_from_list(list_file_path):
    list_file_path = my_utils.clean_path(list_file_path)
    if not Path(list_file_path).exists():
        return "list文件不存在"
    
    renamed_count = 0
    updated_lines = []
    
    with open(list_file_path, "r", encoding="utf8") as f:
        lines = [line.rstrip('\n') for line in f]  # 保留原始行内容
    
    for line_number, line in enumerate(lines):
        try:
            if not line.strip():
                updated_lines.append(line)
                continue
                
            wav_path, speaker, language, text = line.split("|")
            original_path = Path(my_utils.clean_path(wav_path))
            
            if not original_path.exists():
                print(f"文件不存在: {original_path}")
                updated_lines.append(line)
                continue
                
            # # 处理文本作为文件名（保留中文标点）
            # safe_text = "".join(c for c in text if c.isalnum() or c in (' ', '_', '-') or '\u4e00' <= c <= '\u9fff' or c in '…，。！？、；：""（）《》【】')
            # safe_text = safe_text.strip()[:50]  # 限制文件名长度

            safe_text = text.strip()
            
            # 生成新路径
            new_path = original_path.parent / f"{safe_text}.wav"
            
            # 避免文件名冲突
            counter = 1
            while new_path.exists():
                new_path = original_path.parent / f"{safe_text}_{counter}.wav"
                counter += 1
                
            # 重命名文件
            success = rename_audio_and_update_list(list_file_path, original_path, new_path, line_number)
            if not success:
                print(f"跳过更新 {list_file_path} 的第 {line_number+1} 行")
                updated_lines.append(line)  # 保留原行
                continue
            
            renamed_count += 1
            
            # 更新list中的路径
            updated_line = f"{new_path}|{speaker}|{language}|{text}"
            updated_lines.append(updated_line)
            
        except Exception as e:
            print(f"处理行出错: {line}, 错误: {str(e)}")
            updated_lines.append(line)  # 出错时保留原行
    
    # 更新list文件
    with open(list_file_path, "w", encoding="utf8") as f:
        f.write("\n".join(updated_lines))
    
    return f"已完成重命名 {renamed_count}/{len(lines)} 个文件，并更新list文件"


def main():
    with gr.Blocks(title="字幕切分工具") as app:
        gr.Markdown(value="根据srt字幕进行音频切分，过滤指定长度")
        with gr.Row():
            raw_audio = gr.Textbox(label="音频")
            input_srt = gr.Textbox(label="srt文件")

        with gr.Row():
            with gr.Column():
                min_seconds = gr.Slider(
                    minimum=1,
                    maximum=40,
                    step=1,
                    label="最小秒数",
                    value=3,
                    interactive=True,
                )
                max_seconds = gr.Slider(
                    minimum=1,
                    maximum=40,
                    step=1,
                    label="最大秒数",
                    value=10,
                    interactive=True,
                )
            output_dir = gr.Textbox(
                label="*生成音频保存位置",
                value=r"D:\aisound\temp",
                interactive=True,
                scale=4,
            )
            run_button = gr.Button("切分", variant="primary", scale=1)

        run_button.click(
            asr_cut,
            [raw_audio, input_srt, min_seconds, max_seconds, output_dir],
            [run_button],
        )

        gr.Markdown(value=i18n("语音文本校对标注工具"))
        with gr.Row():
            if_label = gr.Checkbox(label=i18n("是否开启打标WebUI"), show_label=True)
            path_list = gr.Textbox(
                label=i18n(".list标注文件的路径"),
                value=r"D:\aisound\sound_data\简一\raw_cut.list",
                interactive=True,
            )
            label_info = gr.Textbox(label=i18n("打标工具进程输出信息"))
        if_label.change(change_label, [if_label, path_list], [label_info])

        gr.Markdown(value=i18n("合并标注文件音频，并转为srt"))
        with gr.Row():
            asr_file = gr.Textbox(
                label="srt文件",
                value=r"D:\aisound\sound_data\简一\raw_cut.list",
                scale=4,
            )
            asr_wavsrt_output_dir = gr.Textbox(
                label="*生成音频和srt保存位置",
                value=r"D:\aisound\temp",
                interactive=True,
                scale=4,
            )
            trans_button = gr.Button("合并", variant="primary", scale=1)

        trans_button.click(
            merge_audio_with_subtitles,
            [asr_file, asr_wavsrt_output_dir],
            [trans_button],
        )

        gr.Markdown(value="从wav目录生成list文件")
        with gr.Row():
            wav_dir_input = gr.Textbox(
                label="wav文件目录",
                value=r"D:\aisound\sound_data\已鸟\已鸟cut",
                interactive=True
            )
            gen_list_button = gr.Button("生成list", variant="primary")
            list_output = gr.Textbox(label="输出信息")
        
        gen_list_button.click(
            generate_list_from_wav,
            [wav_dir_input],
            [list_output]
        )

        gr.Markdown(value="根据list文件重命名wav文件")
        with gr.Row():
            rename_list_input = gr.Textbox(
                label="list文件路径",
                value=r"D:\aisound\sound_data\简一\raw_cut.list",
                interactive=True
            )
            rename_button = gr.Button("重命名wav", variant="primary")
            rename_output = gr.Textbox(label="输出信息")
        
        rename_button.click(
            rename_wav_from_list,
            [rename_list_input],
            [rename_output]
        )

    app.queue().launch(
        server_name="0.0.0.0",
        inbrowser=True,
        server_port=9889,
        quiet=True,
    )


# 使用示例
if __name__ == "__main__":
    main()
