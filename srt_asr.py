
import os
from subprocess import Popen
import tempfile
import gradio as gr
import re  
from pathlib import Path
from pydub import AudioSegment
import pysrt
from tqdm import tqdm
from tools import my_utils
import platform
import psutil
import signal
from tools.i18n.i18n import I18nAuto
from config import python_exec, webui_port_subfix, is_share
i18n = I18nAuto()

p_label=None
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

system=platform.system()
def kill_process(pid):
    if(system=="Windows"):
        cmd = "taskkill /t /f /pid %s" % pid
        os.system(cmd)
    else:
        kill_proc_tree(pid)

    

def change_label(if_label,path_list):
    global p_label
    if(if_label==True and p_label==None):
        path_list=my_utils.clean_path(path_list)
        cmd = '"%s" tools/subfix_webui.py --load_list "%s" --webui_port %s --is_share %s'%(python_exec,path_list,webui_port_subfix,is_share)
        yield i18n("打标工具WebUI已开启")
        print(cmd)
        p_label = Popen(cmd, shell=True)
    elif(if_label==False and p_label!=None):
        kill_process(p_label.pid)
        p_label=None
        yield i18n("打标工具WebUI已关闭")


def asr_cut(raw_audio, input_srt, min_seconds, max_seconds, output_dir,languate='ZH'):  
    # temp_path = Path(tempfile.mkdtemp(dir='TEMP'))
    main_audio = AudioSegment.from_wav(raw_audio)
    subs = pysrt.open(input_srt)
    
    output_path = Path(output_dir)
    asr_list_file = output_path / 'raw_cut.list'
    raw_cut_path = output_path / 'raw_cut'
    raw_cut_path.mkdir(exist_ok=True)

    asr_list = []

    for i,sub in tqdm(enumerate(subs)):
        start = sub.start.ordinal  # 字幕开始时间，单位为毫秒
        end = sub.end.ordinal       # 字幕结束时间，单位为毫秒   
        text = sub.text
        # 从音频中提取对应的片段
        if end - start < min_seconds * 1000 or end - start > max_seconds * 1000:
            print(f'skip:{text}')
            continue


            
        sub_wav = main_audio[start:end]
        sub_wav_fname =raw_cut_path / f"{i:09d}_{text}.wav"  
        print(f'>>>{text}')  

        asr_list.append(f"{sub_wav_fname}|{raw_cut_path.stem}|{languate.upper()}|{text}")

        sub_wav.export(sub_wav_fname, format="wav")
    
    with open(asr_list_file,'w',encoding='utf8') as f:
        for i in asr_list:
            f.write(i+'\n')

    return '切分'

def main():
   

    with gr.Blocks(title="字幕切分工具") as app:
        gr.Markdown(value="根据srt字幕进行音频切分，过滤指定长度")
        with gr.Row():
            raw_audio = gr.Audio(label='音频', type='filepath')
            input_srt = gr.File(label='srt文件')

            
        with gr.Row():
            with gr.Column():
                min_seconds =  gr.Slider(minimum=1,maximum=40,step=1,label="最小秒数",value=3,interactive=True)
                max_seconds =  gr.Slider(minimum=1,maximum=40,step=1,label="最大秒数",value=10,interactive=True)
            output_dir = gr.Textbox(label="*生成音频保存位置", value=r"D:\aisound\temp", interactive=True,scale=4)
            run_button = gr.Button("切分", variant="primary", scale=1)

        run_button.click(asr_cut, [raw_audio, input_srt, min_seconds, max_seconds, output_dir],[run_button])

        gr.Markdown(value=i18n("语音文本校对标注工具"))
        with gr.Row():
            if_label = gr.Checkbox(label=i18n("是否开启打标WebUI"),show_label=True)
            path_list = gr.Textbox(
                label=i18n(".list标注文件的路径"),
                value=r"D:\aisound\sound_data\简一\raw_cut.list",
                interactive=True,
            )
            label_info = gr.Textbox(label=i18n("打标工具进程输出信息"))
        if_label.change(change_label, [if_label,path_list], [label_info])

    app.queue(max_size=1022).launch(
        server_name="0.0.0.0",
        inbrowser=True,
       
        server_port=9889,
        quiet=True,
    )

# 使用示例
if __name__ == "__main__":
    main()