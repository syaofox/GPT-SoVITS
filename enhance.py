import gradio as gr
import torch
import torchaudio
import gc
from pathlib import Path
from tqdm import tqdm

from tools import my_utils
from scipy.io.wavfile import write

from resemble_enhance.enhancer.inference import denoise, enhance

import os
now_dir = os.getcwd()
# sys.path.append(now_dir)
os.environ['GRADIO_TEMP_DIR'] = f'{now_dir}/TEMP'

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def clear_gpu_cash():
    # del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _fn(path, solver, nfe, tau,chunk_seconds,chunks_overlap, denoising):
    if path is None:
        return None, None

    print(path)

    solver = solver.lower()
    nfe = int(nfe)
    lambd = 0.9 if denoising else 0.1

    dwav, sr = torchaudio.load(path)
    dwav = dwav.mean(dim=0)

    wav1, new_sr = denoise(dwav, sr, device)
    wav2, new_sr = enhance(dwav = dwav, sr = sr, device = device, nfe=nfe,chunk_seconds=chunk_seconds,chunks_overlap=chunks_overlap, solver=solver, lambd=lambd, tau=tau)

    wav1 = wav1.cpu().numpy()
    wav2 = wav2.cpu().numpy()

    clear_gpu_cash()
    return (new_sr, wav1), (new_sr, wav2)


def _batch_denoise(in_path,output_path):


    in_path = my_utils.clean_path(in_path)
    output_path = my_utils.clean_path(output_path)

    for wav_file in tqdm(Path(in_path).glob("*.wav")):

        temp_path = Path(output_path) / Path(wav_file).name
        if temp_path.exists():
            tqdm.write(f'跳过:{str(temp_path)}')
            continue

        temp_path = str(temp_path)
        tqdm.write(temp_path)
        

        # solver = solver.lower()
        # nfe = int(nfe)
        # lambd = 0.9 if denoising else 0.1

        dwav, sr = torchaudio.load(wav_file)
        dwav = dwav.mean(dim=0)

        
        wav1, new_sr1 = denoise(dwav, sr, device)
        wav1 = wav1.cpu().numpy()
        write(temp_path, new_sr1, wav1)



    clear_gpu_cash()    
    return '批量降噪'

def _batch_enhance(xpath, solver, nfe, tau,chunk_seconds,chunks_overlap, denoising,in_path,output_path):


    in_path = my_utils.clean_path(in_path)
    output_path = my_utils.clean_path(output_path)

    for wav_file in tqdm(Path(in_path).glob("*.wav")):

        temp_path = Path(output_path) / Path(wav_file).name
        if temp_path.exists():
            tqdm.write(f'跳过:{str(temp_path)}')
            continue

        temp_path = str(temp_path)
        tqdm.write(temp_path)
        

        solver = solver.lower()
        nfe = int(nfe)
        lambd = 0.9 if denoising else 0.1

        dwav, sr = torchaudio.load(wav_file)
        dwav = dwav.mean(dim=0)

        wav2, new_sr2 = enhance(dwav = dwav, sr = sr, device = device, nfe=nfe,chunk_seconds=chunk_seconds,chunks_overlap=chunks_overlap, solver=solver, lambd=lambd, tau=tau)
        wav2 = wav2.cpu().numpy()
        write(temp_path, new_sr2, wav2)

    clear_gpu_cash()    
    return '批量增强'


def main():
    with gr.Blocks(title="Resemble Enhance") as app:

        with gr.Group():
            gr.Markdown(value="输入")
            with gr.Row():
                with gr.Column():

                    inputs: list = [
                        gr.Audio(type="filepath", label="输入音频"),
                        gr.Dropdown(choices=["Midpoint", "RK4", "Euler"], value="Midpoint", label="CFM ODE Solver (推荐 Midpoint)"),
                        gr.Slider(minimum=1, maximum=128, value=64, step=1, label="CFM数(通常值越高，质量越好，但可能会较慢"),
                        gr.Slider(minimum=0, maximum=1, value=0.5, step=0.01, label="CFM之前的温度(较高的值可以提高质量，但会降低稳定性)"),
                        gr.Slider(minimum=1, maximum=40, value=10, step=1, label="区块秒数(秒越多，vRAM使用率越高)"),
                        gr.Slider(minimum=0, maximum=5, value=1, step=0.5, label="数据块重叠"),
                        # chunk_seconds, chunks_overlap
                        gr.Checkbox(value=False, label="增强前去噪(如果您的音频包含较强的背景噪声，请勾选)"),
                    ]
            
                with gr.Column():

                    outputs: list = [
                        gr.Audio(label="降噪"),
                        gr.Audio(label="增强"),
                    ]

                    

                    enhance_button = gr.Button("处理", variant="primary")

                    gr.Markdown(value="批量处理")

                    in_wav_dir = gr.Textbox(label="*待处理音频路径", value=r"D:\aisound\GPT-SoVITS\sample\阿醋", interactive=True)
                    out_wav_dir = gr.Textbox(label="*结果输出路径", value=r"D:\aisound\temp\enhance", interactive=True)
                    

                    with gr.Row():
                        batch_denoise_button = gr.Button("批量降噪", variant="primary")
                        batch_enhance_button = gr.Button("批量增强", variant="primary")


            enhance_button.click(_fn, inputs, outputs)

            
            batch_denoise_button.click(_batch_denoise, [in_wav_dir,out_wav_dir], [batch_denoise_button])
            inputs.append(in_wav_dir)
            inputs.append(out_wav_dir)
            batch_enhance_button.click(_batch_enhance, inputs, [batch_enhance_button])


    app.queue(max_size=1022).launch(
        server_name="0.0.0.0",
        inbrowser=True,
        share=False,
        server_port=7860,
        quiet=True,
    )


if __name__ == "__main__":
    main()
