import gradio as gr
import torch
import torchaudio
import gc

import webbrowser

from resemble_enhance.enhancer.inference import denoise, enhance

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


def main():
    inputs: list = [
        gr.Audio(type="filepath", label="Input Audio"),
        gr.Dropdown(choices=["Midpoint", "RK4", "Euler"], value="Midpoint", label="CFM ODE Solver (Midpoint is recommended)"),
        gr.Slider(minimum=1, maximum=128, value=64, step=1, label="CFM数(通常值越高，质量越好，但可能会较慢"),
        gr.Slider(minimum=0, maximum=1, value=0.5, step=0.01, label="CFM之前的温度(较高的值可以提高质量，但会降低稳定性)"),
        gr.Slider(minimum=1, maximum=40, value=10, step=1, label="区块秒数(秒越多，vRAM使用率越高)"),
        gr.Slider(minimum=0, maximum=5, value=1, step=0.5, label="数据块重叠"),
        # chunk_seconds, chunks_overlap
        gr.Checkbox(value=False, label="增强前去噪(如果您的音频包含较强的背景噪声，请勾选)"),
    ]

    outputs: list = [
        gr.Audio(label="Output Denoised Audio"),
        gr.Audio(label="Output Enhanced Audio"),
    ]


   
    interface = gr.Interface(
        fn=_fn,
        title="Resemble Enhance",
        description="音频提纯增强",
        inputs=inputs,
        outputs=outputs,
    )

    webbrowser.open(f"http://127.0.0.1:7860")
    interface.launch(share=True, server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
