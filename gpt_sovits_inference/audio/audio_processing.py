"""
音频处理和变换功能
"""

import torch
import torchaudio
import librosa
import numpy as np
from typing import Tuple, Dict, Any, Optional


def resample(audio_tensor: torch.Tensor, sr0: int, sr1: int, device: str = "cuda") -> torch.Tensor:
    """
    重采样音频张量
    
    参数:
        audio_tensor: 输入音频张量
        sr0: 原始采样率
        sr1: 目标采样率
        device: 计算设备
        
    返回:
        重采样后的音频张量
    """
    resample_transform_dict = {}
    key = f"{sr0}-{sr1}"
    if key not in resample_transform_dict:
        resample_transform_dict[key] = torchaudio.transforms.Resample(sr0, sr1).to(device)
    return resample_transform_dict[key](audio_tensor)


def get_spepc(filename: str, hps: Any, device: str = "cuda") -> torch.Tensor:
    """
    获取音频频谱
    
    参数:
        filename: 音频文件路径
        hps: 超参数配置
        device: 计算设备
        
    返回:
        频谱张量
    """
    from module.mel_processing import spectrogram_torch
    
    # 读取音频
    audio, sampling_rate = librosa.load(filename, sr=int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    
    # 归一化
    maxx = audio.abs().max()
    if maxx > 1:
        audio /= min(2, maxx)
        
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    
    # 转换为频谱
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    
    return spec.to(device)


def audio_sr(audio: torch.Tensor, sr: int, device: str = "cuda") -> Tuple[np.ndarray, int]:
    """
    音频超分辨率增强
    
    参数:
        audio: 输入音频张量
        sr: 原始采样率
        device: 计算设备
        
    返回:
        (enhanced_audio, new_sr): 增强后的音频和新采样率
    """
    try:
        from tools.audio_sr import AP_BWE
        from gpt_sovits_inference.utils import DictToAttrRecursive
        
        sr_model = AP_BWE(device, DictToAttrRecursive)
        return sr_model(audio, sr)
    except FileNotFoundError:
        print("你没有下载超分模型的参数，因此不进行超分。如想超分请先参照教程把文件下载好")
        return audio.cpu().detach().numpy(), sr


def mel_spec_v3(x: torch.Tensor) -> torch.Tensor:
    """
    V3模型的Mel频谱计算
    
    参数:
        x: 输入音频张量
        
    返回:
        Mel频谱张量
    """
    from module.mel_processing import mel_spectrogram_torch
    
    return mel_spectrogram_torch(
        x,
        **{
            "n_fft": 1024,
            "win_size": 1024,
            "hop_size": 256,
            "num_mels": 100,
            "sampling_rate": 24000,
            "fmin": 0,
            "fmax": None,
            "center": False,
        },
    )


def mel_spec_v4(x: torch.Tensor) -> torch.Tensor:
    """
    V4模型的Mel频谱计算
    
    参数:
        x: 输入音频张量
        
    返回:
        Mel频谱张量
    """
    from module.mel_processing import mel_spectrogram_torch
    
    return mel_spectrogram_torch(
        x,
        **{
            "n_fft": 1280,
            "win_size": 1280,
            "hop_size": 320,
            "num_mels": 100,
            "sampling_rate": 32000,
            "fmin": 0,
            "fmax": None,
            "center": False,
        },
    )


def norm_spec(x: torch.Tensor) -> torch.Tensor:
    """
    频谱归一化
    
    参数:
        x: 输入频谱张量
        
    返回:
        归一化后的频谱张量
    """
    spec_min, spec_max = -12, 2
    return (x - spec_min) / (spec_max - spec_min) * 2 - 1


def denorm_spec(x: torch.Tensor) -> torch.Tensor:
    """
    频谱反归一化
    
    参数:
        x: 输入归一化频谱张量
        
    返回:
        原始尺度的频谱张量
    """
    spec_min, spec_max = -12, 2
    return (x + 1) / 2 * (spec_max - spec_min) + spec_min 