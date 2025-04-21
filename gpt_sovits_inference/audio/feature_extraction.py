"""
音频特征提取功能
"""

import torch
import librosa
import torchaudio
from typing import Tuple, Dict, Any, Optional


def extract_ref_features(
    ref_wav_path: str,
    ssl_model: Any,
    vq_model: Any,
    device: str = "cuda",
    is_half: bool = True,
    zero_wav_torch: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从参考音频提取特征
    
    参数:
        ref_wav_path: 参考音频路径
        ssl_model: 自监督学习模型
        vq_model: VQ-VAE模型
        device: 计算设备
        is_half: 是否使用半精度
        zero_wav_torch: 静音音频，用于填充
        
    返回:
        (prompt, codes): 提取的特征和编码
    """
    with torch.no_grad():
        wav16k, sr = librosa.load(ref_wav_path, sr=16000)
        
        # 检查参考音频长度
        if wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000:
            raise ValueError("参考音频在3~10秒范围外，请更换！")
            
        wav16k = torch.from_numpy(wav16k)
        if is_half:
            wav16k = wav16k.half().to(device)
        else:
            wav16k = wav16k.to(device)
            
        # 如果提供了静音填充，则添加
        if zero_wav_torch is not None:
            wav16k = torch.cat([wav16k, zero_wav_torch])
            
        # 提取特征
        ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)
        codes = vq_model.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]
        prompt = prompt_semantic.unsqueeze(0).to(device)
        
    return prompt, codes


def process_reference_audio(
    ref_wav_path: str,
    model_version: str,
    vq_model: Any,
    phones: Any,
    refer: torch.Tensor,
    device: str = "cuda",
    speed: float = 1.0,
    dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor, Any]:
    """
    处理参考音频用于v3/v4模型
    
    参数:
        ref_wav_path: 参考音频路径
        model_version: 模型版本
        vq_model: VQ-VAE模型
        phones: 音素序列
        refer: 参考音频频谱
        device: 计算设备
        speed: 语速
        dtype: 数据类型
        
    返回:
        (fea_ref, mel2, ge): 特征、mel频谱和解码器状态
    """
    from .audio_processing import resample, mel_spec_v3, mel_spec_v4, norm_spec
    
    # 创建音素ID张量
    phoneme_ids = torch.LongTensor(phones).to(device).unsqueeze(0)
    
    # 解码生成特征
    fea_ref, ge = vq_model.decode_encp(phoneme_ids.unsqueeze(0), phoneme_ids, refer)
    
    # 加载参考音频
    ref_audio, sr = torchaudio.load(ref_wav_path)
    ref_audio = ref_audio.to(device).float()
    
    # 如果是立体声，转换为单声道
    if ref_audio.shape[0] == 2:
        ref_audio = ref_audio.mean(0).unsqueeze(0)
    
    # 重采样到目标采样率
    tgt_sr = 24000 if model_version == "v3" else 32000
    if sr != tgt_sr:
        ref_audio = resample(ref_audio, sr, tgt_sr, device)
    
    # 根据模型版本计算mel频谱
    if model_version == "v3":
        mel2 = mel_spec_v3(ref_audio)
    else:
        mel2 = mel_spec_v4(ref_audio)
    
    # 归一化
    mel2 = norm_spec(mel2)
    
    # 截取最小长度
    T_min = min(mel2.shape[2], fea_ref.shape[2])
    mel2 = mel2[:, :, :T_min]
    fea_ref = fea_ref[:, :, :T_min]
    
    # 根据模型版本设置参数
    Tref = 468 if model_version == "v3" else 500
    
    # 如果超过Tref，截取
    if T_min > Tref:
        mel2 = mel2[:, :, -Tref:]
        fea_ref = fea_ref[:, :, -Tref:]
        T_min = Tref
    
    # 转换为指定数据类型
    mel2 = mel2.to(dtype)
    
    return fea_ref, mel2, ge 