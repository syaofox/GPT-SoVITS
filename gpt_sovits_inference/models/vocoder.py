"""
声码器初始化和管理
"""

import os
import torch
from typing import Any, Optional


def init_bigvgan(
    device: str = "cuda", 
    is_half: bool = True,
    hifigan_model: Optional[Any] = None
) -> Any:
    """
    初始化BigVGAN声码器(v3模型使用)
    
    参数:
        device: 计算设备
        is_half: 是否使用半精度
        hifigan_model: 已加载的HiFiGAN模型(如果有)
        
    返回:
        BigVGAN模型
    """
    from BigVGAN import bigvgan
    
    # 清理现有声码器占用的内存
    if hifigan_model is not None:
        hifigan_model = hifigan_model.cpu()
        hifigan_model = None
        try:
            torch.cuda.empty_cache()
        except:
            pass
    
    # 获取当前目录
    now_dir = os.getcwd()
    
    # 加载BigVGAN模型
    bigvgan_model = bigvgan.BigVGAN.from_pretrained(
        f"{now_dir}/GPT_SoVITS/pretrained_models/models--nvidia--bigvgan_v2_24khz_100band_256x",
        use_cuda_kernel=False,
    )
    
    # 移除权重标准化并设置为评估模式
    bigvgan_model.remove_weight_norm()
    bigvgan_model = bigvgan_model.eval()
    
    # 转为half精度并移到设备上
    if is_half:
        bigvgan_model = bigvgan_model.half().to(device)
    else:
        bigvgan_model = bigvgan_model.to(device)
        
    return bigvgan_model


def init_hifigan(
    device: str = "cuda", 
    is_half: bool = True,
    bigvgan_model: Optional[Any] = None
) -> Any:
    """
    初始化HiFiGAN声码器(v4模型使用)
    
    参数:
        device: 计算设备
        is_half: 是否使用半精度
        bigvgan_model: 已加载的BigVGAN模型(如果有)
        
    返回:
        HiFiGAN模型
    """
    from GPT_SoVITS.module.models import Generator
    
    # 清理现有声码器占用的内存
    if bigvgan_model is not None:
        bigvgan_model = bigvgan_model.cpu()
        bigvgan_model = None
        try:
            torch.cuda.empty_cache()
        except:
            pass
    
    # 创建HiFiGAN模型
    hifigan_model = Generator(
        initial_channel=100,
        resblock="1",
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[10, 6, 2, 2, 2],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[20, 12, 4, 4, 4],
        gin_channels=0, 
        is_bias=True
    )
    
    # 设置为评估模式并移除权重标准化
    hifigan_model.eval()
    hifigan_model.remove_weight_norm()
    
    # 获取当前目录
    now_dir = os.getcwd()
    
    # 加载权重
    state_dict_g = torch.load(
        f"{now_dir}/GPT_SoVITS/pretrained_models/gsv-v4-pretrained/vocoder.pth", 
        map_location="cpu"
    )
    print("loading vocoder", hifigan_model.load_state_dict(state_dict_g))
    
    # 转为half精度并移到设备上
    if is_half:
        hifigan_model = hifigan_model.half().to(device)
    else:
        hifigan_model = hifigan_model.to(device)
        
    return hifigan_model 