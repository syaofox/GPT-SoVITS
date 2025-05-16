import torch
import torchaudio
import librosa
from typing import Any

from models.logger import debug
from models.cache_manager import CacheManager


class AudioProcessor:
    """
    音频处理器，负责音频的加载、转换和频谱提取
    """

    def __init__(self, device: str, is_half: bool):
        """
        初始化音频处理器

        Args:
            device: 运行设备 (cuda或cpu)
            is_half: 是否使用半精度
        """
        self.device = device
        self.is_half = is_half
        self.dtype = torch.float16 if is_half else torch.float32
        self.resample_transform_dict = {}
        self.hps = None

        # 需要外部设置的函数
        self.mel_spectrogram_torch = None
        self.spectrogram_torch = None

    def set_hps(self, hps: Any) -> None:
        """
        设置音频超参数

        Args:
            hps: 超参数配置
        """
        self.hps = hps

    def resample(self, audio_tensor: torch.Tensor, sr0: int, sr1: int) -> torch.Tensor:
        """
        重采样音频

        Args:
            audio_tensor: 音频张量
            sr0: 原采样率
            sr1: 目标采样率

        Returns:
            重采样后的音频张量
        """
        key = f"{sr0}-{sr1}"
        if key not in self.resample_transform_dict:
            self.resample_transform_dict[key] = torchaudio.transforms.Resample(
                sr0, sr1
            ).to(self.device)
        return self.resample_transform_dict[key](audio_tensor)

    def get_spepc(self, filename: str, spectrogram_torch_fn: Any) -> torch.Tensor:
        """
        获取音频的频谱图

        Args:
            filename: 音频文件路径
            spectrogram_torch_fn: 频谱转换函数

        Returns:
            音频的频谱图
        """
        if not self.hps:
            raise ValueError("请先设置hps参数")

        # 检查缓存
        cache_key = CacheManager.get_audio_cache_key(
            filename,
            getattr(self, "model_version", None),
            getattr(self, "version", None),
            self.hps.data.sampling_rate if self.hps else 0,
        )

        if cache_key and CacheManager.get_audio_feature(cache_key) is not None:
            debug(f"从缓存获取音频特征: {filename}")
            return CacheManager.get_audio_feature(cache_key)

        debug(f"提取音频特征: {filename}")
        audio, sampling_rate = librosa.load(
            filename, sr=int(self.hps.data.sampling_rate)
        )
        audio = torch.FloatTensor(audio)
        maxx = audio.abs().max()
        if maxx > 1:
            audio /= min(float(2), float(maxx))
        audio_norm = audio
        audio_norm = audio_norm.unsqueeze(0)
        spec = spectrogram_torch_fn(
            audio_norm,
            self.hps.data.filter_length,
            self.hps.data.sampling_rate,
            self.hps.data.hop_length,
            self.hps.data.win_length,
            center=False,
        )

        # 缓存特征
        if cache_key:
            CacheManager.store_audio_feature(cache_key, spec)

        return spec

    def mel_fn_v3(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算梅尔频谱图 (v3模型)

        Args:
            x: 输入音频

        Returns:
            梅尔频谱图
        """
        if self.mel_spectrogram_torch is None:
            raise ValueError("请先设置mel_spectrogram_torch函数")

        return self.mel_spectrogram_torch(
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

    def mel_fn_v4(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算梅尔频谱图 (v4模型)

        Args:
            x: 输入音频

        Returns:
            梅尔频谱图
        """
        if self.mel_spectrogram_torch is None:
            raise ValueError("请先设置mel_spectrogram_torch函数")

        return self.mel_spectrogram_torch(
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

    def norm_spec(self, x: torch.Tensor) -> torch.Tensor:
        """
        标准化频谱图

        Args:
            x: 频谱图

        Returns:
            标准化后的频谱图
        """
        spec_min, spec_max = -12, 2
        return (x - spec_min) / (spec_max - spec_min) * 2 - 1

    def denorm_spec(self, x: torch.Tensor) -> torch.Tensor:
        """
        反标准化频谱图

        Args:
            x: 标准化后的频谱图

        Returns:
            原始频谱图
        """
        spec_min, spec_max = -12, 2
        return (x + 1) / 2 * (spec_max - spec_min) + spec_min
