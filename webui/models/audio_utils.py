import soundfile as sf
import numpy as np
import librosa
from scipy import signal
from typing import List, Dict, Any


def normalize_audio(
    audio_data: List[Dict[str, Any]],
    target_db=-18,
    compress=True,
    threshold_db=-20,
    ratio=2.0,
):
    """
    对音频进行归一化处理，避免破音

    参数:
        audio_data: 音频数据列表或包含音频数据的字典列表({'audio': 音频数据, 'sr': 采样率})
        target_db: 目标RMS音量(dB)，一般音频混音使用-18dB左右比较安全
        compress: 是否应用动态范围压缩
        threshold_db: 压缩器阈值(dB)
        ratio: 压缩比例

    返回:
        归一化后的音频数据列表或字典列表(与输入格式一致)
    """

    # 计算每个音频段落的RMS值
    def rms(x):
        return np.sqrt(np.mean(np.square(x)))

    # 将线性值转换为dB
    def to_db(x):
        return 20 * np.log10(np.maximum(1e-8, x))  # 避免log(0)

    # 将dB转换为线性值
    def from_db(x):
        return np.power(10, x / 20)

    # 软限幅函数，避免硬截断造成的破音
    def soft_clip(x, threshold=0.9):
        # tanh软限幅，保持-1到1之间
        return np.tanh(x / threshold) * threshold

    # 动态范围压缩
    def compress_dynamic_range(audio, threshold_db, ratio):
        # 转换为浮点数确保计算精度
        audio = audio.astype(np.float64)

        # 计算信号包络
        abs_audio = np.abs(audio)
        # 使用简单的低通滤波器获取包络
        window_size = min(1024, len(audio) // 4)
        if window_size > 0:
            envelope = signal.convolve(
                abs_audio, np.ones(window_size) / window_size, mode="same"
            )
        else:
            envelope = abs_audio

        # 将包络转换为dB
        envelope_db = to_db(envelope)

        # 应用压缩
        delta_db = envelope_db - threshold_db
        delta_db[delta_db < 0] = delta_db[delta_db < 0] / ratio  # 只压缩超过阈值的部分

        # 计算增益并应用
        gain_db = delta_db - envelope_db
        gain = from_db(gain_db)

        # 应用增益
        return audio * gain

    # 字典列表格式输入
    normalized_segments = []

    for segment in audio_data:
        audio = segment["audio"].astype(np.float64)  # 确保使用浮点数进行计算

        # 应用动态范围压缩
        if compress and len(audio) > 0:
            audio = compress_dynamic_range(audio, threshold_db, ratio)

        # 计算当前RMS
        current_rms = rms(audio)

        if current_rms > 0:
            # 计算需要的增益
            current_db = to_db(current_rms)
            gain = from_db(target_db - current_db)

            # 限制增益，避免过度放大噪音
            gain = min(gain, 3.0)  # 最大放大3倍

            # 应用增益
            audio = audio * gain

            # 软限幅，避免破音
            audio = soft_clip(audio)

        # 创建新的字典，保留原始字典中的所有属性
        normalized_segment = segment.copy()
        normalized_segment["audio"] = audio
        normalized_segments.append(normalized_segment)

    return normalized_segments


def merge_audio(audio_data: List[Dict[str, Any]], target_sr=None):
    """
    合并音频数据，支持不同采样率的音频合并

    参数:
        audio_data: 音频数据列表或包含音频数据的字典列表({'audio': 音频数据, 'sr': 采样率})
        target_sr: 目标采样率，如果不指定，则使用第一个音频的采样率

    返回:
        合并后的音频数据和采样率
    """

    # 字典列表格式处理
    if target_sr is None:
        # 如果未指定目标采样率，使用第一个片段的采样率
        target_sr = audio_data[0]["sr"]

    # 重采样所有音频到目标采样率
    processed_audios = []
    for segment in audio_data:
        audio = segment["audio"]
        sr = segment["sr"]

        if sr != target_sr:
            # 使用librosa重采样
            resampled_audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            processed_audios.append(resampled_audio)
        else:
            processed_audios.append(audio)

    # 合并处理后的音频
    return np.concatenate(processed_audios), target_sr


def save_audio(sr, audio_data, filename):
    """
    保存音频数据
    """
    sf.write(filename, audio_data, sr)
