"""
GPT-SoVITS 语音合成推理模块

提供与原Web UI相同的语音合成功能，但更易于集成
"""

from .inference import GPTSoVITSInference

__version__ = "1.0.0"
__all__ = ["GPTSoVITSInference"] 