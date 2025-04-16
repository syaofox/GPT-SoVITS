"""
GPT-SoVITS 本地库

此库提供了直接调用 GPT-SoVITS 模型进行语音合成的功能
"""

from .core import GPTSoVITS, Speaker
from .config import GPTSoVITSConfig

__all__ = ['GPTSoVITS', 'Speaker', 'GPTSoVITSConfig']
__version__ = '0.1.0' 