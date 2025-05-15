import os
from typing import Dict, Any, Optional

from models.logger import debug, info


class CacheManager:
    """
    缓存管理器，负责管理模型和音频特征的缓存
    """

    # 类级别缓存
    _model_cache: Dict[str, Any] = {}
    _audio_feature_cache: Dict[str, Any] = {}
    _semantic_cache: Dict[str, Any] = {}  # 新增语义缓存

    def __init__(self):
        pass

    @staticmethod
    def get_model_cache_key(
        model_type: str, model_path: str, device: str, is_half: bool
    ) -> str:
        """
        生成模型缓存的键

        Args:
            model_type: 模型类型
            model_path: 模型路径
            device: 设备
            is_half: 是否使用半精度

        Returns:
            缓存键
        """
        return f"{model_type}:{model_path}:{device}:{is_half}"

    @staticmethod
    def get_audio_cache_key(
        filename: str,
        model_version: Optional[str] = None,
        version: Optional[str] = None,
        sampling_rate: int = 0,
    ) -> Optional[str]:
        """
        生成音频特征缓存的键

        Args:
            filename: 音频文件路径
            model_version: 模型版本(v1-v4)
            version: 模型子版本(v1/v2)
            sampling_rate: 采样率

        Returns:
            缓存键或None(如果文件不存在)
        """
        if not os.path.exists(filename):
            return None

        # 使用文件路径、修改时间、文件大小、采样率和模型版本信息生成唯一键
        file_stat = os.stat(filename)
        model_version_info = (
            f"{version}:{model_version}" if version and model_version else "none:none"
        )
        return f"{filename}:{file_stat.st_mtime}:{file_stat.st_size}:{sampling_rate}:{model_version_info}"

    @classmethod
    def store_model(cls, cache_key: str, model_data: Any) -> None:
        """
        存储模型到缓存

        Args:
            cache_key: 缓存键
            model_data: 模型数据
        """
        cls._model_cache[cache_key] = model_data

    @classmethod
    def get_model(cls, cache_key: str) -> Optional[Any]:
        """
        从缓存获取模型

        Args:
            cache_key: 缓存键

        Returns:
            模型数据或None(如果缓存中不存在)
        """
        return cls._model_cache.get(cache_key)

    @classmethod
    def store_audio_feature(cls, cache_key: str, feature_data: Any) -> None:
        """
        存储音频特征到缓存

        Args:
            cache_key: 缓存键
            feature_data: 特征数据
        """
        cls._audio_feature_cache[cache_key] = feature_data

    @classmethod
    def get_audio_feature(cls, cache_key: str) -> Optional[Any]:
        """
        从缓存获取音频特征

        Args:
            cache_key: 缓存键

        Returns:
            特征数据或None(如果缓存中不存在)
        """
        return cls._audio_feature_cache.get(cache_key)

    @classmethod
    def get(cls, cache_key: str) -> Optional[Any]:
        """
        从语义缓存中获取语义生成结果

        Args:
            cache_key: 缓存键，通常包含所有影响生成结果的参数

        Returns:
            语义数据或None(如果缓存中不存在)
        """
        return cls._semantic_cache.get(cache_key)

    @classmethod
    def set(cls, cache_key: str, semantic_data: Any) -> None:
        """
        存储语义生成结果到缓存

        Args:
            cache_key: 缓存键，通常包含所有影响生成结果的参数
            semantic_data: 语义数据
        """
        cls._semantic_cache[cache_key] = semantic_data
        debug(f"缓存语义生成结果，键: {cache_key[:50]}...")

    @classmethod
    def clear_cache(cls, cache_type: Optional[str] = None) -> None:
        """
        清除缓存

        Args:
            cache_type: 缓存类型，可选值: "model", "audio", "semantic", None(全部)
        """
        if cache_type is None or cache_type == "model":
            cls._model_cache.clear()
            info("已清除模型缓存")

        if cache_type is None or cache_type == "audio":
            cls._audio_feature_cache.clear()
            info("已清除音频特征缓存")

        if cache_type is None or cache_type == "semantic":
            cls._semantic_cache.clear()
            info("已清除语义生成缓存")

    @classmethod
    def clear_incompatible_audio_cache(
        cls, model_version: Optional[str] = None, version: Optional[str] = None
    ) -> None:
        """
        仅清除与指定模型版本不兼容的音频特征缓存

        Args:
            model_version: 模型主版本(v1-v4)
            version: 模型子版本(v1/v2)
        """
        if not model_version or not version:
            # 如果没有有效的版本信息，清空所有缓存
            cls._audio_feature_cache.clear()
            info("已清除所有音频特征缓存（无法确定模型版本）")
            return

        version_tag = f"{version}:{model_version}"
        keys_to_remove = []

        for key in cls._audio_feature_cache.keys():
            # 查找不与当前版本匹配的缓存
            if not key.endswith(version_tag) and "none:none" not in key:
                keys_to_remove.append(key)

        # 删除不兼容的缓存
        for key in keys_to_remove:
            del cls._audio_feature_cache[key]

        if keys_to_remove:
            info(
                f"已清除{len(keys_to_remove)}项与当前模型版本({version_tag})不兼容的音频特征缓存"
            )
        else:
            info(f"未发现与当前模型版本({version_tag})不兼容的音频特征缓存")
