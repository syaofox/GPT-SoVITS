import os

from typing import List, Dict, Optional
from models.logger import debug
import json
from dataclasses import dataclass


@dataclass
class Prompt:
    ref_wav_path: str  # 参考音频路径
    prompt_text: str  # 参考文本
    prompt_language: str = "中文"  # 参考文本语言
    text_language: str = "中文"  # 文本语言
    how_to_cut: str = "按中文句号。切"  # 文本切分方式
    top_k: int = 15  # GPT采样参数top_k
    top_p: float = 1  # GPT采样参数top_p
    temperature: float = 1  # GPT采样参数temperature
    ref_free: bool = False  # 是否无参考文本模式
    speed: float = 1.0  # 语速
    if_sr: bool = False  # 是否使用超分辨率模型 (仅v3支持)
    sample_steps: int = 32  # 采样步数 (仅v3,v4支持)
    pause_second: float = 0.3  # 句间停顿秒数
    inp_refs: Optional[List[str]] = None  # 多个参考音频路径列表 (仅v1,v2支持)
    gpt_path: Optional[str] = None  # GPT模型路径，如果提供会覆盖初始化时设置的路径
    sovits_path: Optional[str] = None  # SoVITS模型路径
    seed: int = 42  # 随机种子


class PromptService:
    PROMPT_DIR = "webui/prompts"

    def __init__(self):
        # 结构: {角色名: {情绪名: Prompt对象}}
        self.last_mtime: float = 0
        self.prompt_datas: Dict[str, Dict[str, Prompt]] = {}
        self.load_prompt_datas()

    def load_prompt_datas(self):
        """
        加载所有角色的配置文件
        目录结构: PROMPT_DIR/角色名/config.json
        """

        # 初始化prompt_datas
        self.prompt_datas = {}

        if not os.path.exists(self.PROMPT_DIR):
            debug(f"提示音目录 {self.PROMPT_DIR} 不存在")
            return

        # 遍历PROMPT_DIR目录,子目录为角色，子目录下的json文件为角色对应的prompt
        for character in os.listdir(self.PROMPT_DIR):
            character_path = os.path.join(self.PROMPT_DIR, character)
            if not os.path.isdir(character_path):
                continue

            debug(f"加载{character}角色")
            config_file = os.path.join(character_path, "config.json")

            if not os.path.exists(config_file):
                debug(f"{character}角色的配置文件不存在")
                continue

            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    config_data = json.load(f)

                # 初始化角色的情绪字典
                self.prompt_datas[character] = {}

                # 处理emotions部分
                if "emotions" in config_data:
                    for emotion_name, emotion_config in config_data["emotions"].items():
                        # 为每个音频文件添加完整路径
                        if "ref_wav_path" in emotion_config and not os.path.isabs(
                            emotion_config["ref_wav_path"]
                        ):
                            emotion_config["ref_wav_path"] = os.path.join(
                                character_path, emotion_config["ref_wav_path"]
                            )

                        # 处理inp_refs列表中的路径
                        if "inp_refs" in emotion_config and emotion_config["inp_refs"]:
                            for i, ref_path in enumerate(emotion_config["inp_refs"]):
                                if not os.path.isabs(ref_path):
                                    emotion_config["inp_refs"][i] = os.path.join(
                                        character_path, ref_path
                                    )

                        self.prompt_datas[character][emotion_name] = Prompt(**emotion_config)
                        debug(f"已加载{character}角色的{emotion_name}情绪配置")
                        
            except Exception as e:
                debug(f"加载{character}角色配置文件失败: {str(e)}")

    def get_prompt(self, character: str, emotion: str) -> Optional[Prompt]:
        """
        获取指定角色和情绪的Prompt对象

        Args:
            character: 角色名
            emotion: 情绪名

        Returns:
            找到的Prompt对象，如果不存在则返回None
        """
        if character in self.prompt_datas and emotion in self.prompt_datas[character]:
            return self.prompt_datas[character][emotion]
        return None

    def get_character_emotions(self, character: str) -> List[str]:
        """
        获取指定角色的所有情绪名列表

        Args:
            character: 角色名

        Returns:
            情绪名列表，如果角色不存在则返回空列表
        """
        if character in self.prompt_datas:
            return list(self.prompt_datas[character].keys())
        return []

    def get_all_characters(self) -> List[str]:
        """
        获取所有角色名列表

        Returns:
            角色名列表
        """
        return list(self.prompt_datas.keys())
