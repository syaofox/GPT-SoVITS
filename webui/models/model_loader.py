import os
import sys
import torch
from typing import Dict, Any

from models.logger import error, info, debug
from models.cache_manager import CacheManager

# 默认路径配置
DEFAULT_PATHS = {
    "cnhubert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
    "bert_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
    "sovits_v3_path": "GPT_SoVITS/pretrained_models/s2Gv3.pth",
    "sovits_v4_path": "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth",
    "vocoder_v4_path": "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/vocoder.pth",
}


class ModelLoader:
    """
    模型加载器，负责加载和管理GPT-SoVITS需要的各种模型
    """

    def __init__(self, device: str, is_half: bool):
        """
        初始化模型加载器

        Args:
            device: 运行设备 (cuda或cpu)
            is_half: 是否使用半精度
        """
        self.device = device
        self.is_half = is_half
        self.dtype = torch.float16 if is_half else torch.float32

        # 可用的模块
        self.get_sovits_version_from_path_fast = None
        self.load_sovits_new = None
        self.SynthesizerTrn = None
        self.SynthesizerTrnV3 = None
        self.Generator = None
        self.Text2SemanticLightningModule = None
        self.get_peft_model = None
        self.LoraConfig = None

        # 模型实例
        self.bert_model = None
        self.tokenizer = None
        self.ssl_model = None
        self.vq_model = None
        self.t2s_model = None
        self.bigvgan_model = None
        self.hifigan_model = None

        # 模型配置
        self.config = None
        self.hps = None
        self.max_sec = None
        self.hz = 50

        # 版本信息
        self.version = None  # v1 or v2
        self.model_version = None  # v1, v2, v3 or v4
        self.if_lora_v3 = False
        self.v3v4set = {"v3", "v4"}

    def load_base_modules(self) -> None:
        """
        加载基础模块
        """
        # 检查是否已经加载了BERT和SSL模型
        if self.bert_model is not None and self.ssl_model is not None:
            debug("已加载BERT和SSL模型，跳过模块加载")
            return

        # 导入模块
        info("正在加载GPT-SoVITS所需模块...")

        # 确保GPT_SoVITS模块可以被导入
        if "GPT_SoVITS" not in sys.modules:
            sys.path.append(os.path.abspath("."))

        from feature_extractor import cnhubert
        from peft import get_peft_model, LoraConfig
        from text import cleaned_text_to_sequence
        from text.LangSegmenter import LangSegmenter
        from text.cleaner import clean_text
        from module.mel_processing import mel_spectrogram_torch, spectrogram_torch
        from AR.models.t2s_lightning_module import Text2SemanticLightningModule
        from GPT_SoVITS.module.models import SynthesizerTrn, SynthesizerTrnV3, Generator
        from process_ckpt import get_sovits_version_from_path_fast, load_sovits_new

        # 保存模块引用
        self.cnhubert = cnhubert
        self.mel_spectrogram_torch = mel_spectrogram_torch
        self.spectrogram_torch = spectrogram_torch
        self.cleaned_text_to_sequence = cleaned_text_to_sequence
        self.clean_text = clean_text
        self.LangSegmenter = LangSegmenter
        self.get_sovits_version_from_path_fast = get_sovits_version_from_path_fast
        self.load_sovits_new = load_sovits_new
        self.SynthesizerTrn = SynthesizerTrn
        self.SynthesizerTrnV3 = SynthesizerTrnV3
        self.Generator = Generator
        self.Text2SemanticLightningModule = Text2SemanticLightningModule
        self.get_peft_model = get_peft_model
        self.LoraConfig = LoraConfig

        # 返回加载的模块，以便外部使用
        return {
            "cleaned_text_to_sequence": cleaned_text_to_sequence,
            "clean_text": clean_text,
            "LangSegmenter": LangSegmenter,
            "mel_spectrogram_torch": mel_spectrogram_torch,
            "spectrogram_torch": spectrogram_torch,
        }

    def load_bert(self, bert_path: str = None) -> None:
        """
        加载BERT模型

        Args:
            bert_path: BERT模型路径
        """
        bert_path = bert_path or DEFAULT_PATHS["bert_path"]

        # 检查BERT模型是否已缓存
        bert_cache_key = CacheManager.get_model_cache_key(
            "bert", bert_path, self.device, self.is_half
        )

        if CacheManager.get_model(bert_cache_key):
            debug("从缓存加载BERT模型")
            cache_data = CacheManager.get_model(bert_cache_key)
            self.tokenizer = cache_data["tokenizer"]
            self.bert_model = cache_data["model"]
            return

        # 加载BERT模型
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        debug(f"加载BERT模型: {bert_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
        self.bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)

        if self.is_half:
            self.bert_model = self.bert_model.half().to(self.device)
        else:
            self.bert_model = self.bert_model.to(self.device)

        # 缓存BERT模型
        CacheManager.store_model(
            bert_cache_key, {"tokenizer": self.tokenizer, "model": self.bert_model}
        )

    def load_ssl(self, cnhubert_base_path: str = None) -> None:
        """
        加载SSL模型

        Args:
            cnhubert_base_path: SSL模型路径
        """
        cnhubert_base_path = cnhubert_base_path or DEFAULT_PATHS["cnhubert_base_path"]

        # 检查SSL模型是否已缓存
        ssl_cache_key = CacheManager.get_model_cache_key(
            "ssl", cnhubert_base_path, self.device, self.is_half
        )

        if CacheManager.get_model(ssl_cache_key):
            debug("从缓存加载SSL模型")
            self.ssl_model = CacheManager.get_model(ssl_cache_key)
            return

        # 加载SSL模型
        debug(f"加载SSL模型: {cnhubert_base_path}")
        self.cnhubert.cnhubert_base_path = cnhubert_base_path
        self.ssl_model = self.cnhubert.get_model()

        if self.is_half:
            self.ssl_model = self.ssl_model.half().to(self.device)
        else:
            self.ssl_model = self.ssl_model.to(self.device)

        # 缓存SSL模型
        CacheManager.store_model(ssl_cache_key, self.ssl_model)

    def load_sovits_model(self, sovits_path: str) -> None:
        """
        加载SoVITS模型

        Args:
            sovits_path: SoVITS模型路径
        """
        # 检查是否已缓存
        sovits_cache_key = CacheManager.get_model_cache_key(
            "sovits", sovits_path, self.device, self.is_half
        )

        if CacheManager.get_model(sovits_cache_key):
            info(f"从缓存加载SoVITS模型: {sovits_path}")
            cache_data = CacheManager.get_model(sovits_cache_key)

            # 记录之前的版本信息，用于判断是否需要更新声码器
            old_model_version = self.model_version

            self.vq_model = cache_data["vq_model"]
            self.version = cache_data["version"]
            self.model_version = cache_data["model_version"]
            self.if_lora_v3 = cache_data["if_lora_v3"]
            self.hps = cache_data["hps"]

            # 检查模型版本是否改变，可能需要更新声码器
            if old_model_version != self.model_version:
                info(
                    f"模型版本从{old_model_version}切换到{self.model_version}，可能需要更新声码器"
                )
                # 声码器会在需要时按需加载

            return

        debug(f"正在加载SoVITS模型: {sovits_path}")

        # 获取模型版本信息
        self.version, self.model_version, self.if_lora_v3 = (
            self.get_sovits_version_from_path_fast(sovits_path)
        )
        debug(
            f"模型信息: version={self.version}, model_version={self.model_version}, if_lora_v3={self.if_lora_v3}"
        )

        # 加载模型权重
        dict_s2 = self.load_sovits_new(sovits_path)
        self.hps = self.dict_to_attr_recursive(dict_s2["config"])
        self.hps.model.semantic_frame_rate = "25hz"

        # 确定模型版本
        if "enc_p.text_embedding.weight" not in dict_s2["weight"]:
            self.hps.model.version = "v2"  # v3model,v2sybomls
        elif dict_s2["weight"]["enc_p.text_embedding.weight"].shape[0] == 322:
            self.hps.model.version = "v1"
        else:
            self.hps.model.version = "v2"

        self.version = self.hps.model.version

        # 加载合适的SoVITS模型
        if self.model_version not in self.v3v4set:
            self.vq_model = self.SynthesizerTrn(
                self.hps.data.filter_length // 2 + 1,
                self.hps.train.segment_size // self.hps.data.hop_length,
                n_speakers=self.hps.data.n_speakers,
                **self.hps.model,
            )
            self.model_version = self.version
        else:
            self.hps.model.version = self.model_version
            self.vq_model = self.SynthesizerTrnV3(
                self.hps.data.filter_length // 2 + 1,
                self.hps.train.segment_size // self.hps.data.hop_length,
                n_speakers=self.hps.data.n_speakers,
                **self.hps.model,
            )

        # 预处理模型
        if "pretrained" not in sovits_path:
            try:
                del self.vq_model.enc_q
            except Exception:
                pass

        # 移动模型到设备
        if self.is_half:
            self.vq_model = self.vq_model.half().to(self.device)
        else:
            self.vq_model = self.vq_model.to(self.device)

        self.vq_model.eval()

        # 加载权重
        if not self.if_lora_v3:
            debug(f"读取 sovits_{self.model_version}")
            self.vq_model.load_state_dict(dict_s2["weight"], strict=False)
        else:
            # LoRA模型加载
            path_sovits = (
                DEFAULT_PATHS["sovits_v3_path"]
                if self.model_version == "v3"
                else DEFAULT_PATHS["sovits_v4_path"]
            )
            debug(f"读取 sovits_{self.model_version}pretrained_G")
            self.vq_model.load_state_dict(
                self.load_sovits_new(path_sovits)["weight"], strict=False
            )

            lora_rank = dict_s2["lora_rank"]
            lora_config = self.LoraConfig(
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
                r=lora_rank,
                lora_alpha=lora_rank,
                init_lora_weights=True,
            )

            self.vq_model.cfm = self.get_peft_model(self.vq_model.cfm, lora_config)
            debug(f"读取 sovits_{self.model_version}_lora{lora_rank}")
            self.vq_model.load_state_dict(dict_s2["weight"], strict=False)
            self.vq_model.cfm = self.vq_model.cfm.merge_and_unload()
            self.vq_model.eval()

        # 缓存模型
        CacheManager.store_model(
            sovits_cache_key,
            {
                "vq_model": self.vq_model,
                "version": self.version,
                "model_version": self.model_version,
                "if_lora_v3": self.if_lora_v3,
                "hps": self.hps,
            },
        )

        debug("SoVITS模型加载完成")

    def load_gpt_model(self, gpt_path: str) -> None:
        """
        加载GPT模型

        Args:
            gpt_path: GPT模型路径
        """
        # 检查是否已缓存
        gpt_cache_key = CacheManager.get_model_cache_key(
            "gpt", gpt_path, self.device, self.is_half
        )

        if CacheManager.get_model(gpt_cache_key):
            info(f"从缓存加载GPT模型: {gpt_path}")
            cache_data = CacheManager.get_model(gpt_cache_key)
            self.t2s_model = cache_data["t2s_model"]
            self.config = cache_data["config"]
            self.max_sec = cache_data["max_sec"]
            self.hz = cache_data["hz"]
            return

        debug(f"正在加载GPT模型: {gpt_path}")

        self.hz = 50
        dict_s1 = torch.load(gpt_path, map_location="cpu")
        self.config = dict_s1["config"]
        self.max_sec = self.config["data"]["max_sec"]

        self.t2s_model = self.Text2SemanticLightningModule(
            self.config, "****", is_train=False
        )
        self.t2s_model.load_state_dict(dict_s1["weight"])

        if self.is_half:
            self.t2s_model = self.t2s_model.half()

        self.t2s_model = self.t2s_model.to(self.device)
        self.t2s_model.eval()

        # 缓存模型
        CacheManager.store_model(
            gpt_cache_key,
            {
                "t2s_model": self.t2s_model,
                "config": self.config,
                "max_sec": self.max_sec,
                "hz": self.hz,
            },
        )

        debug("GPT模型加载完成")

    def init_bigvgan(self) -> None:
        """
        初始化BigVGAN声码器 (用于v3模型)
        """
        # 检查是否已缓存
        bigvgan_cache_key = CacheManager.get_model_cache_key(
            "bigvgan", "v3", self.device, self.is_half
        )

        if CacheManager.get_model(bigvgan_cache_key) and self.bigvgan_model is None:
            debug("从缓存加载BigVGAN声码器")
            self.bigvgan_model = CacheManager.get_model(bigvgan_cache_key)
            return

        if self.bigvgan_model is not None:
            debug("BigVGAN声码器已加载，跳过初始化")
            return

        debug("正在加载BigVGAN声码器...")

        from BigVGAN import bigvgan

        self.bigvgan_model = bigvgan.BigVGAN.from_pretrained(
            "GPT_SoVITS/pretrained_models/models--nvidia--bigvgan_v2_24khz_100band_256x",
            use_cuda_kernel=False,
        )

        # 移除权重标准化并设置为评估模式
        self.bigvgan_model.remove_weight_norm()
        self.bigvgan_model = self.bigvgan_model.eval()

        # 移动到设备
        if self.is_half:
            self.bigvgan_model = self.bigvgan_model.half().to(self.device)
        else:
            self.bigvgan_model = self.bigvgan_model.to(self.device)

        # 缓存声码器
        CacheManager.store_model(bigvgan_cache_key, self.bigvgan_model)

        info("BigVGAN声码器加载完成")

    def init_hifigan(self) -> None:
        """
        初始化HiFiGAN声码器 (用于v4模型)
        """
        # 检查是否已缓存
        hifigan_cache_key = CacheManager.get_model_cache_key(
            "hifigan", "v4", self.device, self.is_half
        )

        if CacheManager.get_model(hifigan_cache_key) and self.hifigan_model is None:
            info("从缓存加载HiFiGAN声码器")
            self.hifigan_model = CacheManager.get_model(hifigan_cache_key)
            return

        if self.hifigan_model is not None:
            info("HiFiGAN声码器已加载，跳过初始化")
            return

        info("正在加载HiFiGAN声码器...")

        self.hifigan_model = self.Generator(
            initial_channel=100,
            resblock="1",
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            upsample_rates=[10, 6, 2, 2, 2],
            upsample_initial_channel=512,
            upsample_kernel_sizes=[20, 12, 4, 4, 4],
            gin_channels=0,
            is_bias=True,
        )

        self.hifigan_model.eval()
        self.hifigan_model.remove_weight_norm()

        # 加载权重
        vocoder_path = DEFAULT_PATHS["vocoder_v4_path"]
        state_dict_g = torch.load(vocoder_path, map_location="cpu")
        info("读取 vocoder")
        self.hifigan_model.load_state_dict(state_dict_g)

        # 移动到设备
        if self.is_half:
            self.hifigan_model = self.hifigan_model.half().to(self.device)
        else:
            self.hifigan_model = self.hifigan_model.to(self.device)

        # 缓存声码器
        CacheManager.store_model(hifigan_cache_key, self.hifigan_model)

        info("HiFiGAN声码器加载完成")

    def manage_vocoders(self) -> None:
        """
        管理声码器，根据当前模型版本确保正确的声码器被加载，并清理不需要的声码器
        """
        if self.model_version == "v3":
            # 确保BigVGAN声码器已加载
            if self.bigvgan_model is None:
                self.init_bigvgan()
            # 清理HiFiGAN声码器以节省内存
            if self.hifigan_model is not None:
                info("清理不需要的HiFiGAN声码器")
                self.hifigan_model = self.hifigan_model.cpu()
                self.hifigan_model = None
                try:
                    torch.cuda.empty_cache()
                except Exception as e:
                    error(f"清理HiFiGAN声码器失败: {e}")

        elif self.model_version == "v4":
            # 确保HiFiGAN声码器已加载
            if self.hifigan_model is None:
                self.init_hifigan()
            # 清理BigVGAN声码器以节省内存
            if self.bigvgan_model is not None:
                info("清理不需要的BigVGAN声码器")
                self.bigvgan_model = self.bigvgan_model.cpu()
                self.bigvgan_model = None
                try:
                    torch.cuda.empty_cache()
                except Exception as e:
                    error(f"清理BigVGAN声码器失败: {e}")

    @staticmethod
    def dict_to_attr_recursive(input_dict: Dict) -> Any:
        """
        将字典转换为对象，支持通过属性访问

        Args:
            input_dict: 输入字典

        Returns:
            转换后的对象
        """

        class DictToAttrRecursive(dict):
            def __init__(self, input_dict):
                super().__init__(input_dict)
                for key, value in input_dict.items():
                    if isinstance(value, dict):
                        value = ModelLoader.dict_to_attr_recursive(value)
                    self[key] = value
                    setattr(self, key, value)

            def __getattr__(self, item):
                try:
                    return self[item]
                except KeyError:
                    raise AttributeError(f"Attribute {item} not found")

            def __setattr__(self, key, value):
                if isinstance(value, dict):
                    value = ModelLoader.dict_to_attr_recursive(value)
                super(DictToAttrRecursive, self).__setitem__(key, value)
                super().__setattr__(key, value)

            def __delattr__(self, item):
                try:
                    del self[item]
                except KeyError:
                    raise AttributeError(f"Attribute {item} not found")

        return DictToAttrRecursive(input_dict)
