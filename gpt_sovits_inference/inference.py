"""
GPT-SoVITS 推理模块主文件
"""

import os,sys
now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS"%(now_dir))

import json
import torch
import numpy as np
import torchaudio
import librosa
from typing import List, Dict, Tuple, Union, Optional, Any
from transformers import AutoModelForMaskedLM, AutoTokenizer

from .utils import setup_logging, DictToAttrRecursive
from .models import load_sovits_model, load_gpt_model, init_bigvgan, init_hifigan
from .text import TextCutter, cut_text, get_phones_and_bert
from .audio import (
    resample, get_spepc, audio_sr, norm_spec, denorm_spec,
    mel_spec_v3, mel_spec_v4, extract_ref_features
)

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class GPTSoVITSInference:
    """
    GPT-SoVITS语音合成推理模块
    提供与原Web UI相同的语音合成功能，但更容易集成到其他应用中
    
    特性:
    1. 模型缓存功能: 同一路径的模型只会加载一次，多次调用时复用已加载的模型，节省GPU内存
    2. 支持手动释放资源: 可以通过clear_model_cache静态方法释放模型缓存
    3. 实例释放: 可以通过release_models方法释放当前实例的模型资源
    4. 多设备支持: 同一模型可以在不同设备上缓存，比如同时在cuda:0和cuda:1上使用
    
    使用示例:
    ```python
    # 创建第一个实例，将加载并缓存模型
    inference1 = GPTSoVITSInference(gpt_path="path/to/gpt", sovits_path="path/to/sovits")
    # 生成语音
    inference1.generate_speech(...)
    
    # 创建第二个使用相同模型的实例，将复用已加载的模型
    inference2 = GPTSoVITSInference(gpt_path="path/to/gpt", sovits_path="path/to/sovits")
    # 生成语音
    inference2.generate_speech(...)
    
    # 创建使用不同模型的实例，将加载新模型并缓存
    inference3 = GPTSoVITSInference(gpt_path="path/to/another/gpt", sovits_path="path/to/sovits")
    
    # 释放所有模型缓存
    GPTSoVITSInference.clear_model_cache()
    
    # 或仅释放特定类型的模型缓存
    GPTSoVITSInference.clear_model_cache(model_type='gpt')
    
    # 或仅释放特定设备上的模型缓存
    GPTSoVITSInference.clear_model_cache(device='cuda:0')
    
    # 或仅释放特定设备上的特定类型模型缓存
    GPTSoVITSInference.clear_model_cache(model_type='gpt', device='cuda:0')
    
    # 释放当前实例使用的模型资源，但不影响全局缓存
    inference1.release_models()
    ```
    """
    
    # 静态变量存储已加载的模型，避免重复加载
    _loaded_models = {
        "ssl_model": None,  # CNHubert模型
        "bert_models": {},  # BERT模型 {path: (tokenizer, model)}
        "gpt_models": {},   # GPT模型 {path: (model, config, hz, max_sec)}
        "sovits_models": {}, # SoVITS模型 {path: (model, hps, version, model_version, if_lora_v3)}
        "vocoder_models": {}  # 声码器模型 {"bigvgan_模型版本_设备": model, "hifigan_模型版本_设备": model}
    }
    
    def __init__(
        self,
        gpt_path: str = None,
        sovits_path: str = None,
        device: str = None,
        half: bool = True,
        cnhubert_path: str = "GPT_SoVITS/pretrained_models/chinese-hubert-base",
        bert_path: str = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
    ):
        """
        初始化GPT-SoVITS推理模块
        
        参数:
            gpt_path: GPT模型路径，默认使用配置文件中的设置
            sovits_path: SoVITS模型路径，默认使用配置文件中的设置
            device: 计算设备，默认根据可用性选择CUDA或CPU
            half: 是否使用半精度计算(float16)，只在CUDA可用时有效
            cnhubert_path: CNHubert模型路径
            bert_path: BERT模型路径
        """
        # 设置日志级别
        setup_logging()
        
        # 确定设备
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.is_half = half and torch.cuda.is_available()
        self.dtype = torch.float16 if self.is_half else torch.float32
        
        # 初始化模型路径
        self.cnhubert_base_path = cnhubert_path
        self.bert_path = bert_path
        
        # 加载配置和权重信息
        self._load_weight_config()
        
        # 如果提供了路径，使用提供的路径
        self.gpt_path = gpt_path or self.gpt_path
        self.sovits_path = sovits_path or self.sovits_path
        
        # 初始化模型属性
        self.ssl_model = None
        self.vq_model = None
        self.t2s_model = None
        self.bert_model = None
        self.tokenizer = None
        self.bigvgan_model = None
        self.hifigan_model = None
        
        # 加载模型
        self._load_models()
        
        # 声码器变换字典
        self.resample_transform_dict = {}
        
        # 缓存推理结果
        self.cache = {}
        
        # 文本切分器
        self.text_cutter = TextCutter()
        
    def _load_weight_config(self):
        """加载权重配置信息"""
        # 定义预训练模型路径
        self.pretrained_sovits_name = [
            "GPT_SoVITS/pretrained_models/s2G488k.pth",
            "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth", 
            "GPT_SoVITS/pretrained_models/s2Gv3.pth",
            "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth"
        ]
        
        self.pretrained_gpt_name = [
            "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
            "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
            "GPT_SoVITS/pretrained_models/s1v3.ckpt",
            "GPT_SoVITS/pretrained_models/s1v3.ckpt",
        ]
        
        # 检查并创建权重配置文件
        if not os.path.exists("./weight.json"):
            with open("./weight.json", "w", encoding="utf-8") as file:
                json.dump({"GPT": {}, "SoVITS": {}}, file)
        
        # 读取权重配置
        with open("./weight.json", "r", encoding="utf-8") as file:
            weight_data = json.loads(file.read())
            self.version = os.environ.get("version", "v2")
            self.gpt_path = os.environ.get(
                "gpt_path", 
                weight_data.get("GPT", {}).get(self.version, self.pretrained_gpt_name)
            )
            self.sovits_path = os.environ.get(
                "sovits_path", 
                weight_data.get("SoVITS", {}).get(self.version, self.pretrained_sovits_name)
            )
            
            if isinstance(self.gpt_path, list):
                self.gpt_path = self.gpt_path[0]
            if isinstance(self.sovits_path, list):
                self.sovits_path = self.sovits_path[0]
        
        # 语言映射字典
        self._init_language_dict()
    
    def _init_language_dict(self):
        """初始化语言映射字典"""
        # 由于原代码使用i18n函数，我们这里直接使用中文键值
        self.dict_language_v1 = {
            "中文": "all_zh",  # 全部按中文识别
            "英文": "en",  # 全部按英文识别
            "日文": "all_ja",  # 全部按日文识别
            "中英混合": "zh",  # 按中英混合识别
            "日英混合": "ja",  # 按日英混合识别
            "多语种混合": "auto",  # 多语种启动切分识别语种
        }
        
        self.dict_language_v2 = {
            "中文": "all_zh",  # 全部按中文识别
            "英文": "en",  # 全部按英文识别
            "日文": "all_ja",  # 全部按日文识别
            "粤语": "all_yue",  # 全部按粤语识别
            "韩文": "all_ko",  # 全部按韩文识别
            "中英混合": "zh",  # 按中英混合识别
            "日英混合": "ja",  # 按日英混合识别
            "粤英混合": "yue",  # 按粤英混合识别
            "韩英混合": "ko",  # 按韩英混合识别
            "多语种混合": "auto",  # 多语种启动切分识别语种
            "多语种混合(粤语)": "auto_yue",  # 多语种启动切分识别语种
        }
        
        # 根据版本选择语言字典
        self.dict_language = self.dict_language_v1 if self.version == "v1" else self.dict_language_v2
    
    def _load_models(self):
        """加载所有必要的模型，复用已加载的模型以节省显存"""
        # 导入必要的模块
        from feature_extractor import cnhubert
        
        # 设置CNHubert路径
        cnhubert.cnhubert_base_path = self.cnhubert_base_path
        
        # 检查当前设备，用于缓存模型的键名格式化
        device_key = f"{self.device}"
        
        # 加载BERT模型（如果已加载过该路径的模型则直接使用）
        bert_cache_key = f"{self.bert_path}_{device_key}"
        if bert_cache_key in self._loaded_models["bert_models"]:
            logger.info(f"使用已加载的BERT模型: {self.bert_path} (设备: {self.device})")
            self.tokenizer, self.bert_model = self._loaded_models["bert_models"][bert_cache_key]
            # 确保精度一致
            model_param = next(self.bert_model.parameters())
            if self.is_half and model_param.dtype != torch.float16:
                logger.info(f"转换BERT模型为半精度")
                self.bert_model = self.bert_model.half()
            elif not self.is_half and model_param.dtype == torch.float16:
                logger.info(f"转换BERT模型为全精度")
                self.bert_model = self.bert_model.float()
        else:
            logger.info(f"加载新的BERT模型: {self.bert_path} (设备: {self.device})")
            self.tokenizer = AutoTokenizer.from_pretrained(self.bert_path)
            self.bert_model = AutoModelForMaskedLM.from_pretrained(self.bert_path)
            if self.is_half:
                self.bert_model = self.bert_model.half().to(self.device)
            else:
                self.bert_model = self.bert_model.to(self.device)
            # 缓存已加载的模型
            self._loaded_models["bert_models"][bert_cache_key] = (self.tokenizer, self.bert_model)
        
        # 加载CNHubert模型（如果已加载则直接使用）
        ssl_cache_key = f"ssl_model_{device_key}"
        if ssl_cache_key in self._loaded_models and self._loaded_models[ssl_cache_key] is not None:
            logger.info(f"使用已加载的CNHubert模型 (设备: {self.device})")
            self.ssl_model = self._loaded_models[ssl_cache_key]
            # 确保精度一致
            model_param = next(self.ssl_model.parameters())
            if self.is_half and model_param.dtype != torch.float16:
                logger.info(f"转换CNHubert模型为半精度")
                self.ssl_model = self.ssl_model.half()
            elif not self.is_half and model_param.dtype == torch.float16:
                logger.info(f"转换CNHubert模型为全精度")
                self.ssl_model = self.ssl_model.float()
        else:
            logger.info(f"加载新的CNHubert模型 (设备: {self.device})")
            self.ssl_model = cnhubert.get_model()
            if self.is_half:
                self.ssl_model = self.ssl_model.half().to(self.device)
            else:
                self.ssl_model = self.ssl_model.to(self.device)
            # 缓存已加载的模型
            self._loaded_models[ssl_cache_key] = self.ssl_model
            # 兼容旧版缓存结构
            self._loaded_models["ssl_model"] = self.ssl_model
        
        # 加载SoVITS模型（如果已加载则直接使用）
        sovits_cache_key = f"{self.sovits_path}_{device_key}"
        if sovits_cache_key in self._loaded_models["sovits_models"]:
            logger.info(f"使用已加载的SoVITS模型: {self.sovits_path} (设备: {self.device})")
            (
                self.vq_model, 
                self.hps, 
                self.version, 
                self.model_version, 
                self.if_lora_v3
            ) = self._loaded_models["sovits_models"][sovits_cache_key]
            # 确保精度一致
            model_param = next(self.vq_model.parameters())
            if self.is_half and model_param.dtype != torch.float16:
                logger.info(f"转换SoVITS模型为半精度")
                self.vq_model = self.vq_model.half()
            elif not self.is_half and model_param.dtype == torch.float16:
                logger.info(f"转换SoVITS模型为全精度")
                self.vq_model = self.vq_model.float()
        else:
            logger.info(f"加载新的SoVITS模型: {self.sovits_path} (设备: {self.device})")
            (
                self.vq_model, 
                self.hps, 
                self.version, 
                self.model_version, 
                self.if_lora_v3
            ) = load_sovits_model(self.sovits_path, self.device, self.is_half)
            # 缓存已加载的模型
            self._loaded_models["sovits_models"][sovits_cache_key] = (
                self.vq_model, 
                self.hps, 
                self.version, 
                self.model_version, 
                self.if_lora_v3
            )
        
        # 加载GPT模型（如果已加载则直接使用）
        gpt_cache_key = f"{self.gpt_path}_{device_key}"
        if gpt_cache_key in self._loaded_models["gpt_models"]:
            logger.info(f"使用已加载的GPT模型: {self.gpt_path} (设备: {self.device})")
            (
                self.t2s_model, 
                self.config, 
                self.hz, 
                self.max_sec
            ) = self._loaded_models["gpt_models"][gpt_cache_key]
            # 确保精度一致
            model_param = next(self.t2s_model.parameters())
            if self.is_half and model_param.dtype != torch.float16:
                logger.info(f"转换GPT模型为半精度")
                self.t2s_model = self.t2s_model.half()
            elif not self.is_half and model_param.dtype == torch.float16:
                logger.info(f"转换GPT模型为全精度")
                self.t2s_model = self.t2s_model.float()
        else:
            logger.info(f"加载新的GPT模型: {self.gpt_path} (设备: {self.device})")
            (
                self.t2s_model, 
                self.config, 
                self.hz, 
                self.max_sec
            ) = load_gpt_model(self.gpt_path, self.device, self.is_half, self.version)
            # 缓存已加载的模型
            self._loaded_models["gpt_models"][gpt_cache_key] = (
                self.t2s_model, 
                self.config, 
                self.hz, 
                self.max_sec
            )
        
        # 定义模型版本集合
        self.v3v4set = {"v3", "v4"}
        
        # 根据模型版本加载相应的声码器
        if self.model_version == "v3":
            # 检查缓存中是否有对应的BigVGAN声码器
            bigvgan_cache_key = f"bigvgan_{self.model_version}_{self.device}"
            if bigvgan_cache_key in self._loaded_models["vocoder_models"]:
                logger.info(f"使用已加载的BigVGAN声码器 (设备: {self.device})")
                self.bigvgan_model = self._loaded_models["vocoder_models"][bigvgan_cache_key]
                # 确保精度一致
                model_param = next(self.bigvgan_model.parameters())
                if self.is_half and model_param.dtype != torch.float16:
                    logger.info(f"转换BigVGAN声码器为半精度")
                    self.bigvgan_model = self.bigvgan_model.half()
                elif not self.is_half and model_param.dtype == torch.float16:
                    logger.info(f"转换BigVGAN声码器为全精度")
                    self.bigvgan_model = self.bigvgan_model.float()
            else:
                logger.info(f"加载新的BigVGAN声码器 (设备: {self.device})")
                self.bigvgan_model = init_bigvgan(self.device, self.is_half, self.hifigan_model)
                # 缓存已加载的声码器
                self._loaded_models["vocoder_models"][bigvgan_cache_key] = self.bigvgan_model
        elif self.model_version == "v4":
            # 检查缓存中是否有对应的HiFiGAN声码器
            hifigan_cache_key = f"hifigan_{self.model_version}_{self.device}"
            if hifigan_cache_key in self._loaded_models["vocoder_models"]:
                logger.info(f"使用已加载的HiFiGAN声码器 (设备: {self.device})")
                self.hifigan_model = self._loaded_models["vocoder_models"][hifigan_cache_key]
                # 确保精度一致
                model_param = next(self.hifigan_model.parameters())
                if self.is_half and model_param.dtype != torch.float16:
                    logger.info(f"转换HiFiGAN声码器为半精度")
                    self.hifigan_model = self.hifigan_model.half()
                elif not self.is_half and model_param.dtype == torch.float16:
                    logger.info(f"转换HiFiGAN声码器为全精度")
                    self.hifigan_model = self.hifigan_model.float()
            else:
                logger.info(f"加载新的HiFiGAN声码器 (设备: {self.device})")
                self.hifigan_model = init_hifigan(self.device, self.is_half, self.bigvgan_model)
                # 缓存已加载的声码器
                self._loaded_models["vocoder_models"][hifigan_cache_key] = self.hifigan_model
    
    def _get_vocoder(self):
        """获取当前模型版本对应的声码器"""
        if self.model_version == "v3":
            # 检查BigVGAN声码器
            bigvgan_cache_key = f"bigvgan_{self.model_version}_{self.device}"
            
            # 如果当前实例没有bigvgan声码器
            if self.bigvgan_model is None:
                # 尝试从缓存获取
                if bigvgan_cache_key in self._loaded_models["vocoder_models"]:
                    logger.info(f"从缓存获取BigVGAN声码器 (设备: {self.device})")
                    self.bigvgan_model = self._loaded_models["vocoder_models"][bigvgan_cache_key]
                    # 确保精度一致
                    model_param = next(self.bigvgan_model.parameters())
                    if self.is_half and model_param.dtype != torch.float16:
                        logger.info(f"转换BigVGAN声码器为半精度")
                        self.bigvgan_model = self.bigvgan_model.half()
                    elif not self.is_half and model_param.dtype == torch.float16:
                        logger.info(f"转换BigVGAN声码器为全精度")
                        self.bigvgan_model = self.bigvgan_model.float()
                else:
                    # 初始化新声码器
                    logger.info(f"初始化新的BigVGAN声码器 (设备: {self.device})")
                    self.bigvgan_model = init_bigvgan(self.device, self.is_half, None)
                    # 缓存新初始化的声码器
                    self._loaded_models["vocoder_models"][bigvgan_cache_key] = self.bigvgan_model
            
            return self.bigvgan_model
        else:  # v4
            # 检查HiFiGAN声码器
            hifigan_cache_key = f"hifigan_{self.model_version}_{self.device}"
            
            # 如果当前实例没有hifigan声码器
            if self.hifigan_model is None:
                # 尝试从缓存获取
                if hifigan_cache_key in self._loaded_models["vocoder_models"]:
                    logger.info(f"从缓存获取HiFiGAN声码器 (设备: {self.device})")
                    self.hifigan_model = self._loaded_models["vocoder_models"][hifigan_cache_key]
                    # 确保精度一致
                    model_param = next(self.hifigan_model.parameters())
                    if self.is_half and model_param.dtype != torch.float16:
                        logger.info(f"转换HiFiGAN声码器为半精度")
                        self.hifigan_model = self.hifigan_model.half()
                    elif not self.is_half and model_param.dtype == torch.float16:
                        logger.info(f"转换HiFiGAN声码器为全精度")
                        self.hifigan_model = self.hifigan_model.float()
                else:
                    # 初始化新声码器
                    logger.info(f"初始化新的HiFiGAN声码器 (设备: {self.device})")
                    self.hifigan_model = init_hifigan(self.device, self.is_half, None)
                    # 缓存新初始化的声码器
                    self._loaded_models["vocoder_models"][hifigan_cache_key] = self.hifigan_model
            
            return self.hifigan_model
    
    def generate_speech(
        self,
        ref_wav_path: str,
        prompt_text: str = "",
        prompt_language: str = "中文",
        text: str = "",
        text_language: str = "中文",
        how_to_cut: str = "凑四句一切",
        top_k: int = 20,
        top_p: float = 0.6,
        temperature: float = 0.6,
        ref_free: bool = False,
        speed: float = 1.0,
        if_freeze: bool = False,
        inp_refs: List[str] = None,
        sample_steps: int = 8,
        if_sr: bool = False,
        pause_second: float = 0.3,
        progress_callback = None,
    ) -> Tuple[int, np.ndarray]:
        """
        生成语音
        
        参数:
            ref_wav_path: 参考音频路径
            prompt_text: 参考音频文本
            prompt_language: 参考音频语言
            text: 需要合成的文本
            text_language: 需要合成的文本语言
            how_to_cut: 文本切分方式
            top_k: GPT采样top_k参数
            top_p: GPT采样top_p参数
            temperature: GPT采样temperature参数
            ref_free: 是否开启无参考文本模式
            speed: 语速，值越大越快
            if_freeze: 是否使用上次的合成结果
            inp_refs: 融合多个参考音频的路径列表
            sample_steps: 采样步数
            if_sr: 是否使用超分辨率增强音频
            pause_second: 句间停顿秒数
            progress_callback: 进度回调函数，接收当前处理的文本段索引（从1开始）和总段数
            
        返回:
            采样率和合成音频数据
        """
        print("-"*100)
        logger.info(f"开始生成语音,参数如下:")
        logger.info(f"device: {self.device}")
        logger.info(f"is_half: {self.is_half}")
        logger.info(f"version: {self.version}")
        logger.info(f"model_version: {self.model_version}")
        logger.info(f"if_lora_v3: {self.if_lora_v3}")
       
        logger.info(f"top_k: {top_k}")
        logger.info(f"top_p: {top_p}")
        logger.info(f"temperature: {temperature}")
        logger.info(f"ref_free: {ref_free}")
        logger.info(f"speed: {speed}")
        logger.info(f"if_freeze: {if_freeze}")
        logger.info(f"inp_refs: {inp_refs}")
        logger.info(f"sample_steps: {sample_steps}")
        logger.info(f"if_sr: {if_sr}")
        logger.info(f"pause_second: {pause_second}")

        logger.info(f"gpt_path: {self.gpt_path}")
        logger.info(f"sovits_path: {self.sovits_path}")
        logger.info(f"ref_wav_path: {ref_wav_path}")
        logger.info(f"prompt_text: {prompt_text}")
        logger.info(f"prompt_language: {prompt_language}")
        
        logger.info(f"how_to_cut: {how_to_cut}")

        logger.info(f"text: {text}")
        logger.info(f"text_language: {text_language}")
        print("-"*100)

        
        # 验证输入
        if not ref_wav_path:
            raise ValueError("请提供参考音频路径")
        if not text:
            raise ValueError("请提供需要合成的文本")
        
        # 如果prompt_text为空，自动启用无参考文本模式
        if not prompt_text:
            ref_free = True
            
        # v3和v4模型不支持无参考文本模式
        if self.model_version in self.v3v4set:
            ref_free = False
        # 非v3模型不支持超分
        else:
            if_sr = False
            
        # 处理语言
        prompt_language = self.dict_language[prompt_language]
        text_language = self.dict_language[text_language]
        
        # 处理参考文本
        if not ref_free:
            prompt_text = prompt_text.strip("\n")
            # 确保文本以标点结尾
            splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…"}
            if prompt_text[-1] not in splits:
                prompt_text += "。" if prompt_language != "en" else "."
            print(f"实际输入的参考文本: {prompt_text}")
            
        # 处理目标文本  
        text = text.strip("\n")
        print(f"实际输入的目标文本: {text}")

        # 替换空行为<brbrbrbrbr>段落停顿标记
        sub_texts = []
        for sub_text in text.split("\n"):
            if sub_text == "":
                sub_text = "<brbrbrbrbr>"
            sub_texts.append(sub_text)

        text = "\n".join(sub_texts)
        
        # 创建停顿音频
        zero_wav = np.zeros(
            int(self.hps.data.sampling_rate * pause_second),
            dtype=np.float16 if self.is_half else np.float32,
        )
        zero_wav_torch = torch.from_numpy(zero_wav)
        if self.is_half:
            zero_wav_torch = zero_wav_torch.half().to(self.device)
        else:
            zero_wav_torch = zero_wav_torch.to(self.device)
            
        # 处理参考音频
        if not ref_free:
            try:
                prompt, _ = extract_ref_features(
                    ref_wav_path, 
                    self.ssl_model, 
                    self.vq_model, 
                    self.device, 
                    self.is_half, 
                    zero_wav_torch
                )
            except ValueError as e:
                raise ValueError(f"处理参考音频失败: {str(e)}")
        
        # 文本切分
        text = cut_text(text, how_to_cut)
        while "\n\n" in text:
            text = text.replace("\n\n", "\n")
        print(f"实际输入的目标文本(切句后): {text}")
        
        # 分割文本行
        texts = text.split("\n")
        texts = self.text_cutter.process_text(texts)
        # texts = self.text_cutter.merge_short_text(texts, 5)
        
        # 获取总文本段数
        total_segments = len(texts)
        logger.info(f"文本总段数: {total_segments}")
        
        # 如果有回调函数，通知总段数
        if progress_callback and callable(progress_callback):
            # 发送段数信息（使用0作为当前段表示准备开始）
            progress_callback(0, total_segments)
        
        # 音频输出列表
        audio_opt = []
        
        # 如果不是无参考文本模式，处理参考文本
        if not ref_free:
            phones1, bert1, norm_text1 = get_phones_and_bert(
                self.tokenizer, 
                self.bert_model, 
                self.device, 
                prompt_text, 
                prompt_language, 
                self.version, 
                self.dtype
            )
        
        # 处理每句文本
        for i_text, text in enumerate(texts):
            # 跳过空行
            if len(text.strip()) == 0:
                continue

            if text == "<brbrbrbrbr>":
                audio_opt.append(zero_wav_torch)
                logger.info(f"插入空白音频")
                
                # 在空白音频处理完成后调用进度回调
                if progress_callback and callable(progress_callback):
                    # 检查是否请求停止处理
                    if progress_callback(i_text + 1, total_segments):
                        logger.info("收到停止请求，中止合成")
                        break
                continue
                
            # 确保文本以标点结尾
            splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…"}
            if text[-1] not in splits:
                text += "。" if text_language != "en" else "."
            print(f"实际输入的目标文本(每句): {text}")
            
            try:
                # 处理当前文本
                phones2, bert2, norm_text2 = get_phones_and_bert(
                    self.tokenizer, 
                    self.bert_model, 
                    self.device, 
                    text, 
                    text_language, 
                    self.version, 
                    self.dtype
                )
                print(f"前端处理后的文本(每句): {norm_text2}")
                
                # 根据是否为无参考文本模式组装输入
                if not ref_free:
                    bert = torch.cat([bert1, bert2], 1)
                    all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(self.device).unsqueeze(0)
                else:
                    bert = bert2
                    all_phoneme_ids = torch.LongTensor(phones2).to(self.device).unsqueeze(0)
                    
                bert = bert.to(self.device).unsqueeze(0)
                all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(self.device)
                
                # 使用缓存或重新生成
                if i_text in self.cache and if_freeze:
                    pred_semantic = self.cache[i_text]
                else:
                    with torch.no_grad():
                        pred_semantic, idx = self.t2s_model.model.infer_panel(
                            all_phoneme_ids,
                            all_phoneme_len,
                            None if ref_free else prompt,
                            bert,
                            top_k=top_k,
                            top_p=top_p,
                            temperature=temperature,
                            early_stop_num=self.hz * self.max_sec,
                        )
                        pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
                        self.cache[i_text] = pred_semantic
                
                
                # 根据模型版本选择不同的解码方式
                logger.info(f"模型版本: {self.model_version},开始解码")
                if self.model_version not in self.v3v4set:
                    # v1和v2模型解码逻辑
                    refers = []
                    if inp_refs:
                        for path in inp_refs:
                            try:
                                refer = get_spepc(path, self.hps, self.device).to(self.dtype)
                                refers.append(refer)
                            except Exception as e:
                                print(f"处理参考音频失败: {e}")
                    
                    if len(refers) == 0:
                        refers = [get_spepc(ref_wav_path, self.hps, self.device).to(self.dtype)]
                    
                    audio = self.vq_model.decode(
                        pred_semantic, 
                        torch.LongTensor(phones2).to(self.device).unsqueeze(0), 
                        refers, 
                        speed=speed
                    )[0][0]
                else:
                    # v3和v4模型解码逻辑
                    refer = get_spepc(ref_wav_path, self.hps, self.device).to(self.dtype)
                    phoneme_ids0 = torch.LongTensor(phones1).to(self.device).unsqueeze(0)
                    phoneme_ids1 = torch.LongTensor(phones2).to(self.device).unsqueeze(0)
                    
                    fea_ref, ge = self.vq_model.decode_encp(prompt.unsqueeze(0), phoneme_ids0, refer)
                    ref_audio, sr = torchaudio.load(ref_wav_path)
                    ref_audio = ref_audio.to(self.device).float()
                    if ref_audio.shape[0] == 2:
                        ref_audio = ref_audio.mean(0).unsqueeze(0)
                    
                    tgt_sr = 24000 if self.model_version == "v3" else 32000
                    if sr != tgt_sr:
                        ref_audio = resample(ref_audio, sr, tgt_sr, self.device)
                    
                    # 计算mel频谱
                    if self.model_version == "v3":
                        mel2 = mel_spec_v3(ref_audio)
                    else:
                        mel2 = mel_spec_v4(ref_audio)
                    
                    mel2 = norm_spec(mel2)
                    T_min = min(mel2.shape[2], fea_ref.shape[2])
                    mel2 = mel2[:, :, :T_min]
                    fea_ref = fea_ref[:, :, :T_min]
                    
                    # 根据模型版本设置参数
                    Tref = 468 if self.model_version == "v3" else 500
                    Tchunk = 934 if self.model_version == "v3" else 1000
                    
                    if T_min > Tref:
                        mel2 = mel2[:, :, -Tref:]
                        fea_ref = fea_ref[:, :, -Tref:]
                        T_min = Tref
                    
                    chunk_len = Tchunk - T_min
                    mel2 = mel2.to(self.dtype)
                    fea_todo, ge = self.vq_model.decode_encp(pred_semantic, phoneme_ids1, refer, ge, speed)
                    cfm_resss = []
                    idx = 0
                    
                    # 恢复为无限循环
                    while True:
                        fea_todo_chunk = fea_todo[:, :, idx:idx + chunk_len]
                        if fea_todo_chunk.shape[-1] == 0:
                            break
                        idx += chunk_len
                        fea = torch.cat([fea_ref, fea_todo_chunk], 2).transpose(2, 1)
                        cfm_res = self.vq_model.cfm.inference(
                            fea, 
                            torch.LongTensor([fea.size(1)]).to(fea.device), 
                            mel2,
                            sample_steps, 
                            inference_cfg_rate=0
                        )
                        cfm_res = cfm_res[:, :, mel2.shape[2]:]
                        mel2 = cfm_res[:, :, -T_min:]
                        fea_ref = fea_todo_chunk[:, :, -T_min:]
                        cfm_resss.append(cfm_res)
                    
                    # 合并结果
                    if cfm_resss:
                        cfm_res = torch.cat(cfm_resss, 2)
                        cfm_res = denorm_spec(cfm_res)
                        
                        # 获取对应的声码器并生成波形
                        try:
                            # 记录当前模型版本
                            current_model = self.model_version
                            # 获取声码器前确保self.model_version是正确的
                            logger.info(f"开始获取声码器，模型版本: {self.model_version}")
                            vocoder_model = self._get_vocoder()
                            
                            # 如果模型版本发生了变化（可能是在多线程环境下被修改），还原回来
                            if self.model_version != current_model:
                                logger.warning(f"模型版本在获取声码器过程中被修改，从 {current_model} 变为 {self.model_version}，正在恢复")
                                self.model_version = current_model
                                # 重新获取声码器
                                vocoder_model = self._get_vocoder()
                            
                            # 检查声码器是否正确初始化
                            if vocoder_model is None:
                                raise ValueError(f"声码器初始化失败，当前模型版本: {self.model_version}")
                                
                            logger.info(f"声码器获取成功，开始生成波形")
                            with torch.inference_mode():
                                wav_gen = vocoder_model(cfm_res)
                                audio = wav_gen[0][0]
                        except Exception as e:
                            logger.error(f"声码器处理失败: {str(e)}")
                            # 创建一个短暂的静音音频作为替代
                            audio = torch.zeros(8000, device=self.device)
                    else:
                        # 如果没有成功生成任何块，创建一个空音频
                        logger.warning(f"段落 {i_text+1} 处理失败，未生成音频")
                        audio = torch.zeros(1000, device=self.device)  # 创建短暂的空白音频
                
                logger.info(f"模型版本: {self.model_version},解码完成")
                # 防止爆音
                max_audio = torch.abs(audio).max()
                if max_audio > 1:
                    audio = audio / max_audio
                    
                # 添加到输出列表
                audio_opt.append(audio)
                audio_opt.append(zero_wav_torch)  # 句间停顿
                
                # 在段落处理完成后调用进度回调
                if progress_callback and callable(progress_callback):
                    # 检查是否请求停止处理
                    if progress_callback(i_text + 1, total_segments):
                        logger.info("收到停止请求，中止合成")
                        break
                
            except Exception as e:
                # 捕获处理单个文本段落时可能发生的异常
                logger.error(f"处理段落 {i_text+1}/{total_segments} 时发生错误: {str(e)}")
                # 添加一个空音频段，以便继续处理后续文本
                audio_opt.append(torch.zeros(8000, device=self.device))  # 添加一个空白音频
                audio_opt.append(zero_wav_torch)  # 句间停顿
                
                # 处理异常后也要调用进度回调
                if progress_callback and callable(progress_callback):
                    # 检查是否请求停止处理
                    if progress_callback(i_text + 1, total_segments):
                        logger.info("收到停止请求，中止合成")
                        break
                
                continue  # 继续处理下一个文本段
        
        # 如果没有生成任何音频（可能是所有段落都处理失败），返回一个空音频
        if len(audio_opt) == 0:
            logger.error("所有文本段落处理失败，返回空音频")
            audio_opt = [torch.zeros(8000, device=self.device)]
        
        # 合并所有音频片段
        audio_opt = torch.cat(audio_opt, 0)
        
        # 确定输出采样率
        if self.model_version in {"v1", "v2"}:
            opt_sr = 32000
        elif self.model_version == "v3":
            opt_sr = 24000
        else:
            opt_sr = 48000  # v4
            
        # 音频超分（仅v3模型支持）
        if if_sr and opt_sr == 24000:
            print("音频超分中")
            audio_opt, opt_sr = audio_sr(audio_opt.unsqueeze(0), opt_sr, self.device)
            max_audio = np.abs(audio_opt).max()
            if max_audio > 1:
                audio_opt /= max_audio
        else:
            audio_opt = audio_opt.cpu().detach().numpy()
            
        # 返回采样率和音频数据（转换为16位整数）
        return opt_sr, (audio_opt * 32767).astype(np.int16) 

    @staticmethod
    def clear_model_cache(model_type=None, device=None):
        """
        清理模型缓存，释放GPU内存
        
        参数:
            model_type: 要清理的模型类型，可以是'bert'、'gpt'、'sovits'、'ssl'、'vocoder'或None
                        如果为None则清理所有模型缓存
            device: 要清理的设备上的模型，如果为None则清理所有设备上的模型
        """
        # 导入必要的垃圾回收模块
        import gc
        
        # 根据设备参数准备过滤条件
        device_filter = f"_{device}" if device else ""
        
        if model_type is None or model_type == 'ssl':
            # 清理所有设备上的ssl模型或者指定设备上的ssl模型
            if device:
                ssl_key = f"ssl_model_{device}"
                if ssl_key in GPTSoVITSInference._loaded_models:
                    GPTSoVITSInference._loaded_models[ssl_key] = None
                    logger.info(f"已清理 {device} 设备上的CNHubert模型缓存")
            else:
                # 清理所有和ssl相关的键
                keys_to_clear = [k for k in GPTSoVITSInference._loaded_models.keys() if k.startswith("ssl_model_")]
                for key in keys_to_clear:
                    GPTSoVITSInference._loaded_models[key] = None
                # 兼容旧版结构
                GPTSoVITSInference._loaded_models["ssl_model"] = None
                logger.info("已清理所有设备上的CNHubert模型缓存")
            
        if model_type is None or model_type == 'bert':
            # 清理指定设备或所有设备上的bert模型
            keys_to_clear = [k for k in GPTSoVITSInference._loaded_models["bert_models"].keys() 
                            if not device or k.endswith(device_filter)]
            for key in keys_to_clear:
                del GPTSoVITSInference._loaded_models["bert_models"][key]
            logger.info(f"已清理{'所有' if not device else device}设备上的BERT模型缓存")
            
        if model_type is None or model_type == 'gpt':
            # 清理指定设备或所有设备上的gpt模型
            keys_to_clear = [k for k in GPTSoVITSInference._loaded_models["gpt_models"].keys() 
                            if not device or k.endswith(device_filter)]
            for key in keys_to_clear:
                del GPTSoVITSInference._loaded_models["gpt_models"][key]
            logger.info(f"已清理{'所有' if not device else device}设备上的GPT模型缓存")
            
        if model_type is None or model_type == 'sovits':
            # 清理指定设备或所有设备上的sovits模型
            keys_to_clear = [k for k in GPTSoVITSInference._loaded_models["sovits_models"].keys() 
                            if not device or k.endswith(device_filter)]
            for key in keys_to_clear:
                del GPTSoVITSInference._loaded_models["sovits_models"][key]
            logger.info(f"已清理{'所有' if not device else device}设备上的SoVITS模型缓存")
        
        if model_type is None or model_type == 'vocoder':
            # 清理指定设备或所有设备上的声码器模型
            if device:
                # 清理指定设备上的声码器
                keys_to_clear = [k for k in GPTSoVITSInference._loaded_models["vocoder_models"].keys() 
                                if k.endswith(device)]
                for key in keys_to_clear:
                    del GPTSoVITSInference._loaded_models["vocoder_models"][key]
                logger.info(f"已清理 {device} 设备上的声码器模型缓存")
            else:
                # 清理所有声码器
                GPTSoVITSInference._loaded_models["vocoder_models"].clear()
                logger.info("已清理所有设备上的声码器模型缓存")
            
        # 强制进行垃圾回收
        gc.collect()
        if device and device.startswith("cuda"):
            try:
                torch.cuda.empty_cache()
                logger.info(f"已清理 {device} 设备上的CUDA缓存")
            except:
                logger.warning(f"清理 {device} 设备上的CUDA缓存失败")
        elif not device:
            try:
                torch.cuda.empty_cache()
                logger.info("已清理所有CUDA设备缓存")
            except:
                logger.warning("清理CUDA缓存失败")
        
        logger.info("已完成垃圾回收")
    
    def release_models(self):
        """释放当前实例使用的模型资源，但不影响缓存"""
        # 导入必要的垃圾回收模块
        import gc
        
        # 记录原始设备
        device = self.device
        
        # 释放模型引用，但不清除静态缓存
        self.ssl_model = None
        self.vq_model = None
        self.t2s_model = None
        self.bert_model = None
        self.tokenizer = None
        
        # 释放声码器引用，但不清除静态缓存
        if hasattr(self, 'bigvgan_model') and self.bigvgan_model is not None:
            self.bigvgan_model = None
            logger.info(f"已释放bigvgan声码器实例引用")
            
        if hasattr(self, 'hifigan_model') and self.hifigan_model is not None:
            self.hifigan_model = None
            logger.info(f"已释放hifigan声码器实例引用")
        
        # 强制进行垃圾回收
        gc.collect()
        
        # 如果是CUDA设备，清空CUDA缓存
        if device.startswith("cuda"):
            try:
                torch.cuda.empty_cache()
                logger.info(f"已清理 {device} 设备上的CUDA缓存")
            except:
                logger.warning(f"清理 {device} 设备上的CUDA缓存失败")
                
        logger.info("已释放当前实例的模型资源") 