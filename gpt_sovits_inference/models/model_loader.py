"""
模型加载函数
"""

import os
import json
import torch
from typing import Tuple, Dict, Any, Optional
from gpt_sovits_inference.utils import DictToAttrRecursive


def load_sovits_model(
    sovits_path: str,
    device: str = "cuda",
    is_half: bool = True
) -> Tuple[Any, Dict[str, Any], str, str, bool]:
    """
    加载SoVITS模型
    
    参数:
        sovits_path: 模型权重路径
        device: 计算设备
        is_half: 是否使用半精度
        
    返回:
        (vq_model, hps, version, model_version, if_lora_v3): 
        VQ-VAE模型、超参数、版本信息、模型版本、是否是LoRA
    """
    from process_ckpt import get_sovits_version_from_path_fast, load_sovits_new
    from GPT_SoVITS.module.models import SynthesizerTrn, SynthesizerTrnV3
    from peft import LoraConfig, get_peft_model
    
    # 检查模型路径是否存在
    path_sovits_v3 = "GPT_SoVITS/pretrained_models/s2Gv3.pth"
    path_sovits_v4 = "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth"
    is_exist_s2gv3 = os.path.exists(path_sovits_v3)
    is_exist_s2gv4 = os.path.exists(path_sovits_v4)
    
    # 获取模型版本信息
    version, model_version, if_lora_v3 = get_sovits_version_from_path_fast(sovits_path)
    v3v4set = {"v3", "v4"}
    
    # 检查是否为LoRA模型但缺少底模
    if if_lora_v3 and not (is_exist_s2gv3 if model_version == "v3" else is_exist_s2gv4):
        error_msg = f"SoVITS {model_version} 底模缺失，无法加载相应 LoRA 权重"
        raise FileExistsError(error_msg)
    
    # 加载模型配置和权重
    dict_s2 = load_sovits_new(sovits_path)
    hps = DictToAttrRecursive(dict_s2["config"])
    hps.model.semantic_frame_rate = "25hz"
    
    # 确定模型版本
    if "enc_p.text_embedding.weight" not in dict_s2["weight"]:
        hps.model.version = "v2"  # v3model,v2sybomls
    elif dict_s2["weight"]["enc_p.text_embedding.weight"].shape[0] == 322:
        hps.model.version = "v1"
    else:
        hps.model.version = "v2"
    version = hps.model.version
    
    # 创建模型实例
    if model_version not in v3v4set:
        vq_model = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        )
        model_version = version
    else:
        vq_model = SynthesizerTrnV3(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        )
    
    # 移除不需要的组件
    if "pretrained" not in sovits_path:
        try:
            del vq_model.enc_q
        except:
            pass
    
    # 转为half精度并移到设备上
    if is_half:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    
    # 加载权重
    if not if_lora_v3:
        print(f"loading sovits_{model_version}", 
              vq_model.load_state_dict(dict_s2["weight"], strict=False))
    else:
        path_sovits = path_sovits_v3 if model_version == "v3" else path_sovits_v4
        print(
            f"loading sovits_{model_version}pretrained_G",
            vq_model.load_state_dict(load_sovits_new(path_sovits)["weight"], strict=False),
        )
        lora_rank = dict_s2["lora_rank"]
        lora_config = LoraConfig(
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights=True,
        )
        vq_model.cfm = get_peft_model(vq_model.cfm, lora_config)
        print(f"loading sovits_{model_version}_lora{lora_rank}")
        vq_model.load_state_dict(dict_s2["weight"], strict=False)
        vq_model.cfm = vq_model.cfm.merge_and_unload()
        vq_model.eval()
    
    # 更新权重配置
    with open("./weight.json") as f:
        data = json.loads(f.read())
        data["SoVITS"][version] = sovits_path
    with open("./weight.json", "w") as f:
        f.write(json.dumps(data))
        
    return vq_model, hps, version, model_version, if_lora_v3


def load_gpt_model(
    gpt_path: str,
    device: str = "cuda",
    is_half: bool = True,
    version: str = "v2"
) -> Tuple[Any, Dict[str, Any], int, int]:
    """
    加载GPT模型
    
    参数:
        gpt_path: 模型权重路径
        device: 计算设备
        is_half: 是否使用半精度
        version: 模型版本
        
    返回:
        (t2s_model, config, hz, max_sec): GPT模型、配置、帧率、最大秒数
    """
    from AR.models.t2s_lightning_module import Text2SemanticLightningModule
    
    # 加载模型
    dict_s1 = torch.load(gpt_path, map_location="cpu")
    config = dict_s1["config"]
    hz = 50
    max_sec = config["data"]["max_sec"]
    
    # 创建模型实例
    t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    
    # 转为half精度并移到设备上
    if is_half:
        t2s_model = t2s_model.half()
    t2s_model = t2s_model.to(device)
    t2s_model.eval()
    
    # 更新权重配置
    with open("./weight.json") as f:
        data = json.loads(f.read())
        data["GPT"][version] = gpt_path
    with open("./weight.json", "w") as f:
        f.write(json.dumps(data))
        
    return t2s_model, config, hz, max_sec 