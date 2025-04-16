"""
GPT-SoVITS 配置类
"""

import os
import sys

class GPTSoVITSConfig:
    """GPT-SoVITS 配置类"""
    
    def __init__(self):
        # 当前工作目录
        self.now_dir = os.getcwd()
        
        # 模型路径
        self.sovits_path = ""
        self.gpt_path = ""
        self.cnhubert_path = os.path.join(self.now_dir, "GPT_SoVITS", "pretrained_models", "chinese-hubert-base")
        self.bert_path = os.path.join(self.now_dir, "GPT_SoVITS", "pretrained_models", "chinese-roberta-wwm-ext-large")
        
        # 预训练模型路径
        self.pretrained_sovits_path = os.path.join(self.now_dir, "GPT_SoVITS", "pretrained_models", "s2G488k.pth")
        self.pretrained_gpt_path = os.path.join(self.now_dir, "GPT_SoVITS", "pretrained_models", "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt")
        
        # 推理设备
        self.infer_device = "cuda" if self.cuda_is_available() else "cpu"
        
        # 半精度推理
        self.is_half = True if self.infer_device == "cuda" else False
        
        # API端口
        self.api_port = 9880
        
        # Python 可执行文件路径
        self.python_exec = sys.executable
        
    def cuda_is_available(self):
        """检查CUDA是否可用"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False 