"""
GUI模型层

处理数据和业务逻辑
"""

import os
import json
import uuid
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime

# 导入推理模块
from gpt_sovits_inference import GPTSoVITSInference


class RoleModel:
    """角色模型类，用于管理角色配置"""
    
    def __init__(self, roles_dir: str = "gui/roles"):
        """
        初始化角色模型
        
        参数:
            roles_dir: 角色配置目录
        """
        self.roles_dir = Path(roles_dir)
        self.roles_dir.mkdir(exist_ok=True)
        self.roles: Dict[str, Dict] = {}
        self.load_roles()
    
    def load_roles(self) -> Dict[str, Dict]:
        """加载所有角色配置"""
        self.roles = {}
        
        for role_dir in self.roles_dir.iterdir():
            if not role_dir.is_dir():
                continue
                
            config_file = role_dir / "config.json"
            if not config_file.exists():
                continue
                
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
                
                # 将相对路径的音频文件转换为绝对路径
                for emotion_name, emotion_config in config.get("emotions", {}).items():
                    if "ref_audio" in emotion_config:
                        ref_audio = emotion_config["ref_audio"]
                        if not os.path.isabs(ref_audio):
                            emotion_config["ref_audio"] = str(role_dir / ref_audio)
                    
                    # 处理辅助参考音频
                    if "aux_refs" in emotion_config:
                        aux_refs = []
                        for aux_ref in emotion_config["aux_refs"]:
                            if not os.path.isabs(aux_ref):
                                aux_refs.append(str(role_dir / aux_ref))
                            else:
                                aux_refs.append(aux_ref)
                        emotion_config["aux_refs"] = aux_refs
                
                self.roles[role_dir.name] = config
            except Exception as e:
                print(f"加载角色配置失败: {role_dir.name}, 错误: {str(e)}")
        
        return self.roles
    
    def get_role_emotions(self, role_name: str) -> Dict[str, Dict]:
        """获取角色的所有情感配置"""
        role_config = self.roles.get(role_name, {})
        return role_config.get("emotions", {})
    
    def get_emotion_config(self, role_name: str, emotion_name: str) -> Dict:
        """获取特定角色的特定情感配置"""
        emotions = self.get_role_emotions(role_name)
        return emotions.get(emotion_name, {})
    
    def save_role_config(self, role_name: str, config: Dict) -> bool:
        """保存角色配置"""
        role_dir = self.roles_dir / role_name
        role_dir.mkdir(exist_ok=True)
        
        config_file = role_dir / "config.json"
        
        try:
            # 深拷贝配置，以防意外修改
            config_copy = json.loads(json.dumps(config))
            
            # 处理音频文件路径，将绝对路径转为相对路径保存
            for emotion_name, emotion_config in config_copy.get("emotions", {}).items():
                if "ref_audio" in emotion_config:
                    ref_audio = emotion_config["ref_audio"]
                    if os.path.isabs(ref_audio):
                        # 获取文件名
                        ref_audio_name = os.path.basename(ref_audio)
                        # 复制音频文件到角色目录
                        if os.path.exists(ref_audio):
                            new_path = role_dir / ref_audio_name
                            if ref_audio != str(new_path):
                                import shutil
                                shutil.copy2(ref_audio, new_path)
                        # 保存为相对路径
                        emotion_config["ref_audio"] = ref_audio_name
                
                # 处理辅助参考音频
                if "aux_refs" in emotion_config:
                    aux_refs = []
                    for aux_ref in emotion_config["aux_refs"]:
                        if os.path.isabs(aux_ref):
                            # 获取文件名
                            aux_ref_name = os.path.basename(aux_ref)
                            # 复制音频文件到角色目录
                            if os.path.exists(aux_ref):
                                new_path = role_dir / aux_ref_name
                                if aux_ref != str(new_path):
                                    import shutil
                                    shutil.copy2(aux_ref, new_path)
                            # 保存为相对路径
                            aux_refs.append(aux_ref_name)
                        else:
                            aux_refs.append(aux_ref)
                    emotion_config["aux_refs"] = aux_refs
            
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config_copy, f, ensure_ascii=False, indent=4)
            
            # 更新内存中的配置
            self.roles[role_name] = config
            return True
        except Exception as e:
            print(f"保存角色配置失败: {role_name}, 错误: {str(e)}")
            return False


class InferenceModel:
    """推理模型类，用于处理语音合成请求"""
    
    def __init__(self, output_dir: str = "output"):
        """
        初始化推理模型
        
        参数:
            output_dir: 输出音频文件目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.inference_engine: Optional[GPTSoVITSInference] = None
        self.history: List[Dict] = []
        self.current_gpt_path: str = ""
        self.current_sovits_path: str = ""
    
    def reset_engine(self):
        """重置推理引擎"""
        if self.inference_engine is not None:
            del self.inference_engine
            self.inference_engine = None
            self.current_gpt_path = ""
            self.current_sovits_path = ""
            # 强制垃圾回收以释放GPU内存
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
    
    def initialize_engine(
        self, 
        gpt_path: str, 
        sovits_path: str, 
        device: str = None, 
        half: bool = True
    ) -> bool:
        """
        初始化推理引擎
        
        参数:
            gpt_path: GPT模型路径
            sovits_path: SoVITS模型路径
            device: 计算设备，默认自动选择
            half: 是否使用半精度
            
        返回:
            初始化是否成功
        """
        # 如果模型路径变化，需要重置引擎
        if (self.inference_engine is not None and 
            (gpt_path != self.current_gpt_path or sovits_path != self.current_sovits_path)):
            self.reset_engine()
            
        try:
            if self.inference_engine is None:
                import torch
                self.inference_engine = GPTSoVITSInference(
                    gpt_path=gpt_path,
                    sovits_path=sovits_path,
                    device=device,
                    half=half
                )
                # 保存当前使用的模型路径
                self.current_gpt_path = gpt_path
                self.current_sovits_path = sovits_path
            return True
        except Exception as e:
            print(f"初始化推理引擎失败: {str(e)}")
            return False
    
    def generate_speech(self, config: Dict) -> Tuple[bool, str]:
        """
        生成语音
        
        参数:
            config: 推理配置
            
        返回:
            (是否成功, 输出文件路径)
        """
        gpt_path = config.get("gpt_path")
        sovits_path = config.get("sovits_path")
        
        if not gpt_path or not sovits_path:
            return False, "未指定模型路径"
        
        # 检查模型路径是否变化
        if (self.inference_engine is not None and 
            (gpt_path != self.current_gpt_path or sovits_path != self.current_sovits_path)):
            self.reset_engine()
                
        # 初始化或重新初始化推理引擎
        success = self.initialize_engine(gpt_path, sovits_path)
        if not success:
            return False, "初始化推理引擎失败"
        
        try:
            # 生成语音
            sample_rate, audio_data = self.inference_engine.generate_speech(
                ref_wav_path=config.get("ref_audio", ""),
                prompt_text=config.get("prompt_text", ""),
                prompt_language=config.get("prompt_lang", "中文"),
                text=config.get("text", ""),
                text_language=config.get("text_lang", "中文"),
                how_to_cut=config.get("how_to_cut", "凑四句一切"),
                top_k=config.get("top_k", 20),
                top_p=config.get("top_p", 0.6),
                temperature=config.get("temperature", 0.6),
                ref_free=config.get("ref_free", False),
                speed=config.get("speed", 1.0),
                inp_refs=config.get("aux_refs", []),
                sample_steps=config.get("sample_steps", 8),
                if_sr=config.get("if_sr", False),
                pause_second=config.get("pause_second", 0.3),
            )
            
            # 生成唯一文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"{timestamp}_{unique_id}.wav"
            output_path = self.output_dir / filename
            
            # 保存音频
            sf.write(output_path, audio_data, sample_rate)
            
            # 添加到历史记录
            history_entry = {
                "timestamp": timestamp,
                "path": str(output_path),
                "config": {k: v for k, v in config.items() if k != "aux_refs"},
                "text": config.get("text", "")
            }
            self.history.append(history_entry)
            
            return True, str(output_path)
        except Exception as e:
            print(f"生成语音失败: {str(e)}")
            return False, f"生成语音失败: {str(e)}"
    
    def get_history(self) -> List[Dict]:
        """获取历史记录"""
        return self.history 