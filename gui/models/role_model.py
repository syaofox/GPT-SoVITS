"""
角色模型

管理角色和情感配置的加载和保存
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Union, Tuple


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
    
    def _process_audio_path_for_loading(self, audio_path: str, role_dir: Path) -> str:
        """
        处理音频路径为加载配置准备，将相对路径转换为绝对路径
        
        参数:
            audio_path: 音频文件路径
            role_dir: 角色目录
            
        返回:
            处理后的音频路径
        """
        if not os.path.isabs(audio_path) and "/" not in audio_path and "\\" not in audio_path:
            # 如果只是文件名而不是路径，则添加角色目录路径
            return str(role_dir / audio_path)
        elif not os.path.isabs(audio_path):
            # 处理可能已包含部分路径的情况
            return str(role_dir / os.path.basename(audio_path))
        return audio_path
    
    def _process_audio_path_for_saving(self, audio_path: str, role_dir: Path) -> str:
        """
        处理音频路径为保存配置准备，将绝对路径转换为相对路径，并复制文件
        
        参数:
            audio_path: 音频文件路径
            role_dir: 角色目录
            
        返回:
            处理后的音频文件名（相对路径）
        """
        if os.path.isabs(audio_path):
            # 获取文件名
            audio_name = os.path.basename(audio_path)
            # 复制音频文件到角色目录
            if os.path.exists(audio_path):
                new_path = role_dir / audio_name
                # 检查源文件和目标文件是否相同，或目标是否已存在
                if os.path.abspath(audio_path) != os.path.abspath(new_path):
                    # 源文件和目标文件不同，才进行复制
                    try:
                        shutil.copy2(audio_path, new_path)
                    except Exception as e:
                        print(f"复制音频文件失败，但继续处理: {str(e)}")
            # 返回文件名（相对路径）
            return audio_name
        else:
            # 对于已经是相对路径的文件，确保只保留文件名
            return os.path.basename(audio_path)
    
    def _process_aux_refs_for_loading(self, aux_refs: List[str], role_dir: Path) -> List[str]:
        """处理辅助参考音频列表为加载配置准备"""
        processed_refs = []
        for aux_ref in aux_refs:
            processed_refs.append(self._process_audio_path_for_loading(aux_ref, role_dir))
        return processed_refs
    
    def _process_aux_refs_for_saving(self, aux_refs: List[str], role_dir: Path) -> List[str]:
        """处理辅助参考音频列表为保存配置准备"""
        processed_refs = []
        for aux_ref in aux_refs:
            processed_refs.append(self._process_audio_path_for_saving(aux_ref, role_dir))
        return processed_refs
    
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
                        emotion_config["ref_audio"] = self._process_audio_path_for_loading(
                            emotion_config["ref_audio"], role_dir
                        )
                    
                    # 处理辅助参考音频
                    if "aux_refs" in emotion_config:
                        emotion_config["aux_refs"] = self._process_aux_refs_for_loading(
                            emotion_config["aux_refs"], role_dir
                        )
                    
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
    
    def get_role_config(self, role_name: str) -> Dict:
        """获取角色的完整配置"""
        return self.roles.get(role_name, {"emotions": {}})
    
    def save_role_config(self, role_name: str, config: Dict) -> bool:
        """
        保存角色配置
        
        如果角色已存在，会合并情感配置而不是完全覆盖
        """
        role_dir = self.roles_dir / role_name
        role_dir.mkdir(exist_ok=True)
        
        config_file = role_dir / "config.json"
        
        try:
            # 检查角色是否已存在，并读取现有配置
            existing_config = {}
            if role_name in self.roles:
                existing_config = self.roles[role_name]
            
            # 合并情感配置
            merged_config = existing_config.copy()
            
            # 确保emotions键存在
            if "emotions" not in merged_config:
                merged_config["emotions"] = {}
                
            # 深拷贝配置，以防意外修改
            new_emotions = json.loads(json.dumps(config.get("emotions", {})))
            
            # 合并新的情感配置到现有配置
            for emotion_name, emotion_config in new_emotions.items():
                merged_config["emotions"][emotion_name] = emotion_config
            
            # 处理音频文件路径，将绝对路径转为相对路径保存
            for emotion_name, emotion_config in merged_config.get("emotions", {}).items():
                if "ref_audio" in emotion_config:
                    emotion_config["ref_audio"] = self._process_audio_path_for_saving(
                        emotion_config["ref_audio"], role_dir
                    )
                
                # 处理辅助参考音频
                if "aux_refs" in emotion_config:
                    emotion_config["aux_refs"] = self._process_aux_refs_for_saving(
                        emotion_config["aux_refs"], role_dir
                    )
            
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(merged_config, f, ensure_ascii=False, indent=4)
            
            # 更新内存中的配置
            self.roles[role_name] = merged_config
            return True
        except Exception as e:
            print(f"保存角色配置失败: {role_name}, 错误: {str(e)}")
            return False 