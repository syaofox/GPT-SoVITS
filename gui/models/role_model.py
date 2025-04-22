"""
角色模型

管理角色和情感配置的加载和保存
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict


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
                        # 如果只是文件名而不是路径，则添加角色目录路径
                        if not os.path.isabs(ref_audio) and "/" not in ref_audio and "\\" not in ref_audio:
                            emotion_config["ref_audio"] = str(role_dir / ref_audio)
                        elif not os.path.isabs(ref_audio):
                            # 处理可能已包含部分路径的情况
                            emotion_config["ref_audio"] = str(role_dir / os.path.basename(ref_audio))
                    
                    # 处理辅助参考音频
                    if "aux_refs" in emotion_config:
                        aux_refs = []
                        for aux_ref in emotion_config["aux_refs"]:
                            if not os.path.isabs(aux_ref) and "/" not in aux_ref and "\\" not in aux_ref:
                                # 如果只是文件名，添加角色目录路径
                                aux_refs.append(str(role_dir / aux_ref))
                            elif not os.path.isabs(aux_ref):
                                # 处理可能已包含部分路径的情况
                                aux_refs.append(str(role_dir / os.path.basename(aux_ref)))
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
                    ref_audio = emotion_config["ref_audio"]
                    if os.path.isabs(ref_audio):
                        # 获取文件名
                        ref_audio_name = os.path.basename(ref_audio)
                        # 复制音频文件到角色目录
                        if os.path.exists(ref_audio):
                            new_path = role_dir / ref_audio_name
                            if ref_audio != str(new_path):
                                shutil.copy2(ref_audio, new_path)
                        # 保存为相对路径（只保存文件名，不包含目录）
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
                                    shutil.copy2(aux_ref, new_path)
                            # 保存为相对路径（只保存文件名，不包含目录）
                            aux_refs.append(aux_ref_name)
                        else:
                            # 对于已经是相对路径的文件，确保只保留文件名
                            aux_ref_name = os.path.basename(aux_ref)
                            aux_refs.append(aux_ref_name)
                    emotion_config["aux_refs"] = aux_refs
            
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(merged_config, f, ensure_ascii=False, indent=4)
            
            # 更新内存中的配置
            self.roles[role_name] = merged_config
            return True
        except Exception as e:
            print(f"保存角色配置失败: {role_name}, 错误: {str(e)}")
            return False 