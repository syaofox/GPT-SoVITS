"""
角色配置模型
负责处理角色和音色配置的数据和业务逻辑
"""
import os
import json
import shutil
from pathlib import Path

class RoleModel:
    """角色配置模型类"""
    
    def __init__(self):
        """初始化角色模型"""
        self.role_dir = Path("configs/roles")
        self.del_role_dir = Path("configs/del_roles")
        self.model_dir = Path("models/GPT_SoVITS/pretrained_models")  # 添加模型目录
        self.current_role = None
        self.current_config = {}
        
        # 确保角色目录存在
        if not self.role_dir.exists():
            self.role_dir.mkdir(parents=True)
        
        # 确保删除的角色目录存在
        if not self.del_role_dir.exists():
            self.del_role_dir.mkdir(parents=True)
    
    def get_role_list(self):
        """获取角色列表"""
        role_list = []
        if self.role_dir.exists():
            for role_path in self.role_dir.iterdir():
                if role_path.is_dir():
                    role_list.append(role_path.name)
        return role_list
    
    def scan_model_dir(self):
        """扫描模型目录获取所有可用角色"""
        role_list = []
        
        # 确保模型目录存在
        if not self.model_dir.exists() or not self.model_dir.is_dir():
            return role_list
            
        # 获取所有子目录，每个子目录代表一个角色
        for role_path in self.model_dir.iterdir():
            if role_path.is_dir():
                # 检查必要的模型文件是否存在
                required_files = ["kmeans.bin", "model.ckpt"]
                if all(os.path.exists(role_path / f) for f in required_files):
                    role_list.append(role_path.name)
        
        return sorted(role_list)  # 返回按字母排序的角色列表
    
    def sync_roles_with_models(self):
        """同步模型目录和角色配置目录"""
        # 从模型目录获取角色列表
        model_roles = self.scan_model_dir()
        # 从配置目录获取角色列表
        config_roles = self.get_role_list()
        
        # 创建不存在的角色配置
        new_roles = []
        for role in model_roles:
            if role not in config_roles:
                # 为此角色创建配置
                if self.create_role(role):
                    new_roles.append(role)
        
        # 返回所有角色和新创建的角色
        return {
            "all_roles": sorted(self.get_role_list()),
            "new_roles": new_roles
        }
    
    def get_role_config(self, role_name):
        """获取角色配置"""
        config_path = self.role_dir / role_name / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                # 确保config中包含角色名称
                config["name"] = role_name
                self.current_role = role_name
                self.current_config = config
                return config
            except Exception as e:
                print(f"加载角色配置出错: {str(e)}")
                return None
        else:
            # 创建新的配置
            config = {
                "name": role_name,  # 添加角色名称
                "version": "v3",
                "emotions": {},
                "text_lang": "中文",
                "prompt_lang": "中文",
                "gpt_path": "",
                "sovits_path": "",
                "speed": 1.0,
                "ref_free": False,
                "if_sr": False,
                "top_k": 15,
                "top_p": 1.0,
                "temperature": 1.0,
                "sample_steps": 32,
                "pause_second": 0.3,
                "description": ""
            }
            self.current_role = role_name
            self.current_config = config
            return config
    
    def save_role_config(self, role_name, config):
        """保存角色配置"""
        role_path = self.role_dir / role_name
        if not role_path.exists():
            role_path.mkdir(parents=True)
        
        config_path = role_path / "config.json"
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            return True
        except Exception as e:
            print(f"保存角色配置出错: {str(e)}")
            return False
    
    def create_role(self, role_name):
        """创建新角色"""
        role_path = self.role_dir / role_name
        if role_path.exists():
            return False
        
        # 创建角色目录
        role_path.mkdir(parents=True)
        
        # 创建默认配置
        default_config = {
            "name": role_name,
            "version": "v3",
            "emotions": {},
            "text_lang": "中文",
            "prompt_lang": "中文",
            "gpt_path": "",
            "sovits_path": "",
            "speed": 1.0,
            "ref_free": False,
            "if_sr": False,
            "top_k": 15,
            "top_p": 1.0,
            "temperature": 1.0,
            "sample_steps": 32,
            "pause_second": 0.3,
            "description": ""
        }
        
        # 保存默认配置
        config_path = role_path / "config.json"
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(default_config, f, ensure_ascii=False, indent=4)
            
            # 设置为当前角色和配置
            self.current_role = role_name
            self.current_config = default_config
            return True
        except Exception as e:
            print(f"创建角色配置出错: {str(e)}")
            # 如果保存配置失败，删除已创建的目录
            if role_path.exists():
                shutil.rmtree(role_path)
            return False
    
    def delete_role(self, role_name):
        """删除角色（移动到回收站）"""
        src_path = self.role_dir / role_name
        dst_path = self.del_role_dir / role_name
        
        if not src_path.exists():
            return False
        
        try:
            # 如果目标存在，先删除
            if dst_path.exists():
                shutil.rmtree(dst_path)
            
            # 移动目录
            shutil.move(str(src_path), str(dst_path))
            
            # 如果是当前角色，清空当前配置
            if self.current_role == role_name:
                self.current_role = None
                self.current_config = {}
            
            return True
        except Exception as e:
            print(f"删除角色出错: {str(e)}")
            return False
    
    def import_role(self, zip_path):
        """导入角色"""
        import zipfile
        import tempfile
        
        try:
            # 创建临时目录
            with tempfile.TemporaryDirectory() as temp_dir:
                # 解压文件
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # 查找config.json
                temp_path = Path(temp_dir)
                config_files = list(temp_path.glob("**/config.json"))
                
                if not config_files:
                    return None
                
                # 获取角色名称
                config_path = config_files[0]
                role_name = config_path.parent.name
                
                # 检查角色是否已存在
                target_path = self.role_dir / role_name
                if target_path.exists():
                    return {"exists": True, "role_name": role_name}
                
                # 复制文件
                shutil.copytree(config_path.parent, target_path)
                
                return {"exists": False, "role_name": role_name}
        except Exception as e:
            print(f"导入角色出错: {str(e)}")
            return None
    
    def export_role(self, role_name, zip_path):
        """导出角色"""
        try:
            import zipfile
            
            role_path = self.role_dir / role_name
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(role_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, self.role_dir)
                        zipf.write(file_path, arcname)
            
            return True
        except Exception as e:
            print(f"导出角色出错: {str(e)}")
            return False
    
    def add_emotion(self, role_name, emotion_name, emotion_config=None):
        """添加音色"""
        if not self.current_config:
            self.get_role_config(role_name)
        
        if not self.current_config:
            return False
        
        emotions = self.current_config.get("emotions", {})
        
        # 如果未提供配置，创建默认配置
        if emotion_config is None:
            emotion_config = {
                "ref_audio": "",
                "prompt_text": "",
                "aux_refs": []
            }
        
        # 添加或更新音色
        emotions[emotion_name] = emotion_config
        self.current_config["emotions"] = emotions
        return True
    
    def delete_emotion(self, emotion_name):
        """删除音色"""
        if not self.current_config:
            return False
        
        emotions = self.current_config.get("emotions", {})
        if emotion_name not in emotions:
            return False
        
        del emotions[emotion_name]
        self.current_config["emotions"] = emotions
        return True
    
    def update_emotion(self, old_name, new_name, emotion_config):
        """更新音色配置"""
        if not self.current_config:
            return False
        
        emotions = self.current_config.get("emotions", {})
        
        # 如果音色名称已更改，需要重命名
        if old_name != new_name and new_name:
            if new_name in emotions and old_name != new_name:
                return False
            
            # 删除旧配置
            if old_name in emotions:
                del emotions[old_name]
        
        # 更新配置
        emotions[new_name] = emotion_config
        self.current_config["emotions"] = emotions
        return True
    
    def get_role_path(self, role_name):
        """获取角色路径"""
        return self.role_dir / role_name
    
    def get_ref_audio_path(self, role_name, ref_audio):
        """获取参考音频路径"""
        return self.role_dir / role_name / ref_audio 